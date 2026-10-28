"""Stage 1: train models once, save per-row predictions for source and target site.
usage: python train_predict.py <A->B|B->A> <cache dir> <out dir> <seeds...>
Models: gbm_values, gbm_mask, gbm_both, gbm_temporal (hist-GBM, hyper-parameters tuned once per direction
on an inner validation split of the seed-0 training patients), mlp_temporal (sklearn MLP).
Also saves covariate-shift density-ratio weights for source rows (domain classifier trained on
source-TRAIN vs unlabeled target-POOL rows; target test rows never used).
"""
import sys, os, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits, row_mask

direction, cache, out = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = [int(s) for s in sys.argv[4:]]
os.makedirs(out, exist_ok=True)
S, T = direction.split("->")
tag = direction.replace("->", "to")

def load(site):
    X = np.load(f"{cache}/{site}_X.npy", mmap_mode="r"); m = np.load(f"{cache}/{site}_meta.npz")
    return X, json.load(open(f"{cache}/{site}_cols.json")), m
Xs, cols, ms = load(S); Xt, _, mt = load(T)
ns, nt = int(ms["loc"].max()) + 1, int(mt["loc"].max()) + 1
ys = ms["y"].astype(int)
base = [c for c in cols if c[:2] in ("v_", "m_", "d_")]
FS = {"values": [c for c in cols if c.startswith("v_")] + ["Age", "Gender"],
      "mask": [c for c in cols if c[:2] in ("m_", "d_")] + ["Age", "Gender"],
      "both": base + ["Age", "Gender"],
      "temporal": base + ["Age", "Gender"] + [c for c in cols if c.startswith("tmp_")]}
_samp = np.asarray(Xs[::5])                       # drop columns with <2 distinct finite values in the SOURCE site
_ok = {c for j, c in enumerate(cols) if len(np.unique(_samp[:, j][np.isfinite(_samp[:, j])])) >= 2}
DROPPED = [c for c in cols if c not in _ok]
print("dropped constant source columns:", DROPPED, flush=True)
idx = {k: [cols.index(c) for c in v if c in _ok] for k, v in FS.items()}

def predict(clf, X, ix, chunk=150_000):
    return np.concatenate([clf.predict_proba(np.asarray(X[i:i + chunk])[:, ix])[:, 1]
                           for i in range(0, X.shape[0], chunk)]).astype(np.float32)

# ---------------------------------------------------------------- tuning (once per direction, seed 0)
hp_path = f"{out}/hp_{tag}.json"
if os.path.exists(hp_path): hp = json.load(open(hp_path))
else:
    sp0 = make_splits(ns, nt, 0); r = np.random.default_rng(123)
    tr_pat = sp0["train"]; k = int(.8 * len(tr_pat)); a, b = tr_pat[:k], tr_pat[k:]
    ra, rb = row_mask(ms["loc"], ns, a), row_mask(ms["loc"], ns, b)
    Xa, Xb = np.asarray(Xs[ra])[:, idx["both"]], np.asarray(Xs[rb])[:, idx["both"]]
    best = (-1, None)
    for i in range(6):
        cfg = dict(max_leaf_nodes=int(r.choice([15, 31, 63])), learning_rate=float(r.choice([0.05, 0.1])),
                   l2_regularization=float(r.choice([0.0, 1.0, 10.0])), min_samples_leaf=int(r.choice([20, 100, 500])))
        t0 = time.time()
        clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, n_iter_no_change=15,
                                             random_state=0, **cfg).fit(Xa, ys[ra])
        ap = average_precision_score(ys[rb], clf.predict_proba(Xb)[:, 1])
        print(f"tune {i} {cfg} AUPRC={ap:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ap > best[0]: best = (ap, cfg)
    hp = best[1]; hp["tuning_auprc"] = best[0]; json.dump(hp, open(hp_path, "w"))
print("hyper-parameters:", hp, flush=True)
hpc = {k: v for k, v in hp.items() if k != "tuning_auprc"}

# ---------------------------------------------------------------- per-seed training
for seed in seeds:
    path = f"{out}/{tag}_s{seed}.npz"
    if os.path.exists(path): continue
    t0 = time.time(); spl = make_splits(ns, nt, seed)
    rtr = row_mask(ms["loc"], ns, spl["train"]); ytr = ys[rtr]
    res = {}
    for name in ["values", "mask", "both", "temporal"]:
        Xtr = np.asarray(Xs[rtr])[:, idx[name]]
        clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, n_iter_no_change=15,
                                             random_state=seed, **hpc).fit(Xtr, ytr)
        res[f"gbm_{name}_src"] = predict(clf, Xs, idx[name]); res[f"gbm_{name}_tgt"] = predict(clf, Xt, idx[name])
        print(f"seed {seed} gbm_{name} done ({time.time()-t0:.0f}s)", flush=True)
    # MLP on temporal features: impute with train mean, standardise
    Xtr = np.asarray(Xs[rtr])[:, idx["temporal"]]
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-6
    prep = lambda A: np.nan_to_num((A - mu) / sd, nan=0.0).clip(-8, 8).astype(np.float32)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, early_stopping=True, n_iter_no_change=3,
                        max_iter=20, random_state=seed).fit(prep(Xtr), ytr)
    pm = lambda X: np.concatenate([mlp.predict_proba(prep(np.asarray(X[i:i + 150_000])[:, idx["temporal"]]))[:, 1]
                                   for i in range(0, X.shape[0], 150_000)]).astype(np.float32)
    res["mlp_src"], res["mlp_tgt"] = pm(Xs), pm(Xt)
    print(f"seed {seed} mlp done ({time.time()-t0:.0f}s)", flush=True)
    # domain classifier -> density-ratio weights on source rows (features: base + Age/Gender, no admin vars)
    rpool = row_mask(mt["loc"], nt, spl["pool"]); r = np.random.default_rng(seed)
    a = r.choice(np.where(rtr)[0], size=min(120_000, rtr.sum()), replace=False); a.sort()
    b = r.choice(np.where(rpool)[0], size=min(120_000, rpool.sum()), replace=False); b.sort()
    Xd = np.vstack([np.asarray(Xs[a])[:, idx["both"]], np.asarray(Xt[b])[:, idx["both"]]])
    yd = np.r_[np.zeros(len(a)), np.ones(len(b))]
    dom = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed).fit(Xd, yd)
    pd_ = np.clip(predict(dom, Xs, idx["both"]), 1e-4, 1 - 1e-4)
    res["dr_src"] = (pd_ / (1 - pd_) * (len(a) / len(b))).astype(np.float32)
    np.savez_compressed(path, **res)
    print(f"seed {seed} saved ({time.time()-t0:.0f}s)", flush=True)
