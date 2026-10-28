"""Stage 2: rigorous evaluation from saved predictions (task 4 + task 5 comparison).

For every direction (A->B, B->A), model, and test set (in-site / other-site):
  * row-level AUROC, AUPRC (patient-level bootstrap)
  * calibration (models only): slope, intercept, Brier, O:E  (isotonic recalibration fitted on source 'iso' patients)
  * normalised PhysioNet/CinC 2019 Utility (re-implementation; threshold chosen on source 'cp' patients)
  * alert-burden operating points (threshold set on source 'cp' non-septic rows for a target number of
    alert-hours per patient-day): patient-level sensitivity before onset, lead time, non-septic patient alarm rate
Confidence intervals: 95% percentile interval of (mean over seeds of per-seed patient-level bootstrap replicates).

usage: python evaluate_rigor.py --cache C --preds P --out O --seeds 0 1 2 3 4 [--B 200]
"""
import sys, os, json, argparse, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (make_splits, row_mask, utility_arrays, util_norm_total, pick_threshold, make_bins, hist_csr,
                    auc_ap, fit_logit_binned, wmedian, logit, STD_CUTOFF)
from helpers2 import iso_fit

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/home/claude/cache"); ap.add_argument("--preds", default="/home/claude/preds")
ap.add_argument("--out", default="/home/claude/results"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--B", type=int, default=200); ap.add_argument("--dirs", nargs="+", default=["A->B", "B->A"])
ap.add_argument("--models", nargs="+", default=["gbm_values", "gbm_mask", "gbm_both", "gbm_temporal", "mlp"])
ap.add_argument("--no_scores", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

MODELS = a.models
SCORES = [] if a.no_scores else ["sirs", "qsofa", "news", "sofa"]
STD_CUTOFF = dict(STD_CUTOFF, sirs=2)
ALERT_HOURS = [0.5, 1.0, 2.0]            # target alert-hours per non-septic patient-day

def load_meta(site):
    m = np.load(f"{a.cache}/{site}_meta.npz"); return {k: m[k] for k in m.files if k != "codes"}
META = {s: load_meta(s) for s in "AB"}

def subset(meta, rowmask):
    loc_full = meta["loc"][rowmask]; u, loc = np.unique(loc_full, return_inverse=True)
    return dict(rows=np.where(rowmask)[0], loc=loc, n=len(u), t=meta["t"][rowmask], y=meta["y"][rowmask].astype(int))

def boot_weights(n, B, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)

def ratio(W, num, den):
    return float(num.sum() / den.sum()) if den.sum() > 0 else np.nan, (W @ num) / np.maximum(W @ den, 1e-12)

def disc_boot(W, S, s):
    """AUROC/AUPRC point + replicates from patient-binned histograms."""
    b, nb = make_bins(s); y = S["y"]; n = S["n"]
    Hp, Hn = hist_csr(S["loc"], b, y == 1, n, nb), hist_csr(S["loc"], b, y == 0, n, nb)
    p0 = np.asarray(Hp.sum(0)).ravel(); n0 = np.asarray(Hn.sum(0)).ravel()
    out = {"auroc": [auc_ap(p0, n0)[0]], "auprc": [auc_ap(p0, n0)[1]]}
    PB, NB = (Hp.T @ W.T).T, (Hn.T @ W.T).T
    r = np.array([auc_ap(PB[i], NB[i]) for i in range(len(W))])
    return {"auroc": (out["auroc"][0], r[:, 0]), "auprc": (out["auprc"][0], r[:, 1])}

def calib_boot(W, S, p):
    b, nb = make_bins(p); y = S["y"]; n = S["n"]
    cnt = np.bincount(b, minlength=nb).astype(float); mp = np.bincount(b, weights=p, minlength=nb) / np.maximum(cnt, 1)
    z = logit(mp)
    Hp, Hn = hist_csr(S["loc"], b, y == 1, n, nb), hist_csr(S["loc"], b, y == 0, n, nb)
    p0 = np.asarray(Hp.sum(0)).ravel(); n0 = np.asarray(Hn.sum(0)).ravel()
    ai, sl = fit_logit_binned(z, p0, n0)
    PB, NB = (Hp.T @ W.T).T, (Hn.T @ W.T).T
    ab = np.array([fit_logit_binned(z, PB[i], NB[i]) for i in range(len(W))])
    pp = np.bincount(S["loc"], weights=p, minlength=n); yy = np.bincount(S["loc"], weights=y, minlength=n)
    sq = np.bincount(S["loc"], weights=(p - y) ** 2, minlength=n); nr = np.bincount(S["loc"], minlength=n).astype(float)
    return {"cal_slope": (sl, ab[:, 1]), "cal_intercept": (ai, ab[:, 0]),
            "brier": ratio(W, sq, nr), "oe": ratio(W, yy, pp)}

def utility_boot(W, S, ua, pred):
    n, loc = S["n"], S["loc"]
    uo = np.bincount(loc, weights=np.where(pred, ua["u1"], ua["u0"]), minlength=n)
    ui = np.bincount(loc, weights=ua["u0"], minlength=n); ub = np.bincount(loc, weights=ua["best"], minlength=n)
    pt = (uo.sum() - ui.sum()) / (ub.sum() - ui.sum())
    return pt, (W @ uo - W @ ui) / np.maximum(W @ ub - W @ ui, 1e-12)

def patient_ops(W, S, ua, alert):
    """Patient-level alert-burden metrics at a fixed alert rule."""
    n, loc, t = S["n"], S["loc"], S["t"]
    sep_p = np.zeros(n, bool); sep_p[loc[ua["septic"]]] = True
    ns = ~ua["septic"]
    pre = alert & ua["septic"] & (ua["rel"] < 0)                      # alerts strictly before onset
    first = np.full(n, np.inf); np.minimum.at(first, loc[pre], ua["rel"][pre])   # most negative rel = earliest
    lead = np.where(np.isfinite(first), -first, np.nan)                # hours before onset of first alert
    det = np.isfinite(first).astype(float)
    nsany = (np.bincount(loc[alert & ns], minlength=n) > 0).astype(float)
    nsal = np.bincount(loc[alert & ns], minlength=n).astype(float); nsr = np.bincount(loc[ns], minlength=n).astype(float)
    sepf, nsp = sep_p.astype(float), (~sep_p).astype(float)
    out = {"pat_sens": ratio(W, det, sepf), "pat_sens_lead6": ratio(W, (lead >= 6).astype(float), sepf),
           "pat_sens_lead12": ratio(W, (lead >= 12).astype(float), sepf),
           "nonsep_pat_alarm_rate": ratio(W, nsany * nsp, nsp), "alert_hours_per_day": (None, None)}
    pt, rep = ratio(W, nsal, nsr); out["alert_hours_per_day"] = (24 * pt, 24 * rep)
    d = np.isfinite(lead); med0 = float(np.median(lead[d])) if d.any() else np.nan
    reps = np.array([wmedian(lead[d], W[i][d]) for i in range(len(W))]) if d.any() else np.full(len(W), np.nan)
    out["median_lead_h"] = (med0, reps)
    return out

store = collections.defaultdict(list)       # key -> list over seeds of (point, reps)
seed_pts = collections.defaultdict(list)
def put(key, d):
    for m, (pt, rp) in d.items():
        store[key + (m,)].append((pt, np.asarray(rp)))

reliab = collections.defaultdict(lambda: np.zeros((8, 3)))   # gbm_both reliability counts (n, sum p, sum y)
EDGES = np.array([0, .005, .01, .02, .04, .08, .16, 1.0001])
subg = []

for direction in a.dirs:
    S_, T_ = direction.split("->"); tag = direction.replace("->", "to")
    ms, mt = META[S_], META[T_]; ns, nt = int(ms["loc"].max()) + 1, int(mt["loc"].max()) + 1
    X_s = np.load(f"{a.cache}/{S_}_X.npy", mmap_mode="r"); X_t = np.load(f"{a.cache}/{T_}_X.npy", mmap_mode="r")
    cols_s = json.load(open(f"{a.cache}/{S_}_cols.json")); cols_t = json.load(open(f"{a.cache}/{T_}_cols.json"))
    sg = lambda X, cols, c: np.asarray(X[:, cols.index(c)])
    SG = {"src": dict(age=sg(X_s, cols_s, "Age"), gender=sg(X_s, cols_s, "Gender"), u1=sg(X_s, cols_s, "Unit1"), u2=sg(X_s, cols_s, "Unit2")),
          "tgt": dict(age=sg(X_t, cols_t, "Age"), gender=sg(X_t, cols_t, "Gender"), u1=sg(X_t, cols_t, "Unit1"), u2=sg(X_t, cols_t, "Unit2"))}
    for seed in a.seeds:
        f = f"{a.preds}/{tag}_s{seed}.npz"
        if not os.path.exists(f): print("missing", f); continue
        P = np.load(f); spl = make_splits(ns, nt, seed); rng = np.random.default_rng(10_000 + seed)
        R = {k: row_mask(ms["loc"], ns, spl[k]) for k in ["iso", "cp", "test_in"]}
        R["test_out"] = row_mask(mt["loc"], nt, spl["test_out"])
        Scp = subset(ms, R["cp"]); uacp = utility_arrays(Scp["loc"], Scp["t"], Scp["y"])
        tests = {"in_site": (subset(ms, R["test_in"]), "src"), "other_site": (subset(mt, R["test_out"]), "tgt")}
        Wd = {k: boot_weights(v[0]["n"], a.B, rng) for k, v in tests.items()}
        UA = {k: utility_arrays(v[0]["loc"], v[0]["t"], v[0]["y"]) for k, v in tests.items()}
        for name in MODELS + SCORES:
            is_model = name in MODELS
            get = lambda side: (P[f"{name}_{side}"] if is_model else (ms if side == "src" else mt)[name]).astype(np.float64)
            s_src, s_tgt = get("src"), get("tgt")
            if is_model:
                iso = iso_fit(s_src[R["iso"]], ms["y"][R["iso"]]); cal_src, cal_tgt = iso.predict(s_src), iso.predict(s_tgt)
            # thresholds fixed on source 'cp' patients
            s_cp = s_src[R["cp"]]
            thr_u = pick_threshold(s_cp, uacp)
            ns_rows = ~uacp["septic"]
            thr_ops = {h: np.quantile(s_cp[ns_rows], max(1 - h / 24.0, 0)) for h in ALERT_HOURS}   # alert-hours/day -> rate
            for tkey, (S, side) in tests.items():
                s = (s_src if side == "src" else s_tgt)[S["rows"]]; W = Wd[tkey]; ua = UA[tkey]
                key = (direction, name, tkey)
                put(key, disc_boot(W, S, s))
                put(key, {"utility_opt": utility_boot(W, S, ua, s >= thr_u)})
                if name in STD_CUTOFF: put(key, {"utility_std": utility_boot(W, S, ua, s >= STD_CUTOFF[name])})
                if is_model:
                    p = (cal_src if side == "src" else cal_tgt)[S["rows"]]
                    put(key, calib_boot(W, S, p))
                    if name == "gbm_both":
                        h = np.digitize(p, EDGES) - 1
                        for j in range(8):
                            m = h == j; reliab[(direction, tkey)][j] += [m.sum(), p[m].sum(), S["y"][m].sum()]
                        gsrc = SG[side]; gi = {k: v[S["rows"]] for k, v in gsrc.items()}
                        for gname, gm in [("age<65", gi["age"] < 65), ("age>=65", gi["age"] >= 65), ("female", gi["gender"] == 0),
                                          ("male", gi["gender"] == 1), ("MICU", gi["u1"] == 1), ("SICU", gi["u2"] == 1)]:
                            if gm.sum() > 500 and S["y"][gm].sum() > 20:
                                from helpers2 import calib_point
                                subg.append(dict(direction=direction, seed=seed, test=tkey, group=gname, n=int(gm.sum()),
                                                 **calib_point(p[gm], S["y"][gm].astype(float))))
                for h in ALERT_HOURS:
                    put(key + (f"ops{h}",), {k: v for k, v in patient_ops(W, S, ua, s >= thr_ops[h]).items()})
                if name in STD_CUTOFF:
                    put(key + ("ops_std",), patient_ops(W, S, ua, s >= STD_CUTOFF[name]))
        print(direction, "seed", seed, "done", flush=True)

# ---------------------------------------------------------------- aggregate
rows = []
for key, lst in store.items():
    pts = np.array([p for p, _ in lst], float); reps = np.mean([r for _, r in lst], axis=0)
    direction, name, tkey = key[0], key[1], key[2]
    op = key[3] if len(key) == 5 else ""
    rows.append(dict(direction=direction, model=name, test=tkey, op=op, metric=key[-1], mean=np.nanmean(pts),
                     ci_lo=np.nanpercentile(reps, 2.5), ci_hi=np.nanpercentile(reps, 97.5),
                     sd_seeds=np.nanstd(pts, ddof=1) if len(pts) > 1 else np.nan, n_seeds=len(pts)))
res = pd.DataFrame(rows).sort_values(["direction", "test", "metric", "op", "model"])
res.to_csv(f"{a.out}/rigor_results.csv", index=False)
if subg:
    pd.DataFrame(subg).groupby(["direction", "test", "group"]).mean(numeric_only=True).reset_index() \
        .to_csv(f"{a.out}/subgroup_calibration.csv", index=False)
rl = [dict(direction=k[0], test=k[1], bin=j, lo=EDGES[j], n=v[j, 0], mean_pred=v[j, 1] / max(v[j, 0], 1),
           obs=v[j, 2] / max(v[j, 0], 1)) for k, v in reliab.items() for j in range(8)]
if rl: pd.DataFrame(rl).to_csv(f"{a.out}/reliability_gbm_both.csv", index=False)
print("saved", f"{a.out}/rigor_results.csv", len(res), "rows")
