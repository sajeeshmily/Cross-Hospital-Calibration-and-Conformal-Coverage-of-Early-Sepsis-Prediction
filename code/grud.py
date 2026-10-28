"""GRU-D baseline (Che et al. 2018) in pure numpy (no torch available in this environment).

Per hour t and variable j:   x_hat = m*x + (1-m)*(gx*x_last + (1-gx)*x_mean),  gx = exp(-relu(wx*delta + bx))
Hidden decay:                h~ = gh*h_prev,  gh = exp(-relu(Wd delta + bd))
GRU update on [x_hat, static, mask];  hourly output sigmoid(wo.h + bo).
x_mean = 0 after standardisation, and the cache stores forward-filled values, so x_hat = x_last * (m + (1-m)*gx).
Hand-written BPTT, verified against finite differences (python grud.py --check).
Static inputs: Age, Gender. Loss: BCE with positive weight, per hour, over all hours.

usage: python grud.py <A->B|B->A> <cache> <out dir> <seeds...>   -> <out>/<tag>_s<seed>.npz with grud_src, grud_tgt
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, "/mnt/attach/outputs")


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRUD:
    def __init__(self, D, H, rng, dtype=np.float32):
        self.D, self.H, self.F = D, H, 2 * D + 2
        F = self.F
        s = lambda *sh, k=1.0: (rng.standard_normal(sh) * k).astype(dtype)
        self.p = dict(wx=np.full(D, 0.1, dtype), bx=np.zeros(D, dtype),
                      Wd=s(D, H, k=0.05), bd=np.zeros(H, dtype),
                      Wi=s(F, 3 * H, k=1 / np.sqrt(F)), Uh=s(H, 2 * H, k=1 / np.sqrt(H)), Uc=s(H, H, k=1 / np.sqrt(H)),
                      b=np.zeros(3 * H, dtype), wo=s(H, k=0.1), bo=np.array(-3.5, dtype))

    def forward(self, dl, ml, xl, st, keep=False):
        """dl, ml, xl: (B,T,D) delta, mask, standardised last value; st: (B,2). Returns logits (B,T) and cache."""
        P, H, D = self.p, self.H, self.D
        B, T, _ = dl.shape
        h = np.zeros((B, H), dl.dtype)
        O = np.zeros((B, T), dl.dtype)
        cache = []
        for t in range(T):
            d, m, x = dl[:, t], ml[:, t], xl[:, t]
            a = P["wx"] * d + P["bx"]
            gx = np.exp(-np.maximum(a, 0))
            xh = x * (m + (1 - m) * gx)
            inp = np.concatenate([xh, st, m], 1)
            ah = d @ P["Wd"] + P["bd"]
            gh = np.exp(-np.maximum(ah, 0))
            ht = gh * h
            Wi, Uh, Uc, b = P["Wi"], P["Uh"], P["Uc"], P["b"]
            ii = inp @ Wi
            z = sig(ii[:, :H] + ht @ Uh[:, :H] + b[:H])
            r = sig(ii[:, H:2 * H] + ht @ Uh[:, H:] + b[H:2 * H])
            c = np.tanh(ii[:, 2 * H:] + (r * ht) @ Uc + b[2 * H:])
            hn = (1 - z) * ht + z * c
            O[:, t] = hn @ P["wo"] + P["bo"]
            if keep:
                cache.append((d, m, x, a, gx, inp, ah, gh, h, ht, z, r, c))
            h = hn
        return O, cache

    def backward(self, cache, dO, hs=None):
        """dO: (B,T) gradient wrt logits. Needs hidden states: recomputed from cache (h at each step)."""
        P, H, D = self.p, self.H, self.D
        G = {k: np.zeros_like(v) for k, v in P.items()}
        T = len(cache)
        dh = np.zeros_like(cache[0][8])
        # h_t (post-update) for output weights = next step's h_prev, last step needs recompute
        hs = [cache[t + 1][8] for t in range(T - 1)]
        z, r, c, ht = cache[-1][10], cache[-1][11], cache[-1][12], cache[-1][9]
        hs.append((1 - z) * ht + z * c)
        for t in range(T - 1, -1, -1):
            d, m, x, a, gx, inp, ah, gh, hp, ht, z, r, c = cache[t]
            G["wo"] += hs[t].T @ dO[:, t]
            G["bo"] += dO[:, t].sum()
            dh = dh + dO[:, t][:, None] * P["wo"][None, :]
            dz = dh * (c - ht)
            dc = dh * z
            dht = dh * (1 - z)
            dpc = dc * (1 - c * c)
            dpz = dz * z * (1 - z)
            Uc, Uh, Wi = P["Uc"], P["Uh"], P["Wi"]
            drht = dpc @ Uc.T
            G["Uc"] += (r * ht).T @ dpc
            dr = drht * ht
            dht = dht + drht * r
            dpr = dr * r * (1 - r)
            G["Wi"][:, :H] += inp.T @ dpz
            G["Wi"][:, H:2 * H] += inp.T @ dpr
            G["Wi"][:, 2 * H:] += inp.T @ dpc
            G["Uh"][:, :H] += ht.T @ dpz
            G["Uh"][:, H:] += ht.T @ dpr
            dht = dht + dpz @ Uh[:, :H].T + dpr @ Uh[:, H:].T
            G["b"][:H] += dpz.sum(0)
            G["b"][H:2 * H] += dpr.sum(0)
            G["b"][2 * H:] += dpc.sum(0)
            dinp = dpz @ Wi[:, :H].T + dpr @ Wi[:, H:2 * H].T + dpc @ Wi[:, 2 * H:].T
            dxh = dinp[:, :D]
            dgx = dxh * x * (1 - m)
            da = dgx * (-gx) * (a > 0)
            G["wx"] += (da * d).sum(0)
            G["bx"] += da.sum(0)
            dgh = dht * hp
            dah = dgh * (-gh) * (ah > 0)
            G["Wd"] += d.T @ dah
            G["bd"] += dah.sum(0)
            dh = dht * gh
        return G


def bce_grad(O, y, valid, posw):
    p = sig(O)
    w = valid * np.where(y > 0, posw, 1.0)
    n = max(valid.sum(), 1.0)
    loss = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() / n
    return loss, w * (p - y) / n


def grad_check():
    rng = np.random.default_rng(0)
    D, H, B, T = 4, 5, 3, 6
    net = GRUD(D, H, rng, np.float64)
    for k in net.p:
        net.p[k] = net.p[k] + 0.3 * rng.standard_normal(np.shape(net.p[k]))  # move off trivial init
    m = (rng.random((B, T, D)) < 0.4).astype(np.float64)
    dl = rng.random((B, T, D)) * 2
    xl = rng.standard_normal((B, T, D))
    st = rng.standard_normal((B, 2))
    y = (rng.random((B, T)) < 0.3).astype(np.float64)
    valid = np.ones((B, T)); valid[0, 4:] = 0

    def L():
        O, _ = net.forward(dl, m, xl, st, keep=False)
        return bce_grad(O, y, valid, 3.0)[0]
    O, cache = net.forward(dl, m, xl, st, keep=True)
    G = net.backward(cache, bce_grad(O, y, valid, 3.0)[1])
    worst = 0.0
    for k, v in net.p.items():
        v = np.atleast_1d(v) if np.ndim(v) == 0 else v
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in np.shape(net.p[k]))
            old = net.p[k][idx] if np.ndim(net.p[k]) else net.p[k]
            eps = 1e-6
            def setv(x):
                if np.ndim(net.p[k]): net.p[k][idx] = x
                else: net.p[k] = np.array(x)
            setv(old + eps); lp = L(); setv(old - eps); lm = L(); setv(old)
            num = (lp - lm) / (2 * eps); ana = G[k][idx] if np.ndim(G[k]) else G[k]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
        print(f"{k:3s} worst rel err so far {worst:.2e}")
    print("PASS" if worst < 1e-5 else "FAIL")


# ---------------------------------------------------------------- data
def load_site(cache, site):
    X = np.load(f"{cache}/{site}_X.npy", mmap_mode="r")
    cols = json.load(open(f"{cache}/{site}_cols.json"))
    m = np.load(f"{cache}/{site}_meta.npz")
    return X, cols, m["loc"], m["y"].astype(np.float32)


def build_arrays(X, cols, vars_, mu, sd, amu, asd):
    ix = lambda pre: [cols.index(pre + v) for v in vars_]
    V = np.asarray(X[:, ix("v_")], dtype=np.float32)
    V = np.nan_to_num((V - mu) / sd, nan=0.0).clip(-8, 8)
    M = np.asarray(X[:, ix("m_")], dtype=np.float32)
    Dl = np.clip(np.expm1(np.asarray(X[:, ix("d_")], dtype=np.float32)) / 48.0, 0, 1)
    S = np.asarray(X[:, [cols.index("Age"), cols.index("Gender")]], dtype=np.float32)
    S = np.nan_to_num((S - amu) / asd, nan=0.0).clip(-8, 8)
    return V, M, Dl, S


def batch_of(pats, starts, lens, V, M, Dl, S, y):
    T = int(lens[pats].max()); B = len(pats); D = V.shape[1]
    bd = np.zeros((B, T, D), np.float32); bm = np.zeros_like(bd); bx = np.zeros_like(bd)
    bs = np.zeros((B, 2), np.float32); by = np.zeros((B, T), np.float32); bv = np.zeros((B, T), np.float32)
    for i, p in enumerate(pats):
        a, n = starts[p], lens[p]
        bd[i, :n], bm[i, :n], bx[i, :n] = Dl[a:a + n], M[a:a + n], V[a:a + n]
        bs[i] = S[a]; by[i, :n] = y[a:a + n]; bv[i, :n] = 1
    return bd, bm, bx, bs, by, bv


def predict_site(net, pats_all, starts, lens, V, M, Dl, S, y, nrows, bs=256):
    out = np.zeros(nrows, np.float32)
    order = pats_all[np.argsort(lens[pats_all])]
    for i in range(0, len(order), bs):
        pb = order[i:i + bs]
        bd, bm, bx, bst, by, bv = batch_of(pb, starts, lens, V, M, Dl, S, y)
        O, _ = net.forward(bd, bm, bx, bst)
        for j, p in enumerate(pb):
            out[starts[p]:starts[p] + lens[p]] = sig(O[j, :lens[p]])
    return out


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)


def fit(net, rng, tr, va, V, M, Dl, S, y, ls, starts, lens, ns, epochs, lr, posw, bs=64, patience=6, tag=""):
    """Adam + gradient clipping; returns best validation AUPRC and sets net.p to the best epoch's parameters."""
    m_ = {k: np.zeros_like(v) for k, v in net.p.items()}; v_ = {k: np.zeros_like(v) for k, v in net.p.items()}
    step = 0; best = (-1.0, None); wait = 0; t0 = time.time(); rv = np.isin(ls, va)
    for ep in range(epochs):
        perm = rng.permutation(tr); chunks = [perm[i:i + bs * 20] for i in range(0, len(perm), bs * 20)]
        batches = []
        for ch in chunks:
            ch = ch[np.argsort(lens[ch])]; batches += [ch[i:i + bs] for i in range(0, len(ch), bs)]
        rng.shuffle(batches)
        for pb in batches:
            bd, bm, bx, bst, by, bv = batch_of(pb, starts, lens, V, M, Dl, S, y)
            O, cache_ = net.forward(bd, bm, bx, bst, keep=True)
            _, dO = bce_grad(O, by, bv, posw)
            G = net.backward(cache_, dO)
            gn = np.sqrt(sum(float((g ** 2).sum()) for g in G.values())); sc = min(1.0, 5.0 / (gn + 1e-9)); step += 1
            for kk in net.p:
                g = G[kk] * sc
                m_[kk] = 0.9 * m_[kk] + 0.1 * g; v_[kk] = 0.999 * v_[kk] + 0.001 * g * g
                net.p[kk] = net.p[kk] - lr * (m_[kk] / (1 - 0.9 ** step)) / (np.sqrt(v_[kk] / (1 - 0.999 ** step)) + 1e-8)
        pv = predict_site(net, va, starts, lens, V, M, Dl, S, y, len(ls))
        ap = auprc(y[rv], pv[rv]); lr *= 0.9
        print(f"{tag} ep {ep} val AUPRC {ap:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ap > best[0] + 1e-5: best = (ap, {kk: vv.copy() for kk, vv in net.p.items()}); wait = 0
        else:
            wait += 1
            if wait >= patience: break
    net.p = best[1]
    return best[0]


def main():
    """Tune on the seed-0 validation split (6 random configurations), freeze the best, then train every seed with it.
    env: GRUD_NAME (output key prefix, default grud), GRUD_DECAY=0 for a plain GRU with masks (no input/hidden decay),
         GRUD_NCFG (number of configurations), GRUD_EPOCHS, GRUD_TUNE_EPOCHS."""
    from common import make_splits
    direction, cache, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in sys.argv[4:]]
    name = os.environ.get("GRUD_NAME", "grud"); decay = os.environ.get("GRUD_DECAY", "1") != "0"
    EPOCHS = int(os.environ.get("GRUD_EPOCHS", 30)); TEPOCHS = int(os.environ.get("GRUD_TUNE_EPOCHS", 15)); NCFG = int(os.environ.get("GRUD_NCFG", 6))
    os.makedirs(out, exist_ok=True)
    S_, T_ = direction.split("->"); tag = direction.replace("->", "to")
    Xs, cols, ls, ys = load_site(cache, S_); Xt, colt, lt, yt = load_site(cache, T_)
    ns, nt = int(ls.max()) + 1, int(lt.max()) + 1
    vs = [c[2:] for c in cols if c.startswith("v_")]
    samp = np.asarray(Xs[::5][:, [cols.index("v_" + v) for v in vs]])
    vars_ = [v for j, v in enumerate(vs) if len(np.unique(samp[:, j][np.isfinite(samp[:, j])])) >= 2]
    lens_s = np.bincount(ls, minlength=ns); starts_s = np.r_[0, np.cumsum(lens_s)[:-1]]
    lens_t = np.bincount(lt, minlength=nt); starts_t = np.r_[0, np.cumsum(lens_t)[:-1]]

    def prepare(seed):
        spl = make_splits(ns, nt, seed); rng = np.random.default_rng(seed)
        tr_all = np.array(spl["train"]); rng.shuffle(tr_all); k = int(0.9 * len(tr_all)); tr, va = tr_all[:k], tr_all[k:]
        rtr = np.isin(ls, tr)
        vsrc = np.asarray(Xs[:, [cols.index("v_" + v) for v in vars_]], dtype=np.float32)
        mu = np.nanmean(vsrc[rtr], 0); sd = np.nanstd(vsrc[rtr], 0) + 1e-6; del vsrc
        ag = np.asarray(Xs[:, [cols.index("Age"), cols.index("Gender")]], dtype=np.float32)
        amu = np.nanmean(ag[rtr], 0); asd = np.nanstd(ag[rtr], 0) + 1e-6; del ag
        return rng, tr, va, mu, sd, amu, asd

    def make_net(cfg, rng):
        net = GRUD(len(vars_), cfg["H"], rng)
        if not decay:                      # plain GRU: decay parameters at 0 give gamma=1 and receive zero gradient
            for k in ("wx", "bx", "Wd", "bd"): net.p[k][...] = 0
        return net

    hp_path = f"{out}/hp_{name}_{tag}.json"
    if os.path.exists(hp_path): cfg = json.load(open(hp_path))
    else:
        rng, tr, va, mu, sd, amu, asd = prepare(0)
        Vs, Ms, Ds, Ss = build_arrays(Xs, cols, vars_, mu, sd, amu, asd)
        r = np.random.default_rng(123); res = []
        for i in range(NCFG):
            c = dict(H=int(r.choice([32, 48, 96])), lr=float(r.choice([1e-3, 3e-3])), posw=float(r.choice([3.0, 5.0, 10.0])))
            net = make_net(c, np.random.default_rng(1000 + i))
            ap = fit(net, np.random.default_rng(i), tr, va, Vs, Ms, Ds, Ss, ys, ls, starts_s, lens_s, ns, TEPOCHS, c["lr"], c["posw"], tag=f"tune {i} {c}")
            print("TUNE", i, c, f"val AUPRC {ap:.4f}", flush=True); res.append((ap, c))
        cfg = dict(max(res, key=lambda x: x[0])[1]); cfg["tuning_auprc"] = max(res, key=lambda x: x[0])[0]
        json.dump(cfg, open(hp_path, "w")); del Vs, Ms, Ds, Ss
    print("selected configuration:", cfg, flush=True)
    for seed in seeds:
        path = f"{out}/{tag}_s{seed}.npz"
        if os.path.exists(path): continue
        t0 = time.time(); rng, tr, va, mu, sd, amu, asd = prepare(seed)
        Vs, Ms, Ds, Ss = build_arrays(Xs, cols, vars_, mu, sd, amu, asd)
        net = make_net(cfg, np.random.default_rng(seed))
        ap = fit(net, rng, tr, va, Vs, Ms, Ds, Ss, ys, ls, starts_s, lens_s, ns, EPOCHS, cfg["lr"], cfg["posw"], tag=f"{direction} seed {seed}")
        res = {f"{name}_src": predict_site(net, np.arange(ns), starts_s, lens_s, Vs, Ms, Ds, Ss, ys, len(ls))}
        del Vs, Ms, Ds, Ss
        Vt, Mt, Dt, St = build_arrays(Xt, colt, vars_, mu, sd, amu, asd)
        res[f"{name}_tgt"] = predict_site(net, np.arange(nt), starts_t, lens_t, Vt, Mt, Dt, St, yt, len(lt))
        np.savez_compressed(path, **res)
        print(f"{direction} seed {seed} saved best val AUPRC {ap:.4f} ({time.time()-t0:.0f}s)", flush=True)


def best_ep(best, ap, ep):
    return ep


if __name__ == "__main__":
    if "--check" in sys.argv: grad_check()
    else: main()
