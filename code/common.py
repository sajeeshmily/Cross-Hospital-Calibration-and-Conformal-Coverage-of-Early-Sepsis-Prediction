"""Shared helpers for train_predict.py, evaluate_rigor.py and remedy.py."""
import numpy as np
from scipy import sparse

# ----------------------------------------------------------------- splits
def make_splits(n_src, n_tgt, seed):
    """Patient-index splits. Source site: train/iso/cp/test_in (60/10/10/20).
    Target site: pool (40%, labeled pool for recalibration + unlabeled covariates) / test_out (60%)."""
    r = np.random.default_rng(seed); ps = r.permutation(n_src)
    a, b, c = int(.6 * n_src), int(.7 * n_src), int(.8 * n_src)
    pt = np.random.default_rng(seed + 1000).permutation(n_tgt); k = int(.4 * n_tgt)
    return dict(train=ps[:a], iso=ps[a:b], cp=ps[b:c], test_in=ps[c:], pool=pt[:k], test_out=pt[k:])

def row_mask(pid_code, n_pat, idx):
    m = np.zeros(n_pat, bool); m[idx] = True
    return m[pid_code]

# ----------------------------------------------------------------- clinical scores (PARTIAL; see README)
def clinical_scores(V):
    """V: DataFrame of forward-filled values with columns v_<name>. Missing -> 0 points.
    qSOFA lacks mentation (GCS unavailable). NEWS lacks consciousness. SOFA lacks respiratory,
    CNS and urine output (only coag, liver, renal, MAP)."""
    g = lambda c: V["v_" + c].values.astype(np.float64)
    bins = lambda x, edges, pts: np.select([x <= e for e in edges], pts[:-1], pts[-1])
    rr, sp, sbp, hr, tp, fio2, mapp = g("Resp"), g("O2Sat"), g("SBP"), g("HR"), g("Temp"), g("FiO2"), g("MAP")
    nz = lambda a: np.nan_to_num(a, nan=0.0)
    qsofa = (np.nan_to_num(rr, nan=0) >= 22).astype(float) + ((sbp <= 100) & ~np.isnan(sbp)).astype(float)
    def pts(x, rules):                      # rules: list of (mask_fn, points); NaN -> 0
        out = np.zeros(len(x))
        for f, p in rules: out = np.where(f(x) & ~np.isnan(x), p, out)
        return out
    news = (pts(rr, [(lambda x: x <= 8, 3), (lambda x: (x >= 9) & (x <= 11), 1), (lambda x: (x >= 21) & (x <= 24), 2), (lambda x: x >= 25, 3)])
            + pts(sp, [(lambda x: x <= 91, 3), (lambda x: (x >= 92) & (x <= 93), 2), (lambda x: (x >= 94) & (x <= 95), 1)])
            + pts(fio2, [(lambda x: x > 0.21, 2)])
            + pts(sbp, [(lambda x: x <= 90, 3), (lambda x: (x >= 91) & (x <= 100), 2), (lambda x: (x >= 101) & (x <= 110), 1), (lambda x: x >= 220, 3)])
            + pts(hr, [(lambda x: x <= 40, 3), (lambda x: (x >= 41) & (x <= 50), 1), (lambda x: (x >= 91) & (x <= 110), 1), (lambda x: (x >= 111) & (x <= 130), 2), (lambda x: x >= 131, 3)])
            + pts(tp, [(lambda x: x <= 35.0, 3), (lambda x: (x > 35.0) & (x <= 36.0), 1), (lambda x: (x > 38.0) & (x <= 39.0), 1), (lambda x: x > 39.0, 2)]))
    plt, bil, cr = g("Platelets"), g("Bilirubin_total"), g("Creatinine")
    sofa = (pts(plt, [(lambda x: x < 150, 1), (lambda x: x < 100, 2), (lambda x: x < 50, 3), (lambda x: x < 20, 4)])
            + pts(bil, [(lambda x: x >= 1.2, 1), (lambda x: x >= 2, 2), (lambda x: x >= 6, 3), (lambda x: x >= 12, 4)])
            + pts(cr, [(lambda x: x >= 1.2, 1), (lambda x: x >= 2, 2), (lambda x: x >= 3.5, 3), (lambda x: x >= 5, 4)])
            + pts(mapp, [(lambda x: x < 70, 1)]))
    return dict(qsofa=qsofa, news=news, sofa=sofa)

STD_CUTOFF = dict(qsofa=2, news=5, sofa=2)

# ----------------------------------------------------------------- PhysioNet/CinC 2019 Utility score
# Re-implementation from the challenge definition (dt_early=-12, dt_optimal=-6, dt_late=+3 h;
# max U_TP=1, min U_FN=-2, U_FP=-0.05, U_TN=0). VERIFY against the official evaluate_sepsis_score.py.
DT_EARLY, DT_OPT, DT_LATE, U_FP = -12.0, -6.0, 3.0, -0.05

def utility_arrays(loc, t, label):
    """loc: compact patient code per row (0..n-1); t: hour index within patient (0-based, position in record)."""
    n = loc.max() + 1
    first = np.full(n, np.inf); pos = label == 1
    np.minimum.at(first, loc[pos], t[pos])
    tsep = first - DT_OPT                                   # t_sepsis = first label hour + 6
    septic = np.isfinite(tsep)[loc]
    rel = np.where(septic, t - tsep[loc], 0.0)
    scored = (~septic) | (rel <= DT_LATE)
    u_tp = np.where(rel <= DT_OPT, np.maximum(rel / 6.0 + 2.0, U_FP), -rel / 9.0 + 1.0 / 3.0)
    u_fn = np.where(rel <= DT_OPT, 0.0, (-2.0 / 9.0) * rel - 4.0 / 3.0)
    return dict(septic=septic, rel=np.where(septic, rel, np.nan), scored=scored,
                u1=np.where(septic, u_tp, U_FP) * scored, u0=np.where(septic, u_fn, 0.0) * scored,
                best=np.where(septic, np.maximum(u_tp, 0.0), 0.0) * scored)

def util_norm_total(ua, pred):
    uo, ui, ub = np.where(pred, ua["u1"], ua["u0"]).sum(), ua["u0"].sum(), ua["best"].sum()
    return (uo - ui) / (ub - ui) if ub != ui else np.nan

def pick_threshold(s, ua, n_cand=80):
    """Threshold maximising normalised utility on validation rows."""
    u = np.unique(s)
    cands = u[1:] if len(u) <= n_cand else np.unique(np.quantile(s, np.linspace(0.80, 0.9995, n_cand)))
    vals = [util_norm_total(ua, s >= c) for c in cands]
    return cands[int(np.nanargmax(vals))]

# ----------------------------------------------------------------- binned patient-level bootstrap machinery
def make_bins(s, nb=4000):
    u = np.unique(s)
    edges = u if len(u) <= nb else np.unique(np.quantile(s, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 1)
    return b, len(edges)

def hist_csr(loc, bins, mask, npat, nbins):
    return sparse.csr_matrix((np.ones(int(mask.sum())), (loc[mask], bins[mask])), shape=(npat, nbins))

def auc_ap(pos, neg):
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0: return np.nan, np.nan
    pd_, nd = pos[::-1], neg[::-1]
    auc = (nd * (np.cumsum(pd_) - pd_ + 0.5 * pd_)).sum() / (P * N)
    tp, fp = np.cumsum(pd_), np.cumsum(nd)
    ap = (pd_ / P * tp / np.maximum(tp + fp, 1e-12)).sum()
    return auc, ap

def fit_logit_binned(z, pos, neg, iters=30):
    """2-parameter logistic recalibration  logit P(y=1) = a + b z  on binned counts."""
    a, b = 0.0, 1.0; m = (pos + neg) > 0; z, pos, neg = z[m], pos[m], neg[m]; n = pos + neg
    if pos.sum() == 0 or neg.sum() == 0: return np.nan, np.nan
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * z)))
        g = np.array([(pos - n * p).sum(), ((pos - n * p) * z).sum()])
        w = n * p * (1 - p) + 1e-12
        H = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g); a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8: break
    return a, b

def wmedian(x, w):
    if len(x) == 0 or w.sum() <= 0: return np.nan
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w)
    return x[np.searchsorted(c, 0.5 * c[-1])]

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
