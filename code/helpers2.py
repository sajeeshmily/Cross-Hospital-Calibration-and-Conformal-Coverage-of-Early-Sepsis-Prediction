"""Extra helpers shared by evaluate_rigor.py and remedy.py (imports common.py)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from common import make_bins, fit_logit_binned, logit

def iso_fit(p, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p, y)

def one_row_per_patient(loc, mask, rng):
    """Index of one random row per patient among rows where mask is True (patient = exchangeable unit)."""
    idx = np.where(mask)[0]
    if len(idx) == 0: return idx
    keys = rng.random(len(idx)); o = np.lexsort((keys, loc[idx])); idx = idx[o]; l = loc[idx]
    return idx[np.r_[True, l[1:] != l[:-1]]]

def fixed_time_unit(loc, t, y, s, n, T, H=None):
    """Patient-level conformal unit at a FIXED prediction time T (hours since ICU admission), the same for every patient.

    Score  m_i = max over the patient's hours t <= T of the model score. It uses only data available at time T and does not
    depend on the outcome, so a prediction set can be formed for any patient at time T.
    Eligible: the patient is still in the ICU at T (has a row at hour >= T) and has not yet had sepsis onset by T
    (onset = first label-positive hour + 6).
    PATIENT-LEVEL OUTCOME (evaluation label only; not the hourly training label, which is positive from 6 h before onset):
      H = h (hours): septic = sepsis onset in (T, T + h]; non-septic = no onset by T + h (never septic, or onset later than T + h,
                     or the patient left the ICU without onset: all count as no onset within the horizon).
      H = None:      septic = onset at any time after T in the remaining stay; non-septic = never septic.
    Returns m (-inf where no row), is_sep, is_non (boolean arrays over patients)."""
    last = np.full(n, -1.0); np.maximum.at(last, loc, t.astype(float))
    first = np.full(n, np.inf); k = y == 1; np.minimum.at(first, loc[k], t[k].astype(float))
    onset = first + 6.0
    elig = last >= T
    has = np.isfinite(onset)
    hor = np.inf if H is None else T + float(H)
    is_sep = elig & has & (onset > T) & (onset <= hor)
    is_non = elig & ((~has) | (onset > hor))
    keep = t <= T
    m = np.full(n, -np.inf); np.maximum.at(m, loc[keep], s[keep])
    return m, is_sep, is_non

def qhat(scores, alpha):
    """Finite-sample split-conformal quantile (inf if too few calibration points)."""
    n = len(scores)
    if n == 0: return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]

def wqhat(scores, w, alpha):
    """Weighted conformal quantile (covariate shift); +1 pseudo-observation at +inf with mean weight."""
    if len(scores) == 0: return np.inf
    o = np.argsort(scores); s, w = scores[o], w[o]
    c = np.cumsum(w) / (w.sum() + w.mean())
    k = np.searchsorted(c, 1 - alpha, side="left")
    return np.inf if k >= len(s) else s[k]

def calib_point(p, y, nb=4000):
    """Calibration slope / intercept (logistic recalibration of logit p), Brier, observed:expected."""
    b, n = make_bins(p, nb)
    cnt = np.bincount(b, minlength=n).astype(float); pos = np.bincount(b, weights=y, minlength=n)
    mp = np.bincount(b, weights=p, minlength=n) / np.maximum(cnt, 1)
    a, s = fit_logit_binned(logit(mp), pos, cnt - pos)
    return dict(intercept=a, slope=s, brier=float(np.mean((p - y) ** 2)),
                oe=float(y.sum() / max(p.sum(), 1e-12)))

def fit_intercept(z, y, iters=50):
    """Recalibration-in-the-large: solve sum sigmoid(a+z)=sum y (slope fixed at 1)."""
    a = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + z))); g = y.sum() - p.sum(); h = (p * (1 - p)).sum() + 1e-12
        step = g / h; a += step
        if abs(step) < 1e-8: break
    return a

def sigmoid(x): return 1 / (1 + np.exp(-x))
