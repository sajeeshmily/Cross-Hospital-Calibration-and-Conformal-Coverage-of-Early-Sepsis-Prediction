#!/usr/bin/env python3
"""Conformal coverage under three definitions of the evaluation unit.

Hour-level analyses (separate empirical procedure; calibration draws one random hour per patient per class):
  hour_pooled   all test hours of a class pooled (long stays weigh more)
  hour_patient  per-patient share of covered hours, averaged over patients
Patient-level analysis at a FIXED prediction time T (default 24 h after ICU admission; sensitivity 12 and 48 h):
  score m = max model score over the patient's hours <= T  (same window for every patient; outcome never moves it)
  nonconformity 1 - m (septic), m (non-septic); quantile from the calibration patients of the class (same scores); coverage =
  share of test patients of the class whose nonconformity <= quantile.
  Eligible: still in the ICU at T and no sepsis onset by T (onset = first septic-label hour + 6).
  PATIENT-LEVEL OUTCOME (evaluation label only; distinct from the hourly training label, which is positive from 6 h before onset):
  with horizon H (default 24 h): septic = onset in (T, T+H]; non-septic = no onset by T+H (never septic, later onset, or left the ICU without onset).
  --H 0 gives the 'rest of stay' outcome (septic = onset any time after T).
  A guarantee holds in-site for exchangeable eligible patients; at the other hospital coverage is an empirical measurement under shift.
Metrics: patient_pos / patient_neg are at the main T; patient_pos_T12, patient_neg_T48, ... are the sensitivity times.
Earlier versions scored septic patients over label-defined or whole-stay windows; neither is used
(old output: results/archive_label_dependent_patient_score/).
Intervals: 95% percentile range of the seed-average of patient-level bootstrap replicates (B draws).

usage: python conformal_unit_analysis.py --cache <cache> --preds <preds_all> --out <csv> [--seeds 0 1 2 3 4] [--B 200] [--models gbm_both grud] [--T 24] [--sens_T 12 48]
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits, row_mask
from helpers2 import one_row_per_patient, qhat, fixed_time_unit

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--B", type=int, default=200)
ap.add_argument("--models", nargs="+", default=["gbm_both", "grud"]); ap.add_argument("--dirs", nargs="+", default=["A->B", "B->A"])
ap.add_argument("--a_pos", type=float, default=0.20); ap.add_argument("--a_neg", type=float, default=0.05)
ap.add_argument("--T", type=int, default=24); ap.add_argument("--sens_T", type=int, nargs="*", default=[12, 48])
ap.add_argument("--H", type=int, default=24, help="outcome horizon in hours after T (septic = onset in (T, T+H]); 0 = onset any time after T")
a = ap.parse_args()
TS = [a.T] + list(a.sens_T)
H = None if a.H <= 0 else a.H
sfx = {T: ("" if T == a.T else f"_T{T}") for T in TS}


def meta(site):
    m = np.load(f"{a.cache}/{site}_meta.npz", allow_pickle=True)
    return m["loc"].astype(int), m["y"].astype(int), m["t"].astype(int)


def summarise(loc, y, s, tt, rows_sel, q1, q0, qp, pats):
    """Per-patient arrays for hour-pooled / hour-patient coverage and for fixed-time patient-level coverage on test patients `pats`."""
    n = int(loc.max()) + 1
    L, Y, S = loc[rows_sel], y[rows_sel], s[rows_sel]
    cov1 = (1 - S) <= q1; cov0 = S <= q0
    c1 = np.bincount(L, weights=cov1 & (Y == 1), minlength=n); c0 = np.bincount(L, weights=cov0 & (Y == 0), minlength=n)
    ns = np.bincount(L, weights=(Y == 1), minlength=n); nn = np.bincount(L, weights=(Y == 0), minlength=n)
    d = dict(ns=ns[pats], nn=nn[pats], c1=c1[pats], c0=c0[pats])
    for T in TS:
        m, is_sep, is_non = fixed_time_unit(L, tt[rows_sel], Y, S, n, T, H)
        q1p, q0p = qp[T]
        d[f"sep{T}"] = is_sep[pats]; d[f"non{T}"] = is_non[pats]
        d[f"pc1_{T}"] = ((1 - m) <= q1p)[pats]; d[f"pc0_{T}"] = (m <= q0p)[pats]
    return d


def stats(d, w):
    out = {}
    out["hour_pooled_pos"] = (w * d["c1"]).sum() / max((w * d["ns"]).sum(), 1e-12)
    out["hour_pooled_neg"] = (w * d["c0"]).sum() / max((w * d["nn"]).sum(), 1e-12)
    m1 = d["ns"] > 0; m0 = d["nn"] > 0
    out["hour_patient_pos"] = (w[m1] * d["c1"][m1] / d["ns"][m1]).sum() / max(w[m1].sum(), 1e-12)
    out["hour_patient_neg"] = (w[m0] * d["c0"][m0] / d["nn"][m0]).sum() / max(w[m0].sum(), 1e-12)
    for T in TS:
        sp, npz = d[f"sep{T}"], d[f"non{T}"]
        out[f"patient_pos{sfx[T]}"] = (w[sp] * d[f"pc1_{T}"][sp]).sum() / max(w[sp].sum(), 1e-12)
        out[f"patient_neg{sfx[T]}"] = (w[npz] * d[f"pc0_{T}"][npz]).sum() / max(w[npz].sum(), 1e-12)
    return out


rows, cnt = [], []
for direction in a.dirs:
    S_, T_ = direction.split("->"); tag = direction.replace("->", "to")
    ls, ys, ts_ = meta(S_); lt, yt, tt_ = meta(T_)
    ns_, nt_ = int(ls.max()) + 1, int(lt.max()) + 1
    for name in a.models:
        acc = {"in_site": [], "other_site": []}
        for seed in a.seeds:
            P = np.load(f"{a.preds}/{tag}_s{seed}.npz"); spl = make_splits(ns_, nt_, seed)
            s_src, s_tgt = P[f"{name}_src"].astype(float), P[f"{name}_tgt"].astype(float)
            Rcp = row_mask(ls, ns_, spl["cp"]); rng = np.random.default_rng(seed)
            # (i) hour-level quantiles: one random hour per patient per class
            c1 = one_row_per_patient(ls, Rcp & (ys == 1), rng); c0 = one_row_per_patient(ls, Rcp & (ys == 0), rng)
            q1, q0 = qhat(1 - s_src[c1], a.a_pos), qhat(s_src[c0], a.a_neg)
            # (ii) patient-level quantiles at each fixed time T, from the calibration patients' own scores
            sel_cp = np.where(Rcp)[0]; cp_pats = spl["cp"]; qp = {}; ncal = {}
            for T in TS:
                m, is_sep, is_non = fixed_time_unit(ls[sel_cp], ts_[sel_cp], ys[sel_cp], s_src[sel_cp], ns_, T, H)
                cs, cn = cp_pats[is_sep[cp_pats]], cp_pats[is_non[cp_pats]]
                qp[T] = (qhat(1 - m[cs], a.a_pos), qhat(m[cn], a.a_neg)); ncal[T] = (len(cs), len(cn))
            for test, (loc, y, s, tt, pats) in {"in_site": (ls, ys, s_src, ts_, spl["test_in"]), "other_site": (lt, yt, s_tgt, tt_, spl["test_out"])}.items():
                sel = np.where(row_mask(loc, int(loc.max()) + 1, pats))[0]
                d = summarise(loc, y, s, tt, sel, q1, q0, qp, pats)
                n = len(pats); brng = np.random.default_rng(1000 + seed)
                point = stats(d, np.ones(n))
                reps = [stats(d, brng.multinomial(n, np.ones(n) / n).astype(float)) for _ in range(a.B)]
                acc[test].append((point, pd.DataFrame(reps), None))
                for T in TS:
                    cnt.append(dict(direction=direction, model=name, test=test, seed=seed, T=T, n_test_patients=n,
                                    n_test_septic=int(d[f"sep{T}"].sum()), n_test_nonseptic=int(d[f"non{T}"].sum()),
                                    n_cal_septic=ncal[T][0], n_cal_nonseptic=ncal[T][1], q_septic=qp[T][0], q_nonseptic=qp[T][1]))
            print(direction, name, "seed", seed, flush=True)
        for test, lst in acc.items():
            pts = pd.DataFrame([p for p, _, _ in lst]); rep = sum(r.values for _, r, _ in lst) / len(lst)
            rep = pd.DataFrame(rep, columns=lst[0][1].columns)
            for k in pts.columns:
                rows.append(dict(direction=direction, model=name, test=test, metric=k, mean=pts[k].mean(), ci_lo=np.percentile(rep[k], 2.5),
                                 ci_hi=np.percentile(rep[k], 97.5), sd_seeds=pts[k].std(ddof=1), n_seeds=len(lst)))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
c = pd.DataFrame(cnt).groupby(["direction", "model", "test", "T"]).mean(numeric_only=True).drop(columns="seed").reset_index()
c.to_csv(os.path.join(os.path.dirname(os.path.abspath(a.out)), "conformal_fixed_time_counts.csv"), index=False)
print("wrote", a.out)
