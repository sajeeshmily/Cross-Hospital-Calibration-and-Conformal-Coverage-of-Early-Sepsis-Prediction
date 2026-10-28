#!/usr/bin/env python3
"""Independent audit of the fixed-time patient-level conformal analysis, plus the prediction-target and set-reading sensitivity analyses.

Everything here is a plain-loop re-implementation (no helpers2.fixed_time_unit / qhat), so it checks the vectorised code behind Table 4.

1. Re-implementation check. For every seed, model and direction the patient-level coverage at the main T and main outcome horizon H (default 24 h; eligible patients;
   septic = onset in (T, T+H]) is recomputed with plain loops, in-site and at the other hospital, and its five-seed mean is compared with conformal_unit_analysis.csv
   (patient_pos / patient_neg). Max absolute difference is written to results/conformal_audit_check.txt; the script exits non-zero if it exceeds 1e-9.
2. Score scale. The scores are the raw model outputs (predicted probability of the hourly label, no isotonic step). The range of the scores and of the patient-level
   score m, the number of distinct values, the class-specific quantiles and the implied thresholds on m are reported.
   Septic is in the set iff m >= 1 - q_septic; non-septic is in the set iff m <= q_nonseptic. Because the two class-specific quantiles are separate, the set can hold
   both labels (m between the two thresholds), one label, or (if the thresholds crossed) neither.
3. Outcome-horizon sensitivity (results/conformal_horizon_sensitivity.csv). Two different things are kept apart: the HOURLY TRAINING LABEL (positive from 6 h before onset, so the
   model is trained for "onset within about 6 h") and the PATIENT-LEVEL OUTCOME used to evaluate the sets (septic = onset in (T, T+H]). The main H is 24 h; H = 6 h (the training-label
   horizon, but only a handful of septic calibration patients) and H = 0 (onset any time after T in the remaining stay) are sensitivity analyses. Never-septic patients and patients whose onset
   comes after T+H form the non-septic class. Eligibility is unchanged (in the ICU at T, no onset by T). The score and the calibration/test patients are the same; only the evaluation class changes.
4. Set reading (results/conformal_set_distribution.csv): share of eligible test patients, by true class, whose prediction set is {non-septic}, {both} or {septic} (or empty).
Seed-level intervals: mean over seeds with a 95% t interval (4 df) in the sensitivity outputs.

usage: python conformal_fixed_time_audit.py --cache C --preds P --results R [--T 24] [--H 24] [--H_other 6 0] [--seeds 0 1 2 3 4] [--models gbm_both grud]"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--results", required=True)
ap.add_argument("--T", type=int, default=24); ap.add_argument("--H", type=int, default=24, help="main outcome horizon (h)"); ap.add_argument("--H_other", type=int, nargs="*", default=[6, 0], help="sensitivity horizons; 0 = rest of stay")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--models", nargs="+", default=["gbm_both", "grud"])
ap.add_argument("--a_pos", type=float, default=0.20); ap.add_argument("--a_neg", type=float, default=0.05)
a = ap.parse_args(); T = a.T; R = a.results; Hm = None if a.H <= 0 else a.H
log = []


def say(s):
    log.append(s); print(s, flush=True)


def site(name):
    m = np.load(f"{a.cache}/{name}_meta.npz", allow_pickle=True)
    loc, y, t = m["loc"].astype(int), m["y"].astype(int), m["t"].astype(int)
    o = np.argsort(loc, kind="stable"); n = int(loc.max()) + 1
    start = np.searchsorted(loc[o], np.arange(n + 1))
    return dict(o=o, start=start, y=y, t=t, n=n)


def unit(S, p, sc, H):
    """Plain-loop unit for patient p. Returns (class, m) or None when not eligible. class 1 = septic, 0 = non-septic.
    H = None: septic = onset any time after T; H = h: septic = onset in (T, T+h], non-septic = no onset by T+h."""
    idx = S["o"][S["start"][p]:S["start"][p + 1]]
    last = max(int(S["t"][i]) for i in idx)
    if last < T: return None
    pos = [int(S["t"][i]) for i in idx if S["y"][i] == 1]
    onset = (min(pos) + 6) if pos else None
    if onset is not None and onset <= T: return None
    if H is None: cls = 0 if onset is None else 1
    else: cls = 1 if (onset is not None and onset <= T + H) else 0
    m = max(float(sc[i]) for i in idx if S["t"][i] <= T)
    return cls, m


def quant(vals, alpha):
    v = sorted(vals); k = int(np.ceil((len(v) + 1) * (1 - alpha)))
    return float("inf") if k > len(v) or len(v) == 0 else v[k - 1]


def run(S_src, S_tgt, sc_src, sc_tgt, spl, H):
    cal = [unit(S_src, int(p), sc_src, H) for p in spl["cp"]]
    c1 = [1 - u[1] for u in cal if u and u[0] == 1]; c0 = [u[1] for u in cal if u and u[0] == 0]
    q1, q0 = quant(c1, a.a_pos), quant(c0, a.a_neg)
    out = dict(q1=q1, q0=q0, n_cal_sep=len(c1), n_cal_non=len(c0))
    for nm, S, sc, pats in (("in_site", S_src, sc_src, spl["test_in"]), ("other_site", S_tgt, sc_tgt, spl["test_out"])):
        us = [unit(S, int(p), sc, H) for p in pats]
        sep = [u[1] for u in us if u and u[0] == 1]; non = [u[1] for u in us if u and u[0] == 0]
        out[nm] = dict(pos=float(np.mean([(1 - m) <= q1 for m in sep])), neg=float(np.mean([m <= q0 for m in non])), n_sep=len(sep), n_non=len(non), sep=sep, non=non)
    return out


ref = pd.read_csv(f"{R}/conformal_unit_analysis.csv")
rows_h, rows_set, rows_scale, diffs = [], [], [], []
for direction in ["A->B", "B->A"]:
    Sn, Tn = direction.split("->"); tag = direction.replace("->", "to")
    S_src, S_tgt = site(Sn), site(Tn)
    for name in a.models:
        acc = {}
        for seed in a.seeds:
            P = np.load(f"{a.preds}/{tag}_s{seed}.npz"); spl = make_splits(S_src["n"], S_tgt["n"], seed)
            sc_src, sc_tgt = P[f"{name}_src"].astype(float), P[f"{name}_tgt"].astype(float)
            for H in [Hm] + [None if h <= 0 else h for h in a.H_other]:
                r = run(S_src, S_tgt, sc_src, sc_tgt, spl, H)
                for nm in ("in_site", "other_site"):
                    acc.setdefault((H, nm), []).append((r[nm]["pos"], r[nm]["neg"], r[nm]["n_sep"], r[nm]["n_non"], r["n_cal_sep"], r["n_cal_non"], r["q1"], r["q0"]))
                if H == Hm:
                    for nm in ("in_site", "other_site"):
                        for cls, vals in (("septic", r[nm]["sep"]), ("non-septic", r[nm]["non"])):
                            v = np.array(vals); both = ((1 - v) <= r["q1"]) & (v <= r["q0"]); sep_only = ((1 - v) <= r["q1"]) & ~(v <= r["q0"])
                            non_only = ~((1 - v) <= r["q1"]) & (v <= r["q0"]); empty = ~((1 - v) <= r["q1"]) & ~(v <= r["q0"])
                            for lab, msk in (("{non-septic}", non_only), ("{septic, non-septic}", both), ("{septic}", sep_only), ("{}", empty)):
                                rows_set.append(dict(direction=direction, model=name, test=nm, seed=seed, true_class=cls, prediction_set=lab, share=float(msk.mean())))
                    allm = np.array(r["in_site"]["sep"] + r["in_site"]["non"])
                    rows_scale.append(dict(direction=direction, model=name, seed=seed, score_min=float(sc_src.min()), score_max=float(sc_src.max()),
                                           score_distinct=int(len(np.unique(sc_src))), m_min=float(allm.min()), m_max=float(allm.max()), m_median=float(np.median(allm)),
                                           q_septic=r["q1"], q_nonseptic=r["q0"], thr_septic_in_set_if_m_ge=1 - r["q1"], thr_nonseptic_in_set_if_m_le=r["q0"]))
            print(direction, name, "seed", seed, flush=True)
        for (H, nm), lst in acc.items():
            A = np.array(lst, dtype=float)
            if H == Hm:
                for j, met in ((0, "patient_pos"), (1, "patient_neg")):
                    m_ref = ref[(ref.direction == direction) & (ref.model == name) & (ref.test == nm) & (ref.metric == met)]["mean"].iloc[0]
                    diffs.append(abs(A[:, j].mean() - m_ref))
            tcrit = sps.t.ppf(0.975, len(lst) - 1)
            d = dict(direction=direction, model=name, test=nm, T=T, horizon_h=("rest of stay" if H is None else H), is_main=(H == Hm), n_seeds=len(lst),
                     n_test_septic=A[:, 2].mean(), n_test_nonseptic=A[:, 3].mean(), n_cal_septic=A[:, 4].mean(), n_cal_nonseptic=A[:, 5].mean())
            for j, met in ((0, "septic"), (1, "nonseptic")):
                mu, sd = A[:, j].mean(), A[:, j].std(ddof=1)
                d.update({f"cov_{met}_mean": mu, f"cov_{met}_sd_seeds": sd, f"cov_{met}_t_lo": mu - tcrit * sd / np.sqrt(len(lst)), f"cov_{met}_t_hi": mu + tcrit * sd / np.sqrt(len(lst))})
            rows_h.append(d)

pd.DataFrame(rows_h).to_csv(f"{R}/conformal_horizon_sensitivity.csv", index=False)
ss = pd.DataFrame(rows_set).groupby(["direction", "model", "test", "true_class", "prediction_set"]).share.mean().reset_index()
ss.to_csv(f"{R}/conformal_set_distribution.csv", index=False)
sc = pd.DataFrame(rows_scale).groupby(["direction", "model"]).mean(numeric_only=True).drop(columns="seed").reset_index()
sc.to_csv(f"{R}/conformal_score_scale.csv", index=False)
mx = max(diffs)
say(f"[{'PASS' if mx < 1e-9 else 'FAIL'}] plain-loop re-implementation of patient-level coverage (T={T}, all seeds, models, directions, both sites; {len(diffs)} five-seed means): max |difference| to conformal_unit_analysis.csv = {mx:.2e}")
say("score scale (mean over seeds):\n" + sc.round(4).to_string(index=False))
say("prediction sets at T (share of eligible test patients, mean over seeds), GBM:\n" + ss[ss.model == "gbm_both"].round(3).to_string(index=False))
open(f"{R}/conformal_audit_check.txt", "w").write("\n".join(log) + "\n")
sys.exit(0 if mx < 1e-9 else 1)
