#!/usr/bin/env python3
"""Check that fixed-time patient-level class-conditional split conformal reaches its target when calibration and test patients are
exchangeable: pool held-out source patients (iso + cp + test_in, none used for training) that are eligible at prediction time T,
repeatedly split them at random into calibration and test patients (calibration size as in the study), and average coverage.
Unit and score: helpers2.fixed_time_unit (max model score over hours <= T; same window for every patient). Outcome: septic = onset in (T, T+H], H = 24 h by default.
This is an implementation check for the in-site claim; it does not define the score or say anything about another hospital.
usage: python conformal_guarantee_check.py --cache C --preds P --out CSV [--seed 0] [--R 2000] [--T 24 12 48]"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits
from helpers2 import qhat, fixed_time_unit
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--R", type=int, default=2000)
ap.add_argument("--models", nargs="+", default=["gbm_both", "grud"]); ap.add_argument("--T", type=int, nargs="+", default=[24, 12, 48]); ap.add_argument("--H", type=int, default=24)
a = ap.parse_args(); rows = []; H = None if a.H <= 0 else a.H
for direction in ["A->B", "B->A"]:
    S_, T_ = direction.split("->"); tag = direction.replace("->", "to")
    m = np.load(f"{a.cache}/{S_}_meta.npz", allow_pickle=True); loc, y, tt = m["loc"].astype(int), m["y"].astype(int), m["t"].astype(int)
    n = int(loc.max()) + 1; tgt = int(np.load(f"{a.cache}/{T_}_meta.npz", allow_pickle=True)["loc"].max()) + 1
    spl = make_splits(n, tgt, a.seed); held = np.concatenate([spl["iso"], spl["cp"], spl["test_in"]]); ncal = len(spl["cp"])
    P = np.load(f"{a.preds}/{tag}_s{a.seed}.npz")
    for name in a.models:
        s = P[f"{name}_src"].astype(float)
        for T in a.T:
            m_pat, is_sep, is_non = fixed_time_unit(loc, tt, y, s, n, T, H)
            pool = held[is_sep[held] | is_non[held]]
            ncal_T = int(round(ncal * len(pool) / len(held)))        # keep the study's calibration share among eligible patients
            rng = np.random.default_rng(0); c1, c0 = [], []
            for _ in range(a.R):
                p = rng.permutation(pool); cal, tst = p[:ncal_T], p[ncal_T:]
                cs, cn = cal[is_sep[cal]], cal[is_non[cal]]; ts, tn = tst[is_sep[tst]], tst[is_non[tst]]
                c1.append(np.mean((1 - m_pat[ts]) <= qhat(1 - m_pat[cs], .20))); c0.append(np.mean(m_pat[tn] <= qhat(m_pat[cn], .05)))
            rows.append(dict(direction=direction, model=name, T=T, n_cal_patients=ncal_T, n_eligible_pool=len(pool), n_pool_septic=int(is_sep[pool].sum()),
                             R=a.R, cov_septic_mean=np.mean(c1), cov_septic_sd=np.std(c1), cov_nonseptic_mean=np.mean(c0), cov_nonseptic_sd=np.std(c0)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
