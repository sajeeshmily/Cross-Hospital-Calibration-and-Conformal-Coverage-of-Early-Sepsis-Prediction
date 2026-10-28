#!/usr/bin/env python3
"""Reliability of repair with labeled target samples: share of repeated draws that reach a target, with uncertainty.
Per (model, direction, n_t) over the 25 draws (5 seeds x 5 random draws of the labeled sample): septic patients in the sample (mean, min-max),
calibration slope after logistic recalibration (median, 5th-95th percentile), share of draws with slope in [0.8, 1.2],
septic-hour coverage after re-estimating conformal quantiles on the sample (mean, 5th-95th percentile), share of draws with coverage >= 0.75 and >= 0.80,
non-septic coverage share >= 0.90. Draws within a seed reuse one labeled pool and one trained model, so they are not independent. The reported 95% intervals therefore use the seed as the unit:
the share is computed per seed (5 draws each), and the interval is mean +/- t(0.975, 4 df) * SD / sqrt(5) of the five seed shares, clipped to [0, 100]%.
(Wilson intervals that treat the 25 draws as independent are kept in the csv as share_*_lo/_hi for comparison; they are too narrow.) Seeds resample the same finite
patient population, so even the seed-level interval reflects split and model variation, not sampling of other hospitals. Per-seed minimum shares are given as well.
Thresholds were set when this table was written, after the draws had been seen; they are not preregistered.
usage: python remedy_reliability.py --results <results dir>"""
import argparse, numpy as np, pandas as pd
from scipy import stats as sps
ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); a = ap.parse_args(); R = a.results
raw = pd.concat([pd.read_csv(f"{R}/remedy_raw.csv"), pd.read_csv(f"{R}/grud_remedy_raw.csv")], ignore_index=True)
raw = raw[(raw.method == "labeled_target") & raw.model.isin(["gbm_both", "grud"])]
T = dict(slope_lo=0.8, slope_hi=1.2, pos_min=0.75, pos_nominal=0.80, neg_min=0.90)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
rows = []
for (model, direction, n_t), g in raw.groupby(["model", "direction", "n_t"]):
    n = len(g); sl = g.platt_slope.values; cp = g.cp_tgt_cov_pos.values; cn = g.cp_tgt_cov_neg.values
    ok_sl = (sl >= T["slope_lo"]) & (sl <= T["slope_hi"]); ok_p = cp >= T["pos_min"]; ok_n = cp >= T["pos_nominal"]; ok_neg = cn >= T["neg_min"]
    per_seed = lambda ok: min(ok[g.seed.values == s].mean() for s in g.seed.unique())
    r = dict(model=model, direction=direction, n_t=int(n_t), n_draws=n, septic_mean=g.n_sep_patients.mean(), septic_min=g.n_sep_patients.min(), septic_max=g.n_sep_patients.max(),
             slope_median=np.median(sl), slope_p05=np.percentile(sl, 5), slope_p95=np.percentile(sl, 95),
             cov_pos_mean=cp.mean(), cov_pos_p05=np.percentile(cp, 5), cov_pos_p95=np.percentile(cp, 95), cov_pos_sd=cp.std(ddof=1),
             cov_neg_mean=cn.mean(), cov_neg_p05=np.percentile(cn, 5))
    def seed_ci(ok):
        sh = np.array([ok[g.seed.values == sd].mean() for sd in sorted(g.seed.unique())]); h = sps.t.ppf(0.975, len(sh) - 1) * sh.std(ddof=1) / np.sqrt(len(sh))
        return max(sh.mean() - h, 0.0), min(sh.mean() + h, 1.0)
    for nm, ok in [("slope_in_band", ok_sl), ("pos_cov_ge_0.75", ok_p), ("pos_cov_ge_0.80", ok_n), ("neg_cov_ge_0.90", ok_neg)]:
        lo, hi = wilson(ok.sum(), n); r[f"share_{nm}"] = ok.mean(); r[f"share_{nm}_lo"] = lo; r[f"share_{nm}_hi"] = hi; r[f"share_{nm}_min_seed"] = per_seed(ok)
        r[f"share_{nm}_seed_lo"], r[f"share_{nm}_seed_hi"] = seed_ci(ok)
    rows.append(r)
out = pd.DataFrame(rows).sort_values(["model", "direction", "n_t"], ascending=[False, True, True]); out.to_csv(f"{R}/remedy_reliability.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(out[["model", "direction", "n_t", "septic_mean", "septic_min", "septic_max", "slope_median", "slope_p05", "slope_p95", "share_slope_in_band", "cov_pos_mean", "cov_pos_p05", "share_pos_cov_ge_0.75", "share_pos_cov_ge_0.80", "share_neg_cov_ge_0.90"]].round(2).to_string())

# ---- Table 8 (manuscript layout): primary model every size; exploratory GRU-D at 100 and 800
def pct(r, nm): return f"{100 * r[f'share_{nm}']:.0f} ({100 * r[f'share_{nm}_seed_lo']:.0f}–{100 * r[f'share_{nm}_seed_hi']:.0f})"
def cap(x): return ">10" if x > 10 else f"{x:.2f}"
names = {"gbm_both": "GBM, values + mask (primary)", "grud": "GRU-D (exploratory)"}
t8 = []
for _, r in out.sort_values(["model", "direction", "n_t"], ascending=[False, True, True]).iterrows():
    if r.model == "grud" and r.n_t not in (100, 800): continue
    t8.append({"Model": names[r.model], "Direction": r.direction.replace("->", "→"), "Labeled patients": f"{r.n_t:,}",
               "Septic patients, mean (range)": f"{r.septic_mean:.0f} ({r.septic_min:.0f}–{r.septic_max:.0f})",
               "Calibration slope, median (5th–95th pct)": f"{r.slope_median:.2f} ({r.slope_p05:.2f}–{cap(r.slope_p95)})",
               "Draws with slope 0.8–1.2, % (95% interval over seeds)": pct(r, "slope_in_band"),
               "Septic-hour coverage, mean (5th–95th pct)": f"{r.cov_pos_mean:.2f} ({r.cov_pos_p05:.2f}–{r.cov_pos_p95:.2f})",
               "Draws with septic coverage ≥0.75, % (95% interval over seeds)": pct(r, "pos_cov_ge_0.75"), "Draws with septic coverage ≥0.80, % (95% interval over seeds)": pct(r, "pos_cov_ge_0.80"),
               "Draws with non-septic coverage ≥0.90, %": f"{100 * r['share_neg_cov_ge_0.90']:.0f}"})
T8 = pd.DataFrame(t8); T8.to_csv(f"{R}/table8_remedy_reliability.csv", index=False)
print(T8.to_markdown(index=False) if hasattr(T8, "to_markdown") else T8.to_string())
