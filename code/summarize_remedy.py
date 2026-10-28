"""Robust summary of remedy_raw.csv: medians/IQR for calibration parameters (fits can diverge when the
target sample has few septic patients), mean/SD for coverage-type metrics.
usage: python summarize_remedy.py <results dir>"""
import sys, numpy as np, pandas as pd
d = sys.argv[1]; r = pd.read_csv(f"{d}/remedy_raw.csv")
cal = [c for c in r.columns if c.endswith(("_slope", "_intercept"))]
other = [c for c in r.columns if r[c].dtype.kind == "f" and c not in cal + ["seed", "draw"]]
rows = []
for k, g in r.groupby(["direction", "model", "test", "method", "n_t"]):
    rec = dict(zip(["direction", "model", "test", "method", "n_t"], k))
    for c in cal:
        v = g[c].to_numpy(float); ok = np.isfinite(v) & (np.abs(v) < 10)
        if np.isfinite(v).any():
            rec[c + "_median"] = np.median(v[ok]) if ok.any() else np.nan
            rec[c + "_q25"], rec[c + "_q75"] = (np.percentile(v[ok], [25, 75]) if ok.any() else (np.nan, np.nan))
            rec[c + "_diverged_frac"] = 1 - ok.mean()
    for c in other:
        if g[c].notna().any(): rec[c + "_mean"], rec[c + "_sd"] = g[c].mean(), g[c].std()
    rows.append(rec)
out = pd.DataFrame(rows); out.to_csv(f"{d}/remedy_summary_robust.csv", index=False); print("saved", len(out))
