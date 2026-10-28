#!/usr/bin/env python3
"""Site characteristics (Table M1) and patient flow into each split (Table M2).
usage: python make_site_tables.py --cache <cache> --results <results> [--seeds 0 1 2 3 4]"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits
ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--results", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); a = ap.parse_args()
D = {}
for s in "AB":
    m = np.load(f"{a.cache}/{s}_meta.npz", allow_pickle=True); cols = json.load(open(f"{a.cache}/{s}_cols.json"))
    X = np.load(f"{a.cache}/{s}_X.npy", mmap_mode="r"); loc = m["loc"].astype(int); y = m["y"].astype(int); n = int(loc.max()) + 1
    first = np.full(n, -1); first[loc[::-1]] = np.arange(len(loc))[::-1]
    col = lambda c: np.asarray(X[:, cols.index(c)], float)
    age = col("Age")[first]; sex = col("Gender")[first]; u1 = col("Unit1")[first]; u2 = col("Unit2")[first]
    los = np.bincount(loc, minlength=n); sep = np.bincount(loc, weights=y, minlength=n) > 0
    q = lambda v: f"{np.nanmedian(v):.0f} ({np.nanpercentile(v,25):.0f} to {np.nanpercentile(v,75):.0f})"
    dens = np.mean([np.mean(np.asarray(X[:, cols.index(c)], float)) for c in cols if c.startswith("m_")])
    D[s] = dict(loc=loc, y=y, n=n, sep=sep, tab={
        "Patients": f"{n:,}", "Hourly rows": f"{len(loc):,}", "Septic patients": f"{sep.sum():,} ({100*sep.mean():.1f}%)",
        "Septic hours (label positive)": f"{int(y.sum()):,} ({100*y.mean():.2f}% of hours)",
        "Age, median (IQR), years": q(age), "Female, %": f"{100*np.nanmean(sex==0):.0f}", "MICU unit, %": f"{100*np.nanmean(u1==1):.0f}",
        "SICU unit, %": f"{100*np.nanmean(u2==1):.0f}", "ICU stay, median (IQR), hours": q(los),
        "Share of variable-hours observed, %": f"{100*dens:.1f}"})
T1 = pd.DataFrame({"Characteristic": list(D["A"]["tab"]), "Set A": list(D["A"]["tab"].values()), "Set B": list(D["B"]["tab"].values())})
T1.to_csv(f"{a.results}/tableM1_site_characteristics.csv", index=False)
rows = []
for direction in ["A->B", "B->A"]:
    S_, T_ = direction.split("->")
    acc = {k: [] for k in ["train", "iso", "cp", "test_in", "pool", "test_out"]}
    for seed in a.seeds:
        sp = make_splits(D[S_]["n"], D[T_]["n"], seed)
        for k in acc:
            D_ = D[S_] if k in ("train", "iso", "cp", "test_in") else D[T_]
            acc[k].append((len(sp[k]), int(D_["sep"][sp[k]].sum())))
    role = {"train": f"{S_} training", "iso": f"{S_} isotonic calibration", "cp": f"{S_} threshold and conformal calibration",
            "test_in": f"{S_} in-site test", "pool": f"{T_} labeled pool (remedies only)", "test_out": f"{T_} other-hospital test"}
    for k, v in acc.items():
        rows.append({"Direction": direction, "Subset": role[k], "Patients": f"{v[0][0]:,}", "Septic patients, mean over seeds (range)": f"{np.mean([x[1] for x in v]):.0f} ({min(x[1] for x in v)}-{max(x[1] for x in v)})"})
pd.DataFrame(rows).to_csv(f"{a.results}/tableM2_patient_flow.csv", index=False)
print(T1.to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False))
