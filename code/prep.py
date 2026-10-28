"""Build and cache per-site feature matrices + metadata.
usage: python prep.py <site A|B> <psv folder> <out dir>
Outputs: <out>/<site>_X.npy (float32), <site>_cols.json, <site>_meta.npz
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sepsis_pipeline as sp
from common import clinical_scores

site, folder, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

B = sp.build(sp.load_set(folder, site), 0)            # sorted by pid, ICULOS; H=0 = official label
codes, loc = np.unique(B["pid"].values.astype(str), return_inverse=True)
t = B.groupby("pid", sort=False).cumcount().values.astype(np.int32)
# build() sorts by pid string; make sure loc is contiguous per patient
assert (np.diff(loc) != 0).sum() == loc.max(), "records not contiguous per patient"

VIT = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
vc = ["v_" + v for v in VIT]
g = B[vc + ["pid"]].groupby("pid", sort=False)
extra = {}
r6 = g[vc].rolling(6, min_periods=1)
for name, df in [("mean6", r6.mean()), ("min6", r6.min()), ("max6", r6.max())]:
    df = df.reset_index(level=0, drop=True).sort_index()
    for c in vc: extra[f"tmp_{name}_{c[2:]}"] = df[c].values.astype(np.float32)
sh = g[vc].shift(6)
for c in vc: extra[f"tmp_d6_{c[2:]}"] = (B[c].values - sh[c].values).astype(np.float32)
labm = [c for c in B.columns if c.startswith("m_") and c[2:] not in VIT]
lab_cnt = pd.Series(B[labm].sum(axis=1).values.astype(np.float32))
lab24 = lab_cnt.groupby(B["pid"].values, sort=False).rolling(24, min_periods=1).sum() \
    .reset_index(level=0, drop=True).sort_index()
extra["tmp_labs24"] = lab24.values.astype(np.float32)

feat_cols = [c for c in B.columns if c[:2] in ("v_", "m_", "d_")] + sp.STATIC + sp.ADMIN
X = np.column_stack([B[feat_cols].values.astype(np.float32)] + [extra[k] for k in extra])
cols = feat_cols + list(extra)

sc = clinical_scores(B)
g_ = lambda c: B["v_" + c].values.astype(np.float64)
tmp, hr, rr, pc, wbc = g_("Temp"), g_("HR"), g_("Resp"), g_("PaCO2"), g_("WBC")
sirs = ((tmp > 38) | (tmp < 36)).astype(float) + (hr > 90) + ((rr > 20) | (pc < 32)) + ((wbc > 12) | (wbc < 4))
sc["sirs"] = np.asarray(sirs, float)

np.save(f"{out}/{site}_X.npy", X)
json.dump(cols, open(f"{out}/{site}_cols.json", "w"))
np.savez(f"{out}/{site}_meta.npz", loc=loc.astype(np.int32), t=t, y=B["y"].values.astype(np.int8),
         density=B["density"].values.astype(np.float32), codes=codes,
         **{k: v.astype(np.float32) for k, v in sc.items()})
print(site, "rows", len(B), "patients", loc.max() + 1, "features", X.shape[1],
      "septic patients", int(pd.Series(B["y"].values).groupby(loc).max().sum()))
