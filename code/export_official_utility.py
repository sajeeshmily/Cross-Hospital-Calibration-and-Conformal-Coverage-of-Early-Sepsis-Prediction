#!/usr/bin/env python3
"""Export test-set predictions in the PhysioNet/CinC 2019 evaluation format, so the OFFICIAL
evaluate_sepsis_score.py can be run on them and compared with this study's own Utility.

usage:
  python export_official_utility.py --cache <cache> --preds <preds dir with per-seed npz> --data <folder with training_setA/ training_setB/>
                                    --out <export dir> [--seed 0] [--models gbm_both grud] [--dirs "A->B" "B->A"]

For every (direction, test set, model) it writes
  <out>/<direction>_<test>_<model>/labels/<patient>.psv        copy of the original file (has SepsisLabel)
  <out>/<direction>_<test>_<model>/predictions/<patient>.psv   header PredictedProbability|PredictedLabel, one row per hour
and <out>/expected_values.csv with the numbers this study's code obtains for the same files (AUROC, AUPRC, normalised Utility).

Predicted probability = raw model score (rank-identical to the calibrated probability, so AUROC/AUPRC are unchanged).
Predicted label = score >= threshold that maximises this study's normalised Utility on the SOURCE conformal-calibration patients
(no test patient is used). The official script is then run as:
  python evaluate_sepsis_score.py <labels dir> <predictions dir> <output file>
and its "Utility" (normalised) should match expected_values.csv. Note: the official script reads PredictedLabel for Utility and
PredictedProbability for AUROC/AUPRC; accuracy and F-measure in its output are not used in the paper.
"""
import argparse, glob, json, os, shutil, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits, row_mask, utility_arrays, util_norm_total, pick_threshold, auc_ap
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--models", nargs="+", default=["gbm_both", "grud"]); ap.add_argument("--dirs", nargs="+", default=["A->B", "B->A"])
a = ap.parse_args()


def meta(site):
    m = np.load(f"{a.cache}/{site}_meta.npz", allow_pickle=True)
    return m["loc"], m["t"], m["y"].astype(int), [str(c).split("_", 1)[1] for c in m["codes"]]


def psv_folder(site):
    hits = glob.glob(f"{a.data}/training_set{site}/**/*.psv", recursive=True)
    if not hits: sys.exit(f"no .psv under {a.data}/training_set{site}")
    return os.path.dirname(hits[0])


rows = []
for direction in a.dirs:
    S_, T_ = direction.split("->"); tag = direction.replace("->", "to")
    ls, ts, ys, cs = meta(S_); lt, tt, yt, ct = meta(T_)
    ns, nt = int(ls.max()) + 1, int(lt.max()) + 1
    P = np.load(f"{a.preds}/{tag}_s{a.seed}.npz"); spl = make_splits(ns, nt, a.seed)
    Rcp = row_mask(ls, ns, spl["cp"])
    ucp = None
    for name in a.models:
        s_src, s_tgt = P[f"{name}_src"].astype(np.float64), P[f"{name}_tgt"].astype(np.float64)
        # threshold from source conformal-calibration patients only (same rule as evaluate_rigor.py)
        loc_cp = np.unique(ls[Rcp], return_inverse=True)[1]
        ucp = utility_arrays(loc_cp, ts[Rcp], ys[Rcp])
        thr = pick_threshold(s_src[Rcp], ucp)
        for test, (loc, t, y, codes, s, site, pats) in {
                "in_site": (ls, ts, ys, cs, s_src, S_, spl["test_in"]),
                "other_site": (lt, tt, yt, ct, s_tgt, T_, spl["test_out"])}.items():
            m = row_mask(loc, int(loc.max()) + 1, pats)
            lc = np.unique(loc[m], return_inverse=True)[1]
            ua = utility_arrays(lc, t[m], y[m])
            pred = s[m] >= thr
            exp = dict(direction=direction, test=test, model=name, seed=a.seed, threshold=float(thr), n_patients=int(lc.max()) + 1,
                       n_rows=int(m.sum()), auroc=roc_auc_score(y[m], s[m]), auprc=average_precision_score(y[m], s[m]),
                       utility_normalised=float(util_norm_total(ua, pred)))
            rows.append(exp)
            d = f"{a.out}/{tag}_{test}_{name}"; os.makedirs(f"{d}/labels", exist_ok=True); os.makedirs(f"{d}/predictions", exist_ok=True)
            src = psv_folder(site); idx = np.where(m)[0]
            for p in np.unique(loc[m]):
                r = idx[loc[idx] == p]; code = codes[p]
                shutil.copy(f"{src}/{code}.psv", f"{d}/labels/{code}.psv")
                n_file = sum(1 for _ in open(f"{src}/{code}.psv")) - 1
                if n_file != len(r): sys.exit(f"row count mismatch for {code}: file {n_file}, cache {len(r)}")
                with open(f"{d}/predictions/{code}.psv", "w") as f:
                    f.write("PredictedProbability|PredictedLabel\n")
                    f.writelines(f"{s[i]:.8f}|{int(s[i] >= thr)}\n" for i in r)
            print(direction, test, name, f"patients {exp['n_patients']} utility {exp['utility_normalised']:.4f}", flush=True)
os.makedirs(a.out, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{a.out}/expected_values.csv", index=False)
print("wrote", f"{a.out}/expected_values.csv")
