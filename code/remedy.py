"""Stage 3 (task 2): remedies for cross-site calibration / conformal failure.

Source model + source isotonic calibration are fixed. On the OTHER site we split patients into a labeled
'pool' (40%) and a held-out 'test_out' (60%, never used for any fitting). For n_t labeled pool patients:
  platt     : logistic recalibration  logit P(y)=a+b*logit(p_src_cal)
  intercept : recalibration-in-the-large only (slope fixed at 1)
  iso_t     : isotonic regression refitted on target sample
  cp_tgt    : class-conditional conformal quantiles from target sample (one row/patient/class); nonconformity uses the RAW model score
  thr_tgt   : alert threshold re-set on target positives (target sensitivity = 1 - A_POS)
Label-free:
  wcp       : covariate-shift weighted class-conditional conformal (source cp rows, domain-classifier weights;
              approximate: one average-weight pseudo-observation, not per-test-point weights)
Reference: source-only quantities evaluated on in-site test and other-site test.
Variability: SD over seeds x labeled-sample draws (no test-set bootstrap here).
"""
import sys, os, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits, row_mask, make_bins, fit_logit_binned, logit
from helpers2 import iso_fit, one_row_per_patient, qhat, wqhat, calib_point, fit_intercept, sigmoid

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/home/claude/cache"); ap.add_argument("--preds", default="/home/claude/preds")
ap.add_argument("--out", default="/home/claude/results"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--models", nargs="+", default=["gbm_both", "mlp"]); ap.add_argument("--dirs", nargs="+", default=["A->B", "B->A"])
ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600, 3200])
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--a_pos", type=float, default=0.2); ap.add_argument("--a_neg", type=float, default=0.05)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

def meta(site):
    m = np.load(f"{a.cache}/{site}_meta.npz"); return m["loc"], m["y"].astype(int)

def platt_fit(p, y):
    b, n = make_bins(p); cnt = np.bincount(b, minlength=n).astype(float); pos = np.bincount(b, weights=y, minlength=n)
    mp = np.bincount(b, weights=p, minlength=n) / np.maximum(cnt, 1)
    # fit on logit(bin mean p): same parameterisation as calib_point
    return fit_logit_binned(logit(mp), pos, cnt - pos)

def cal_metrics(prefix, p, y):
    d = calib_point(p, y.astype(float)); return {f"{prefix}_{k}": v for k, v in d.items()}

rows = []
for direction in a.dirs:
    S_, T_ = direction.split("->"); tag = direction.replace("->", "to")
    ls, ys = meta(S_); lt, yt = meta(T_); ns, nt = int(ls.max()) + 1, int(lt.max()) + 1
    sep_t = (np.bincount(lt, weights=yt) > 0); ns_row_t = ~sep_t[lt]
    sep_s = (np.bincount(ls, weights=ys) > 0); ns_row_s = ~sep_s[ls]
    for seed in a.seeds:
        f = f"{a.preds}/{tag}_s{seed}.npz"
        if not os.path.exists(f): print("missing", f); continue
        P = np.load(f); spl = make_splits(ns, nt, seed); rng = np.random.default_rng(777 + seed)
        Rcp, Riso, Rin = (row_mask(ls, ns, spl[k]) for k in ["cp", "iso", "test_in"])
        Rpool, Rout = row_mask(lt, nt, spl["pool"]), row_mask(lt, nt, spl["test_out"])
        dr = P["dr_src"].astype(np.float64); dr = np.minimum(dr, np.quantile(dr[Rcp], 0.99))
        for model in a.models:
            s_src, s_tgt = P[f"{model}_src"].astype(np.float64), P[f"{model}_tgt"].astype(np.float64)
            iso = iso_fit(s_src[Riso], ys[Riso]); cs, ct = iso.predict(s_src), iso.predict(s_tgt)
            c1 = one_row_per_patient(ls, Rcp & (ys == 1), rng); c0 = one_row_per_patient(ls, Rcp & (ys == 0), rng)
            q1, q0 = qhat(1 - s_src[c1], a.a_pos), qhat(s_src[c0], a.a_neg)   # raw continuous scores (isotonic scores are tied -> conservative)
            w1, w0 = wqhat(1 - s_src[c1], dr[c1], a.a_pos), wqhat(s_src[c0], dr[c0], a.a_neg)
            thr_src = np.quantile(s_src[Rcp & (ys == 1)], a.a_pos)

            def cp_eval(p, y, qa, qb):
                i1, i0 = (1 - p) <= qa, p <= qb
                return dict(cov_pos=i1[y == 1].mean(), cov_neg=i0[y == 0].mean(), set2=(i1 & i0).mean(),
                            empty=(~i1 & ~i0).mean())
            def thr_eval(s, y, nsr, thr):
                return dict(sens=(s[y == 1] >= thr).mean(), alert_hours=24 * (s[nsr] >= thr).mean())
            base = dict(direction=direction, seed=seed, model=model)
            # --- references (n_t = 0)
            Ri, yi = Rin, ys[Rin]
            rows.append(dict(base, test="in_site", method="source_only", n_t=0, draw=0, **cal_metrics("cal", cs[Ri], yi),
                             **cp_eval(s_src[Ri], yi, q1, q0), **thr_eval(s_src[Ri], yi, ns_row_s[Ri], thr_src)))
            yo = yt[Rout]
            rows.append(dict(base, test="other_site", method="source_only", n_t=0, draw=0, **cal_metrics("cal", ct[Rout], yo),
                             **cp_eval(s_tgt[Rout], yo, q1, q0), **thr_eval(s_tgt[Rout], yo, ns_row_t[Rout], thr_src)))
            rows.append(dict(base, test="other_site", method="weighted_cp_labelfree", n_t=0, draw=0,
                             ess_pos=float(dr[c1].sum() ** 2 / (dr[c1] ** 2).sum()), n_cal_pos=len(c1),
                             ess_neg=float(dr[c0].sum() ** 2 / (dr[c0] ** 2).sum()), n_cal_neg=len(c0),
                             **cp_eval(s_tgt[Rout], yo, w1, w0)))
            # --- labeled-target remedies
            pool_pat = spl["pool"]
            for draw in range(a.draws):
                perm = np.random.default_rng(5000 * seed + draw).permutation(pool_pat)
                for n_t in a.sizes:
                    if n_t > len(perm): continue
                    Rs = row_mask(lt, nt, perm[:n_t]); ysub = yt[Rs]
                    n_sep_pat = int(sep_t[perm[:n_t]].sum()); rec = dict(base, test="other_site", method="labeled_target",
                                                                       n_t=n_t, draw=draw, n_sep_patients=n_sep_pat)
                    z_test = logit(ct[Rout]); z_s = ct[Rs]
                    if n_sep_pat >= 2:
                        pa, pb = platt_fit(z_s, ysub.astype(float))      # NB: fitted on p_src_cal rows of the sample
                        rec.update(cal_metrics("platt", sigmoid(pa + pb * z_test), yo))
                        a0 = fit_intercept(logit(z_s), ysub.astype(float))
                        rec.update(cal_metrics("intercept", sigmoid(a0 + z_test), yo))
                        rec.update(cal_metrics("iso_t", iso_fit(s_tgt[Rs], ysub).predict(s_tgt[Rout]), yo))
                    t1 = one_row_per_patient(lt, Rs & (yt == 1), rng); t0 = one_row_per_patient(lt, Rs & (yt == 0), rng)
                    rec.update({f"cp_tgt_{k}": v for k, v in cp_eval(s_tgt[Rout], yo, qhat(1 - s_tgt[t1], a.a_pos), qhat(s_tgt[t0], a.a_neg)).items()})
                    if (ysub == 1).sum() >= 5:
                        thr_t = np.quantile(s_tgt[Rs & (yt == 1)], a.a_pos)
                        rec.update({f"thr_tgt_{k}": v for k, v in thr_eval(s_tgt[Rout], yo, ns_row_t[Rout], thr_t).items()})
                    rows.append(rec)
        print(direction, "seed", seed, "done", flush=True)

df = pd.DataFrame(rows); df.to_csv(f"{a.out}/remedy_raw.csv", index=False)
num = [c for c in df.columns if df[c].dtype.kind == "f" and c not in ("seed",)]
summ = df.groupby(["direction", "model", "test", "method", "n_t"])[num].agg(["mean", "std"]).reset_index()
summ.columns = ["_".join(c).strip("_") for c in summ.columns]
summ.to_csv(f"{a.out}/remedy_summary.csv", index=False)
print("saved", len(df), "records")
