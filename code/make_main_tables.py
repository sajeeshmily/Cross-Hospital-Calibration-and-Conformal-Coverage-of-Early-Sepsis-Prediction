#!/usr/bin/env python3
"""Build the manuscript's results tables from the result CSVs (single source of truth for every number in the paper).

Inputs (results dir): rigor_results.csv, grud_rigor_results.csv, conformal_unit_analysis.csv
Cache dir: for test-sample sizes (patients, septic patients, hourly rows) per seed, from the same splits the experiments used.
Outputs (results dir):
  main_table_A.csv / main_table_B.csv   primary model (gradient boosting, values + mask) and exploratory GRU-D
  main_tables.md                        both panels as markdown with notes
  supplementary_all_models.csv          every model, every metric (tidy), with n and SD across seeds
usage: python make_main_tables.py --results <results dir> --cache <cache dir> [--seeds 0 1 2 3 4]
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits, row_mask

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
a = ap.parse_args()
R = a.results

rig = pd.concat([pd.read_csv(f"{R}/rigor_results.csv"), pd.read_csv(f"{R}/grud_rigor_results.csv")], ignore_index=True)
rig["op"] = rig["op"].fillna("-")
cu = pd.read_csv(f"{R}/conformal_unit_analysis.csv")

# ---------------------------------------------------------------- test-sample sizes
sizes = {}
for direction in ["A->B", "B->A"]:
    S_, T_ = direction.split("->")
    meta = {s: np.load(f"{a.cache}/{s}_meta.npz", allow_pickle=True) for s in (S_, T_)}
    loc = {s: meta[s]["loc"].astype(int) for s in meta}; y = {s: meta[s]["y"].astype(int) for s in meta}
    n = {s: int(loc[s].max()) + 1 for s in meta}
    sep = {s: np.bincount(loc[s], weights=(y[s] == 1), minlength=n[s]) > 0 for s in meta}
    acc = {"in_site": [], "other_site": []}
    for seed in a.seeds:
        spl = make_splits(n[S_], n[T_], seed)
        for test, site, pats in [("in_site", S_, spl["test_in"]), ("other_site", T_, spl["test_out"])]:
            rows = int(row_mask(loc[site], n[site], pats).sum())
            acc[test].append((len(pats), int(sep[site][pats].sum()), rows))
    for test, v in acc.items():
        v = np.array(v, float); sizes[(direction, test)] = dict(n_patients=v[:, 0].mean(), n_septic=v[:, 1].mean(), n_rows=v[:, 2].mean())

def site_label(direction, test):
    s, t = direction.split("->")
    return f"{s}\u2192{s} (in-site)" if test == "in_site" else f"{s}\u2192{t} (cross-site)"   # trained on first letter, tested on second

def cell(df, model, direction, test, metric, op="-", nd=3, pct=False):
    r = df[(df.model == model) & (df.direction == direction) & (df.test == test) & (df.metric == metric) & (df.op == op)]
    if r.empty: return ""
    r = r.iloc[0]; k = 100 if pct else 1; f = f"{{:.{0 if pct else nd}f}}"
    return f"{f.format(r['mean'] * k)} ({f.format(r['ci_lo'] * k)}–{f.format(r['ci_hi'] * k)})"

def conf(model, direction, test, metric, nd=2):
    r = cu[(cu.model == model) & (cu.direction == direction) & (cu.test == test) & (cu.metric == metric)]
    if r.empty: return ""
    r = r.iloc[0]; return f"{r['mean']:.{nd}f} ({r['ci_lo']:.{nd}f}–{r['ci_hi']:.{nd}f})"

MODELS = [("gbm_both", "GBM, values + mask (primary)"), ("grud", "GRU-D (exploratory)")]
ROWS = [("A->B", "in_site"), ("A->B", "other_site"), ("B->A", "in_site"), ("B->A", "other_site")]
ALERT = "ops1.0"

A_rows, B_rows = [], []
for model, mname in MODELS:
    for d, t in ROWS:
        sz = sizes[(d, t)]
        base = {"Model": mname, "Evaluation": site_label(d, t), "Test patients (septic)": f"{int(round(sz['n_patients'])):,} ({sz['n_septic']:.0f})"}
        A_rows.append({**base,
                       "AUROC": cell(rig, model, d, t, "auroc"), "AUPRC": cell(rig, model, d, t, "auprc"),
                       "Utility (normalised)": cell(rig, model, d, t, "utility_opt"),
                       "Calibration slope": cell(rig, model, d, t, "cal_slope", nd=2), "Calibration intercept": cell(rig, model, d, t, "cal_intercept", nd=2),
                       "Brier score": cell(rig, model, d, t, "brier", nd=4)})
        B_rows.append({**base,
                       "Coverage, septic (target 0.80)": conf(model, d, t, "patient_pos"),
                       "Coverage, non-septic (target 0.95)": conf(model, d, t, "patient_neg"),
                       "Sensitivity, % septic patients alerted": cell(rig, model, d, t, "pat_sens", ALERT, pct=True),
                       "Alert-hours per non-septic patient-day": cell(rig, model, d, t, "alert_hours_per_day", ALERT, nd=2),
                       "Non-septic patients alerted, %": cell(rig, model, d, t, "nonsep_pat_alarm_rate", ALERT, pct=True),
                       "Median lead time, h": cell(rig, model, d, t, "median_lead_h", ALERT, nd=1)})
TA, TB = pd.DataFrame(A_rows), pd.DataFrame(B_rows)
TA.to_csv(f"{R}/main_table_A.csv", index=False); TB.to_csv(f"{R}/main_table_B.csv", index=False)

def md(df):
    h = "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    return h + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values)

note = (
    "Values are the mean of five seeds, each seed with its own patient-level split and retrained models, followed by the 95% interval "
    "(percentile range of the seed-averaged patient-level bootstrap replicates, 200 per seed). Intervals reflect resampling of test patients; "
    "the spread across seeds is larger and is given as SD in supplementary_all_models.csv. "
    "Training hospital split 60/10/10/20% (train, isotonic, threshold and conformal calibration, in-site test); the other hospital contributes a 60% test set. "
    "AUROC and AUPRC are pooled over hourly rows. Calibration is measured on isotonic-recalibrated probabilities. "
    "Utility is normalised (silent model 0, oracle 1) at the threshold that maximises Utility on the training-hospital calibration patients. "
    "Alert results use the threshold giving 1 alert-hour per non-septic patient-day at the training hospital; sensitivity is the share of septic patients alerted before onset, "
    "lead time the median hours before onset of the first alert among alerted septic patients. "
    "Conformal coverage is patient-level class-conditional split conformal at a fixed prediction time of 24 h: one score per patient, the highest model score over hours up to 24, for patients still in the ICU at hour 24 with no sepsis onset by then. The patient-level outcome used to evaluate the sets is septic = sepsis onset within the next 24 h (hours 25 to 48), non-septic = no onset within those 24 h; it is a different quantity from the hourly training label (positive from 6 h before onset) and is used only as the evaluation label. A coverage guarantee applies in-site only (exchangeable eligible patients, that single time and outcome); values at the other hospital are measurements under shift. Interval from resampling test patients given the calibration draw."
)
open(f"{R}/main_tables.md", "w").write(
    "## Panel A. Discrimination, Utility and calibration\n\n" + md(TA) + "\n\n## Panel B. Conformal coverage and alert performance\n\n" + md(TB) + "\n\n" + note + "\n")


# ---------------------------------------------------------------- Table 2/3: every model, AUROC and Utility
NAMES = [("gbm_temporal", "GBM, values + mask + temporal"), ("gbm_both", "GBM, values + mask (primary)"), ("gbm_values", "GBM, values only"),
         ("gbm_mask", "GBM, mask and timing only"), ("mlp", "MLP, temporal features"), ("grud", "GRU-D (exploratory)"),
         ("gru", "GRU, no decay (exploratory)"), ("sirs", "SIRS"), ("qsofa", "qSOFA (partial)"), ("news", "NEWS (partial)"), ("sofa", "SOFA (partial)")]
def wide(metric, nd=3):
    out = []
    for m, nm in NAMES:
        row = {"Model": nm}
        for d, t in ROWS: row[site_label(d, t)] = cell(rig, m, d, t, metric, nd=nd)
        out.append(row)
    return pd.DataFrame(out)
T2, T3 = wide("auroc"), wide("utility_opt")
T2.to_csv(f"{R}/table2_auroc_all_models.csv", index=False); T3.to_csv(f"{R}/table3_utility_all_models.csv", index=False)

# ---------------------------------------------------------------- Table 4: conformal coverage by unit and class
T4r = []
for model, mname in MODELS:
    for d in ["A->B", "B->A"]:
        s_, t_ = d.split("->")
        for cls, tgt, key in [("septic", 0.80, "pos"), ("non-septic", 0.95, "neg")]:
            T4r.append({"Model": mname, "Direction and class": f"{s_}→{t_}, {cls}", "Target": f"{tgt:.2f}",
                        "Patient-level at 24 h, training hospital": conf(model, d, "in_site", f"patient_{key}"),
                        "Patient-level at 24 h, other hospital": conf(model, d, "other_site", f"patient_{key}"),
                        "Hours pooled, training hospital": conf(model, d, "in_site", f"hour_pooled_{key}"),
                        "Hours pooled, other hospital": conf(model, d, "other_site", f"hour_pooled_{key}")})
T4 = pd.DataFrame(T4r); T4.to_csv(f"{R}/table4_conformal_coverage.csv", index=False)

# ---------------------------------------------------------------- Tables A2/A3: fixed prediction time sensitivity and eligible patients
A2r = []
for model, mname in MODELS:
    for d in ["A->B", "B->A"]:
        s_, t_ = d.split("->")
        for cls, tgt, key in [("septic", 0.80, "pos"), ("non-septic", 0.95, "neg")]:
            row = {"Model": mname, "Direction and class": f"{s_}→{t_}, {cls}", "Target": f"{tgt:.2f}"}
            for T in (12, 24, 48):
                sf = "" if T == 24 else f"_T{T}"
                vi = cu[(cu.model == model) & (cu.direction == d) & (cu.test == "in_site") & (cu.metric == f"patient_{key}{sf}")].iloc[0]["mean"]
                vo = cu[(cu.model == model) & (cu.direction == d) & (cu.test == "other_site") & (cu.metric == f"patient_{key}{sf}")].iloc[0]["mean"]
                row[f"T = {T} h: training / other hospital"] = f"{vi:.2f} / {vo:.2f}"
            A2r.append(row)
A2 = pd.DataFrame(A2r); A2.to_csv(f"{R}/tableA2_fixed_time_sensitivity.csv", index=False)
cn = pd.read_csv(f"{R}/conformal_fixed_time_counts.csv"); A3r = []
for d in ["A->B", "B->A"]:
    s_, t_ = d.split("->")
    for T in (12, 24, 48):
        g1 = cn[(cn.model == "gbm_both") & (cn.direction == d) & (cn["T"] == T)]
        ins_ = g1[g1.test == "in_site"].iloc[0]; oth = g1[g1.test == "other_site"].iloc[0]
        A3r.append({"Direction": f"{s_}→{t_}", "Prediction time": f"{T} h",
                    "Calibration patients (septic / non-septic)": f"{ins_.n_cal_septic:.0f} / {ins_.n_cal_nonseptic:,.0f}",
                    "Test patients, training hospital (septic / non-septic)": f"{ins_.n_test_septic:.0f} / {ins_.n_test_nonseptic:,.0f}",
                    "Test patients, other hospital (septic / non-septic)": f"{oth.n_test_septic:.0f} / {oth.n_test_nonseptic:,.0f}"})
A3 = pd.DataFrame(A3r); A3.to_csv(f"{R}/tableA3_fixed_time_eligible_patients.csv", index=False)


# ---------------------------------------------------------------- Tables A4-A6: seed-level intervals, prediction-target sensitivity, reading of prediction sets
from scipy import stats as _sps
A4r = []
for model, mname in MODELS:
    for d in ["A->B", "B->A"]:
        s_, t_ = d.split("->")
        for cls, tgt, key in [("septic", 0.80, "pos"), ("non-septic", 0.95, "neg")]:
            row = {"Model": mname, "Direction and class": f"{s_}→{t_}, {cls}", "Target": f"{tgt:.2f}"}
            for test, lab in (("in_site", "training hospital"), ("other_site", "other hospital")):
                g = cu[(cu.model == model) & (cu.direction == d) & (cu.test == test) & (cu.metric == f"patient_{key}")].iloc[0]
                h = _sps.t.ppf(0.975, g.n_seeds - 1) * g.sd_seeds / np.sqrt(g.n_seeds)
                row[f"Patient-level at 24 h, {lab}: mean (seed SD; 95% interval over seeds)"] = f"{g['mean']:.2f} ({g.sd_seeds:.2f}; {max(g['mean'] - h, 0):.2f}–{min(g['mean'] + h, 1):.2f})"
            A4r.append(row)
A4 = pd.DataFrame(A4r); A4.to_csv(f"{R}/tableA4_seed_level_intervals.csv", index=False)
hz = pd.read_csv(f"{R}/conformal_horizon_sensitivity.csv"); A5r = []
for model, mname in MODELS:
    for d in ["A->B", "B->A"]:
        s_, t_ = d.split("->")
        ncal = {str(h): hz[(hz.model == model) & (hz.direction == d) & (hz.test == "in_site") & (hz.horizon_h.astype(str) == str(h))].iloc[0].n_cal_septic for h in ("6", "24", "rest of stay")}
        for cls, tgt, key in [("septic", 0.80, "septic"), ("non-septic", 0.95, "nonseptic")]:
            row = {"Model": mname, "Direction and class": f"{s_}→{t_}, {cls}", "Target": f"{tgt:.2f}",
                   "Septic calibration patients (6 h / 24 h / rest of stay)": f"{ncal['6']:.0f} / {ncal['24']:.0f} / {ncal['rest of stay']:.0f}"}
            for h, lab in (("6", "onset within 6 h of T (training-label horizon)"), ("24", "onset within 24 h of T (main)"), ("rest of stay", "onset any time after T")):
                vi = hz[(hz.model == model) & (hz.direction == d) & (hz.test == "in_site") & (hz.horizon_h.astype(str) == h)].iloc[0][f"cov_{key}_mean"]
                vo = hz[(hz.model == model) & (hz.direction == d) & (hz.test == "other_site") & (hz.horizon_h.astype(str) == h)].iloc[0][f"cov_{key}_mean"]
                row[f"Septic = {lab}: training / other hospital"] = f"{vi:.2f} / {vo:.2f}"
            A5r.append(row)
A5 = pd.DataFrame(A5r); A5.to_csv(f"{R}/tableA5_prediction_target_sensitivity.csv", index=False)
sd_ = pd.read_csv(f"{R}/conformal_set_distribution.csv"); A6r = []
for d in ["A->B", "B->A"]:
    s_, t_ = d.split("->")
    for test, lab in (("in_site", f"{s_} (training hospital)"), ("other_site", f"{t_} (other hospital)")):
        for cls in ("septic", "non-septic"):
            g = sd_[(sd_.model == "gbm_both") & (sd_.direction == d) & (sd_.test == test) & (sd_.true_class == cls)].set_index("prediction_set").share
            A6r.append({"Direction": f"{s_}→{t_}", "Test hospital": lab, "True class": cls, "{non-septic} only, %": f"{100 * g['{non-septic}']:.0f}",
                        "{septic, non-septic}, %": f"{100 * g['{septic, non-septic}']:.0f}", "{septic} only, %": f"{100 * g['{septic}']:.0f}"})
A6 = pd.DataFrame(A6r); A6.to_csv(f"{R}/tableA6_prediction_sets.csv", index=False)

# ---------------------------------------------------------------- Table 6/7: remedies with labeled target patients, label-free weighting
rem = pd.concat([pd.read_csv(f"{R}/remedy_summary_robust.csv"), pd.read_csv(f"{R}/grud_remedy_summary_robust.csv")], ignore_index=True)
def q(r, c, nd=2): return f"{r[c]:.{nd}f}"
T5r = []
for model, mname in MODELS:
    for d in ["A->B", "B->A"]:
        s_, t_ = d.split("->")
        sub = rem[(rem.model == model) & (rem.direction == d)]
        src = sub[(sub.method == "source_only") & (sub.test == "other_site")].iloc[0]
        T5r.append({"Model": mname, "Direction": f"{s_}→{t_}", "Labeled target patients": "0 (source only)", "Septic patients in sample": "0",
                    "Median calibration slope (IQR)": f"{q(src, 'cal_slope_median')}", "Septic-hour coverage, mean (SD)": q(src, "cov_pos_mean"),
                    "Non-septic-hour coverage, mean (SD)": q(src, "cov_neg_mean"), "Sensitivity after threshold reset": "", "Alert-hours per day after reset": ""})
        for _, r in sub[sub.method == "labeled_target"].sort_values("n_t").iterrows():
            if model == "grud" and int(r.n_t) not in (100, 800): continue
            T5r.append({"Model": mname, "Direction": f"{s_}→{t_}", "Labeled target patients": f"{int(r.n_t):,}", "Septic patients in sample": f"{r.n_sep_patients_mean:.0f}",
                        "Median calibration slope (IQR)": f"{r.platt_slope_median:.2f} ({r.platt_slope_q25:.2f}–{r.platt_slope_q75:.2f})",
                        "Septic-hour coverage, mean (SD)": f"{r.cp_tgt_cov_pos_mean:.2f} ({r.cp_tgt_cov_pos_sd:.2f})",
                        "Non-septic-hour coverage, mean (SD)": f"{r.cp_tgt_cov_neg_mean:.2f} ({r.cp_tgt_cov_neg_sd:.2f})",
                        "Sensitivity after threshold reset": f"{r.thr_tgt_sens_mean:.2f}", "Alert-hours per day after reset": f"{r.thr_tgt_alert_hours_mean:.1f}"})
T5 = pd.DataFrame(T5r); T5.to_csv(f"{R}/table6_labeled_target_remedies.csv", index=False)
T6r = []
for model, mname in MODELS:
    for d in ["A->B", "B->A"]:
        s_, t_ = d.split("->")
        sub = rem[(rem.model == model) & (rem.direction == d) & (rem.test == "other_site")]
        so = sub[sub.method == "source_only"].iloc[0]; wt = sub[sub.method == "weighted_cp_labelfree"].iloc[0]
        T6r.append({"Model": mname, "Direction": f"{s_}→{t_}",
                    "Septic-hour coverage, source only": q(so, "cov_pos_mean"), "Septic-hour coverage, weighted": q(wt, "cov_pos_mean"),
                    "Non-septic-hour coverage, source only": q(so, "cov_neg_mean"), "Non-septic-hour coverage, weighted": q(wt, "cov_neg_mean"),
                    "Effective sample size, septic calibration points": f"{wt.ess_pos_mean:.1f} of {wt.n_cal_pos_mean:.0f}"})
T6 = pd.DataFrame(T6r); T6.to_csv(f"{R}/table7_label_free_weighting.csv", index=False)

# ---------------------------------------------------------------- Table 5: guarantee check (exchangeable calibration/test patients)
gc = pd.read_csv(f"{R}/conformal_guarantee_check.csv"); gc_all = gc.copy(); gc = gc[gc["T"] == 24]
T7 = pd.DataFrame([{"Model": dict(MODELS)[r.model], "Source hospital": r.direction.split("->")[0], "Calibration patients": f"{int(r.n_cal_patients):,}",
                    "Septic coverage, mean (SD) over splits": f"{r.cov_septic_mean:.3f} ({r.cov_septic_sd:.3f})",
                    "Non-septic coverage, mean (SD) over splits": f"{r.cov_nonseptic_mean:.3f} ({r.cov_nonseptic_sd:.3f})"} for r in gc.itertuples()])
T7.to_csv(f"{R}/table5_guarantee_check.csv", index=False)

NOTE2 = ("Tables 6 and 7 come from the remedy experiments (hour-pooled coverage, five seeds x five random draws of the labeled sample; medians for calibration parameters, "
         "means and SD for coverage). They redraw the calibration hour independently of Table 4, so hour-pooled septic coverage can differ between tables by up to about 0.03, "
         "within the split-to-split SD. The source-only calibration slope in Table 6 is a median over seeds; Table 1 gives the mean.")
open(f"{R}/main_tables.md", "a").write(NOTE2 + "\n" + 
    "\n## Table 2. AUROC, all models\n\n" + md(T2) + "\n\n## Table 3. Normalised Utility, all models\n\n" + md(T3) +
    "\n\n## Table 4. Conformal coverage\n\n" + md(T4) + "\n\n## Table 5. Guarantee check\n\n" + md(T7) + "\n\n## Table A2. Fixed prediction time sensitivity (patient-level coverage, training hospital / other hospital)\n\n" + md(A2) + "\n\n## Table A3. Eligible patients by prediction time (gradient boosting, mean over seeds)\n\n" + md(A3) + "\n\n## Table A4. Seed-level intervals for patient-level coverage at 24 h\n\n" + md(A4) + "\n\n## Table A5. Prediction-target sensitivity (horizon of the septic class)\n\n" + md(A5) + "\n\n## Table A6. Prediction sets at 24 h (gradient boosting)\n\n" + md(A6) + "\n\n## Table 6. Labeled target patients\n\n" + md(T5) +
    "\n\n## Table 7. Label-free weighting\n\n" + md(T6) + "\n")

# ---------------------------------------------------------------- supplementary: every model, every metric
sup = rig.copy()
sup["n_test_patients"] = [sizes[(d, t)]["n_patients"] for d, t in zip(sup.direction, sup.test)]
sup["n_test_septic_patients"] = [sizes[(d, t)]["n_septic"] for d, t in zip(sup.direction, sup.test)]
sup["n_test_rows"] = [sizes[(d, t)]["n_rows"] for d, t in zip(sup.direction, sup.test)]
sup["source"] = "rigor_results"
c2 = cu.rename(columns={}).copy(); c2["op"] = "-"
c2["n_test_patients"] = [sizes[(d, t)]["n_patients"] for d, t in zip(c2.direction, c2.test)]
c2["n_test_septic_patients"] = [sizes[(d, t)]["n_septic"] for d, t in zip(c2.direction, c2.test)]
c2["n_test_rows"] = [sizes[(d, t)]["n_rows"] for d, t in zip(c2.direction, c2.test)]
c2["source"] = "conformal_unit_analysis"
cols = ["source", "direction", "model", "test", "metric", "op", "mean", "ci_lo", "ci_hi", "sd_seeds", "n_seeds", "n_test_patients", "n_test_septic_patients", "n_test_rows"]
pd.concat([sup[cols], c2[cols]], ignore_index=True).to_csv(f"{R}/supplementary_all_models.csv", index=False)
print("tables written to", R)
