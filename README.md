[README.md](https://github.com/user-attachments/files/34061610/README.md)
# Study: rigor, baselines, shift remedies (PhysioNet/CinC 2019, sets A and B)


    python code/run_all.py --data <folder with training_setA/ and training_setB/> --work <run folder>
Runs prep -> training -> GRU-D -> merge -> rigor -> remedies -> figures; finished stages are skipped on re-run.
`--quick` is a 1-seed smoke test; `--dry-run` prints the commands; `--reuse-hp results/` skips boosting hyper-parameter tuning.
Needs python 3 with numpy, pandas, scikit-learn, matplotlib (no GPU, no deep-learning library). About 3-4 h on 2 CPUs with the defaults.
Outputs: `<run folder>/results` (tables), `figures`, `logs`.

`code/` is self-contained (`common.py` and `sepsis_pipeline.py` are included).
Manual run order (what run_all.py does) (paths are arguments/flags):
1. `python prep.py A <setA psv dir> <cache>`; same for B       -> features + metadata cache
2. `python train_predict.py "A->B" <cache> <preds> 0 1 2 3 4`; same for "B->A"   (about 7 min/seed/direction, 1 CPU each)
3. `python evaluate_rigor.py --dirs "A->B" --B 200 --out <res>`   (patient-level bootstrap, utility, alert burden)
4. `python remedy.py --dirs "A->B" --draws 5 --out <res>`; then `python summarize_remedy.py <res>`
5. `python make_figures.py <res> <figdir>`

## Protocol
* Source site: patients split 60/10/10/20 into train / isotonic-calibration / conformal+threshold calibration / in-site test.
* Other site: 40% labeled pool (remedies only) / 60% test. Test patients are never used for fitting anything.
* Models: HistGradientBoosting on values / mask+time-since / values+mask / +rolling temporal features
  (hyper-parameters tuned once per direction on seed-0 inner validation); sklearn MLP on temporal features;
  partial clinical scores (SIRS, qSOFA, NEWS, SOFA) from the available variables.
* CIs: 95% percentile interval of the mean over 5 seeds of per-seed patient-level bootstrap replicates (200 each).
* Utility: re-implementation of the challenge definition (dt_early=-12, dt_optimal=-6, dt_late=3; U_TP=1, U_FN=-2, U_FP=-0.05).
  Checked: never-alert = 0, oracle = 1, always-alert < 0, utility curve matches the manuscript. NOT run through the official script.
* Alert burden: threshold set on source calibration patients for a target number of alert-hours per non-septic patient-day.

## Known limitations 
* Sequence baselines: GRU-D (tuned) and a plain GRU, both pure numpy (PyTorch could not be installed here), see below. A PyTorch reference implementation and a transformer were not run.
* Clinical scores are PARTIAL (no GCS, no PaO2, no urine output, no supplemental-O2 flag; FiO2>0.21 used as proxy).
* Hyper-parameters were tuned on seed-0 training patients and reused for other seeds (small selection overlap with other seeds' test patients).
* AUROC/AUPRC/calibration are row-level (per patient-hour); patient-level metrics are the alert-burden ones.
* One bootstrap CI (gbm_mask, A->B other-site calibration slope, upper bound) is numerically degenerate; point estimate is fine.
* Platt/isotonic refits diverge for some small target samples (see *_diverged_frac in remedy_summary_robust.csv); use the medians.
* Two sites only. MIMIC-IV demo has too few sepsis cases to use.

## GRU-D baseline
`code/grud.py` is a numpy implementation of GRU-D (input and hidden-state decay; no torch available here), gradient-checked against finite differences (`python grud.py --check`). Hidden size 48, Adam, 30 epochs, best epoch by validation AUPRC on 10% of training patients, positive weight 5. Predictions are merged into the per-seed prediction files as `grud_src` / `grud_tgt`, then run through `evaluate_rigor.py --models grud --no_scores` and `remedy.py --models grud`. Results: `results/grud_rigor_results.csv`, `grud_remedy_summary_robust.csv`, `grud_remedy_raw.csv`. Hyperparameters were not tuned.

## GRU-D and plain GRU baselines (numpy, tuned)
`code/grud.py` trains GRU-D (input and hidden-state decay) in numpy with hand-written backpropagation (`python grud.py --check` runs the finite-difference gradient test). Hidden size, learning rate and positive-class weight are chosen from 6 random configurations on the seed-0 validation split (10% of training patients, AUPRC), then reused for all seeds. Selected: GRU-D hidden 48 (A->B) and 32 (B->A), lr 0.003, weight 5; plain GRU hidden 32, lr 0.003, weight 3 (A->B) and 5 (B->A). `GRUD_NAME=gru GRUD_DECAY=0` gives the plain GRU (same inputs, decay terms fixed at 0). Results are in `results/grud_*.csv` (both models); the earlier untuned GRU-D is kept in `results/archive_numpy_grud_untuned/`.
`code/grud_torch.py` from the collaborator's PyTorch version is not part of this archive's results.

## Conformal coverage by evaluation unit (added)
`code/conformal_unit_analysis.py` reports class-conditional coverage as hour-pooled, hour-patient (per-patient share, averaged) and patient-level (one score per patient at a fixed prediction time), with patient-level bootstrap intervals.

Patient-level unit (deployable): the prediction time T is fixed in advance (`--T 24` main, `--sens_T 12 48` sensitivity) and does not depend on the outcome. A patient is eligible if still in the ICU at T (last hour >= T) and has no onset by T (onset = first label-positive hour + 6). The score is m = highest model score over hours <= T (`helpers2.fixed_time_unit`), the same window for every patient; nonconformity is 1 - m (septic) and m (non-septic); the patient-level quantile comes from those same calibration-patient scores and is used for patient-level coverage. A prediction set can be formed at hour T from data up to T alone.

Two quantities are kept apart. The hourly TRAINING LABEL is the Challenge label (positive from 6 h before onset onward; the models are trained for onset within about 6 h). The PATIENT-LEVEL OUTCOME is used only to evaluate the sets: septic = sepsis onset in (T, T + H], H = 24 h (`--H 24`); non-septic = no onset in that window (never septic, later onset, or left the ICU). H = 6 h (the training-label horizon; too few septic calibration patients) and `--H 0` (onset any time after T) are sensitivity analyses (Table A5, `conformal_horizon_sensitivity.csv`). The 24 h choice was made after seeing the 6 h version was underpowered. The split-conformal guarantee is claimed only in-site, at the single time T, for eligible exchangeable patients; it is not a guarantee for hourly or continuous alerting. Coverage at the other hospital is an empirical measurement under shift. The hour-level coverages (hour-pooled, hour-patient) are a separate empirical analysis with their own procedure (one random calibration hour per patient per class). (Earlier versions used a label-defined window end or the whole-stay maximum, which depend on the outcome or use later data; neither is used. Old outputs are in `results/archive_label_dependent_patient_score/`; hour-level and non-septic-score-independent results are unchanged.)
`python code/conformal_unit_analysis.py --cache <cache> --preds <preds_all> --out results/conformal_unit_analysis.csv`
Outputs: `results/conformal_unit_analysis.csv` (patient_pos/neg at T = 24, `_T12` and `_T48` for the sensitivity analysis) and `results/conformal_fixed_time_counts.csv` (eligible and calibration patients by T). Tables A2 (`tableA2_fixed_time_sensitivity.csv`) and A3 (`tableA3_fixed_time_eligible_patients.csv`) are built from them. Calibration draws use their own random seed, so hour-pooled values can differ from `rigor`/`remedy` outputs by about 0.01.

`code/conformal_guarantee_check.py` splits held-out source patients eligible at T at random into calibration and test sets (2,000 times, calibration size scaled to the eligible pool) for T = 24, 12, 48 with the 24 h outcome and checks that class-conditional coverage averages to its target (`results/conformal_guarantee_check.csv`: septic about 0.81-0.82, non-septic 0.95 at T = 24).

`code/conformal_fixed_time_audit.py` is an independent plain-loop re-implementation (no `helpers2` functions) of eligibility, score, class-specific quantiles and coverage. It (1) reproduces every five-seed patient-level mean in `conformal_unit_analysis.csv` (both models, both directions, both sites, H = 24 h outcome; exits non-zero if the difference exceeds 1e-9; `results/conformal_audit_check.txt`), (2) reports the score scale and the implied thresholds on the patient-level score (`results/conformal_score_scale.csv`), (3) repeats the analysis with the septic class restricted to onset within 6 h or 24 h after T (`results/conformal_horizon_sensitivity.csv`; the hourly models are trained for onset within about 6 h, while the main class is onset at any later time in the stay), and (4) tabulates how prediction sets split into {non-septic}, {both} and {septic} (`results/conformal_set_distribution.csv`). Tables A4 (seed-level intervals), A5 (prediction-target sensitivity) and A6 (prediction sets) are built from these in `make_main_tables.py`. It runs in the `tables` stage before the table builders.

`code/remedy_reliability.py` (Table 8): draws within a seed share a labeled pool and a model, so the 95% intervals on the shares use the five seeds as the unit (t interval, 4 df, clipped to 0-100%); the earlier Wilson intervals that treated the 25 draws as independent remain in `remedy_reliability.csv` as `share_*_lo/_hi` and are too narrow.

`code/validate_conformal_scores.py` validates the fixed-time score (unchanged when all later scores are overwritten and when outcome labels are permuted, eligibility rules, sets built from the score alone, independent plain-loop re-implementation, synthetic finite-sample coverage, held-out guarantee, in-site coverage not below target beyond a Bonferroni-adjusted seed-noise bound over the 24 in-site cells) and writes `results/validation_conformal_scores.txt`; it is run at the end of the `tables` stage and exits non-zero on failure.

## Results tables (single source of truth)
`code/make_main_tables.py` builds Tables 1-7 of the paper (Table 8 comes from `remedy_reliability.py`; Tables M1 and M2, site characteristics and patient flow, from `make_site_tables.py`; Table A1 from `subgroup_calibration.csv`) from the result CSVs: `results/main_table_A.csv`, `main_table_B.csv`, `table2_auroc_all_models.csv`, `table3_utility_all_models.csv`, `table4_conformal_coverage.csv`, `table6_labeled_target_remedies.csv`, `table7_label_free_weighting.csv`, `table5_guarantee_check.csv`, `main_tables.md` (all tables with notes) and `supplementary_all_models.csv` (every model and metric, with SD across seeds and test sample sizes). `run_all.py --only tables` regenerates them. Every number in the abstract and discussion is read from these tables.

`code/remedy_reliability.py` summarises the 25 draws per cell of the labeled-target experiments (`results/remedy_reliability.csv`, `table8_remedy_reliability.csv`): septic patients per sample, spread of calibration slope and septic coverage, and the share of draws reaching each target (slope 0.8-1.2; septic coverage >=0.75 and >=0.80; non-septic >=0.90). The thresholds were set when the table was written and are not preregistered.


## Table numbering (final manuscript)

Tables appear in the paper in this order: 1 (main_table_A/B), 2 (table2), 3 (table3), 4 (table4), 5 (table5_guarantee_check), 6 (table6_labeled_target_remedies), 7 (table7_label_free_weighting), 8 (table8_remedy_reliability); M1 and M2 are tableM1_site_characteristics.csv and tableM2_patient_flow.csv; A1 is the subgroup slope table from subgroup_calibration.csv.
