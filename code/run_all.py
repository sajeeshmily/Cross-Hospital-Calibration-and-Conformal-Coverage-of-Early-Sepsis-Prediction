#!/usr/bin/env python3
"""Full experimental run: Cross-Hospital Calibration and Conformal Validity of Early Sepsis Prediction.

One command reproduces every table and figure from the raw PhysioNet/CinC 2019 files.

    python run_all.py --data /path/to/physionet2019 --work /path/to/run_dir

--data  folder holding training_setA/ and training_setB/ (the .psv files may sit in a sub-folder; it is found automatically)
--work  output folder; everything is written below it and finished stages are skipped on re-run

Stages (each skipped if its output exists):
  1 prep       features + metadata cache per site                       (prep.py)
  2 train      gradient boosting, MLP, density-ratio weights, 5 seeds   (train_predict.py)
  3 grud       numpy GRU-D (tuned) and plain GRU baselines, 5 seeds     (grud.py)
  4 merge      GRU-D predictions merged into the per-seed prediction files
  5 rigor      bootstrap metrics, Utility, calibration, alert burden    (evaluate_rigor.py)
  6 remedy     recalibration / conformal / threshold remedies           (remedy.py + summarize_remedy.py)
  7 figures    figures 1-5                                              (make_figures.py)
  8 tables     conformal coverage by unit, guarantee check, Tables 1-8, M1, M2  (conformal_unit_analysis.py, conformal_guarantee_check.py, remedy_reliability.py, make_main_tables.py, make_site_tables.py, validate_conformal_scores.py)

Typical cost on 2 CPUs / 8 GB RAM with the defaults: about 3-4 hours (stage 2 dominates; both directions run in parallel).
Options:  --seeds 0 1 2 3 4   --boot 200   --draws 5   --grud-epochs 30   --quick (smoke test)   --dry-run   --only prep train ...
Requirements: python 3, numpy, pandas, scikit-learn, matplotlib. No GPU or deep-learning library is needed.
Hyper-parameters of the boosting models are tuned once per direction on seed 0 (stage 2) and saved as hp_*.json;
pass --reuse-hp <dir with hp_*.json> to skip the tuning.
"""
import argparse, glob, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--grud-epochs", type=int, default=30)
ap.add_argument("--reuse-hp", default=None)
ap.add_argument("--quick", action="store_true", help="1 seed, 20 bootstrap replicates, 1 remedy draw, 3 GRU-D epochs")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--only", nargs="+", default=None, choices=["prep", "train", "grud", "merge", "rigor", "remedy", "figures", "tables"])
a = ap.parse_args()
if a.quick:
    a.seeds, a.boot, a.draws, a.grud_epochs = a.seeds[:1], 20, 1, 3
W = os.path.abspath(a.work)
D = {k: f"{W}/{k}" for k in ["cache", "preds", "preds_grud", "preds_gru", "preds_all", "logs", "results", "figures"]}
for d in D.values():
    os.makedirs(d, exist_ok=True)
DIRS = ["A->B", "B->A"]
TAG = {d: d.replace("->", "to") for d in DIRS}
seeds = [str(s) for s in a.seeds]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", GRUD_EPOCHS=str(a.grud_epochs))
if a.quick:
    env.update(GRUD_NCFG="1", GRUD_TUNE_EPOCHS="3")


def want(stage):
    return a.only is None or stage in a.only


def run_parallel(jobs):
    """jobs: list of (name, argv). Runs them concurrently, one log each, stops on the first failure."""
    procs = []
    for job in jobs:
        name, argv = job[0], job[1]
        print("  $", " ".join(argv), flush=True)
        if a.dry_run:
            continue
        procs.append((name, subprocess.Popen(argv, stdout=open(f"{D['logs']}/{name}.log", "w"), stderr=subprocess.STDOUT,
                                             env=dict(env, **(job[2] if len(job) > 2 else {})))))
    for name, p in procs:
        if p.wait() != 0:
            sys.exit(f"stage failed: {name} (see {D['logs']}/{name}.log)")


def find_psv_dir(site):
    root = f"{a.data}/training_set{site}"
    hits = glob.glob(f"{root}/**/*.psv", recursive=True)
    if not hits and not a.dry_run:
        sys.exit(f"no .psv files under {root}")
    return os.path.dirname(hits[0]) if hits else root


py = sys.executable
t0 = time.time()
print(f"work dir {W}; seeds {seeds}; boot {a.boot}; draws {a.draws}; grud epochs {a.grud_epochs}")

# 1 ---------------------------------------------------------------- prep
if want("prep"):
    print("[1/7] prep")
    jobs = [(f"prep_{s}", [py, f"{HERE}/prep.py", s, find_psv_dir(s), D["cache"]]) for s in "AB"
            if not os.path.exists(f"{D['cache']}/{s}_meta.npz")]
    run_parallel(jobs)

# 2 ---------------------------------------------------------------- train
if want("train"):
    print("[2/7] train")
    if a.reuse_hp:
        for d in DIRS:
            src = f"{a.reuse_hp}/hp_{TAG[d]}.json"
            if os.path.exists(src) and not a.dry_run:
                shutil.copy(src, f"{D['preds']}/hp_{TAG[d]}.json")
    run_parallel([(f"train_{TAG[d]}", [py, f"{HERE}/train_predict.py", d, D["cache"], D["preds"]] + seeds) for d in DIRS])

# 3 ---------------------------------------------------------------- grud
if want("grud"):
    print("[3/7] grud")
    # GRU-D: 6 random configurations tuned on the seed-0 validation split, then all seeds (grud.py)
    run_parallel([(f"grud_{TAG[d]}", [py, f"{HERE}/grud.py", d, D["cache"], D["preds_grud"]] + seeds) for d in DIRS])
    # plain GRU with masks, no decay terms (same code, same tuning)
    run_parallel([(f"gru_{TAG[d]}", [py, f"{HERE}/grud.py", d, D["cache"], D["preds_gru"]] + seeds,
                   {"GRUD_NAME": "gru", "GRUD_DECAY": "0"}) for d in DIRS])

# 4 ---------------------------------------------------------------- merge
if want("merge"):
    print("[4/7] merge GRU-D predictions into per-seed files")
    if not a.dry_run:
        for d in DIRS:
            for s in seeds:
                p = np.load(f"{D['preds']}/{TAG[d]}_s{s}.npz")
                out = dict(p.items())
                for pdir, key in [("preds_grud", "grud"), ("preds_gru", "gru")]:
                    g = np.load(f"{D[pdir]}/{TAG[d]}_s{s}.npz")
                    out[key + "_src"], out[key + "_tgt"] = g[key + "_src"], g[key + "_tgt"]
                np.savez_compressed(f"{D['preds_all']}/{TAG[d]}_s{s}.npz", **out)

# 5 ---------------------------------------------------------------- rigor
R = D["results"]
if want("rigor"):
    print("[5/7] rigor")
    common = ["--cache", D["cache"], "--preds", D["preds_all"], "--seeds"] + seeds + ["--B", str(a.boot)]
    run_parallel([(f"rigor_{TAG[d]}", [py, f"{HERE}/evaluate_rigor.py", "--dirs", d, "--out", f"{R}/rigor_{TAG[d]}"] + common) for d in DIRS])
    run_parallel([(f"rigor_grud_{TAG[d]}", [py, f"{HERE}/evaluate_rigor.py", "--dirs", d, "--models", "grud", "gru", "--no_scores",
                                            "--out", f"{R}/rigor_grud_{TAG[d]}"] + common) for d in DIRS])
    if not a.dry_run:
        cat = lambda pat, name: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat)) if os.path.exists(f)],
                                          ignore_index=True).to_csv(f"{R}/{name}", index=False)
        cat(f"{R}/rigor_[AB]to[AB]/rigor_results.csv", "rigor_results.csv")
        cat(f"{R}/rigor_grud_[AB]to[AB]/rigor_results.csv", "grud_rigor_results.csv")
        cat(f"{R}/rigor_[AB]to[AB]/reliability_gbm_both.csv", "reliability_gbm_both.csv")
        cat(f"{R}/rigor_[AB]to[AB]/subgroup_calibration.csv", "subgroup_calibration.csv")

# 6 ---------------------------------------------------------------- remedy
if want("remedy"):
    print("[6/7] remedy")
    common = ["--cache", D["cache"], "--preds", D["preds_all"], "--seeds"] + seeds + ["--draws", str(a.draws)]
    run_parallel([(f"remedy_{TAG[d]}", [py, f"{HERE}/remedy.py", "--dirs", d, "--models", "gbm_both", "mlp",
                                        "--out", f"{R}/remedy_{TAG[d]}"] + common) for d in DIRS])
    run_parallel([(f"remedy_grud_{TAG[d]}", [py, f"{HERE}/remedy.py", "--dirs", d, "--models", "grud", "gru",
                                             "--out", f"{R}/remedy_grud_{TAG[d]}"] + common) for d in DIRS])
    if not a.dry_run:
        for name, pat in [("remedy_main", "remedy_[AB]to[AB]"), ("remedy_grud", "remedy_grud_[AB]to[AB]")]:
            os.makedirs(f"{R}/{name}", exist_ok=True)
            pd.concat([pd.read_csv(f"{x}/remedy_raw.csv") for x in sorted(glob.glob(f"{R}/{pat}"))], ignore_index=True) \
                .to_csv(f"{R}/{name}/remedy_raw.csv", index=False)
            subprocess.run([py, f"{HERE}/summarize_remedy.py", f"{R}/{name}"], check=True)
        shutil.copy(f"{R}/remedy_main/remedy_raw.csv", f"{R}/remedy_raw.csv")
        shutil.copy(f"{R}/remedy_main/remedy_summary_robust.csv", f"{R}/remedy_summary_robust.csv")
        shutil.copy(f"{R}/remedy_grud/remedy_raw.csv", f"{R}/grud_remedy_raw.csv")
        shutil.copy(f"{R}/remedy_grud/remedy_summary_robust.csv", f"{R}/grud_remedy_summary_robust.csv")

# 7 ---------------------------------------------------------------- figures
if want("figures"):
    print("[7/7] figures")
    run_parallel([("figures", [py, f"{HERE}/make_figures.py", R, D["figures"]])])

# 8 ---------------------------------------------------------------- tables
if want("tables"):
    print("[8/8] tables")
    run_parallel([("conformal_units", [py, f"{HERE}/conformal_unit_analysis.py", "--cache", D["cache"], "--preds", D["preds_all"], "--out", f"{R}/conformal_unit_analysis.csv",
                                       "--seeds"] + seeds + ["--B", str(a.boot)]),
                  ("conformal_guarantee", [py, f"{HERE}/conformal_guarantee_check.py", "--cache", D["cache"], "--preds", D["preds_all"], "--out", f"{R}/conformal_guarantee_check.csv"]),
                  ("remedy_reliability", [py, f"{HERE}/remedy_reliability.py", "--results", R])])
    run_parallel([("conformal_audit", [py, f"{HERE}/conformal_fixed_time_audit.py", "--cache", D["cache"], "--preds", D["preds_all"], "--results", R] + ["--seeds"] + seeds)])
    run_parallel([("main_tables", [py, f"{HERE}/make_main_tables.py", "--results", R, "--cache", D["cache"], "--seeds"] + seeds),
                  ("site_tables", [py, f"{HERE}/make_site_tables.py", "--results", R, "--cache", D["cache"], "--seeds"] + seeds)])
    # validation of the patient-level conformal score (fails the run if any check fails)
    run_parallel([("validate_conformal", [py, f"{HERE}/validate_conformal_scores.py", "--cache", D["cache"], "--preds", D["preds_all"], "--results", R,
                                          "--out", f"{R}/validation_conformal_scores.txt"])])

print(f"done in {(time.time() - t0) / 60:.1f} min. Results: {R}   Figures: {D['figures']}   Logs: {D['logs']}")
