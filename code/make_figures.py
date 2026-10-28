"""Figures from rigor_results.csv, reliability_gbm_both.csv, remedy_summary.csv.
usage: python make_figures.py <results dir> <figure dir>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, fig = sys.argv[1], sys.argv[2]; os.makedirs(fig, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C_IN, C_OUT = "#2a6fbb", "#d9822b"                    # blue / orange: separable under common colour-vision deficiencies
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
LBL = {"gbm_both": "GBM values+mask", "gbm_temporal": "GBM + temporal", "gbm_values": "GBM values only",
       "gbm_mask": "GBM mask only", "mlp": "MLP (temporal)", "news": "NEWS (partial)", "sirs": "SIRS",
       "sofa": "SOFA (partial)", "qsofa": "qSOFA (partial)", "grud": "GRU-D (exploratory)", "gru": "GRU (exploratory, no decay)"}
ORDER = ["gbm_temporal", "gbm_both", "gbm_values", "grud", "gru", "mlp", "gbm_mask", "news", "sirs", "sofa", "qsofa"]
R = pd.read_csv(f"{res}/rigor_results.csv")
if os.path.exists(f"{res}/grud_rigor_results.csv"): R = pd.concat([R, pd.read_csv(f"{res}/grud_rigor_results.csv")], ignore_index=True)
R["op"] = R["op"].fillna("")

def dot_panel(metric, title, fname, op=""):
    f, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, d in zip(axs, ["A->B", "B->A"]):
        for j, (t, c, off) in enumerate([("in_site", C_IN, -0.12), ("other_site", C_OUT, 0.12)]):
            x = R[(R.direction == d) & (R.test == t) & (R.metric == metric) & (R.op == op)].set_index("model")
            ys = [i for i, m in enumerate(ORDER) if m in x.index]; ms = [m for m in ORDER if m in x.index]
            ax.errorbar(x.loc[ms, "mean"], np.array(ys) + off, xerr=[x.loc[ms, "mean"] - x.loc[ms, "ci_lo"],
                        x.loc[ms, "ci_hi"] - x.loc[ms, "mean"]], fmt="o", ms=5, color=c, elinewidth=1.2, capsize=0,
                        label="same hospital" if t == "in_site" else "other hospital")
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LBL[m] for m in ORDER]); ax.invert_yaxis()
        ax.set_title(f"trained on {d.split('->')[0]}, tested on {d.split('->')[0]} / {d.split('->')[1]}", fontsize=9)
        ax.set_xlabel(title)
    h, l = axs[0].get_legend_handles_labels(); f.legend(h, l, frameon=False, loc="lower center", ncol=2)
    f.tight_layout(rect=(0, 0.07, 1, 1)); f.savefig(f"{fig}/{fname}", dpi=200); plt.close(f)

dot_panel("auroc", "AUROC (row-level, 95% patient-bootstrap CI)", "fig1_auroc.png")
dot_panel("utility_opt", "Normalised Utility (threshold tuned on source cp split)", "fig2_utility.png")

# reliability
rl = pd.read_csv(f"{res}/reliability_gbm_both.csv"); rl = rl[rl.n > 200]
f, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharex=True, sharey=True)
for ax, d in zip(axs, ["A->B", "B->A"]):
    for t, c, lab in [("in_site", C_IN, "same hospital"), ("other_site", C_OUT, "other hospital")]:
        x = rl[(rl.direction == d) & (rl.test == t)]
        ax.plot(x.mean_pred, x.obs, "-o", ms=4, lw=1.4, color=c, label=lab)
    ax.plot([0.002, 0.5], [0.002, 0.5], color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("mean predicted probability (per hour)")
    ax.set_title(f"train {d.split('->')[0]}", fontsize=9)
axs[0].set_ylabel("observed frequency"); axs[0].legend(frameon=False); f.tight_layout()
f.savefig(f"{fig}/fig3_reliability.png", dpi=200); plt.close(f)

# alert burden at ops1.0
f, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, (metric, ttl) in zip(axs, [("alert_hours_per_day", "alert-hours per non-septic patient-day"),
                                  ("nonsep_pat_alarm_rate", "share of non-septic patients with any alert")]):
    for d_i, d in enumerate(["A->B", "B->A"]):
        for t, c, off in [("in_site", C_IN, -0.18), ("other_site", C_OUT, 0.18)]:
            x = R[(R.direction == d) & (R.test == t) & (R.metric == metric) & (R.op == "ops1.0") & (R.model == "gbm_both")]
            m = x["mean"].mean(); lo, hi = x.ci_lo.mean(), x.ci_hi.mean()
            ax.bar(d_i + off, m, width=0.3, color=c, label=("same hospital" if t == "in_site" else "other hospital") if d_i == 0 else None)
            ax.errorbar(d_i + off, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["train A", "train B"]); ax.set_title(ttl, fontsize=9)
axs[0].legend(frameon=False, loc="upper left"); f.suptitle("Same alert threshold (1 alert-hour/day on source cp split); GBM values+mask model", fontsize=8, color=MUTED)
f.tight_layout(); f.savefig(f"{fig}/fig4_alert_burden.png", dpi=200); plt.close(f)

# remedies
S = pd.read_csv(f"{res}/remedy_summary_robust.csv"); S = S[S.model == "gbm_both"]
f, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for d, ls in [("A->B", "-"), ("B->A", "--")]:
    x = S[(S.direction == d) & (S.method == "labeled_target")].sort_values("n_t"); n = x.n_t
    ref = S[(S.direction == d) & (S.method == "source_only") & (S.test == "other_site")].iloc[0]
    for ax, (col, rcol, ttl, tgt) in zip(axs, [("platt_slope_median", "cal_slope_median", "calibration slope on target test (median)", 1.0),
                                              ("cp_tgt_cov_pos_mean", "cov_pos_mean", "septic-row coverage (target 0.80)", 0.8),
                                              ("cp_tgt_cov_neg_mean", "cov_neg_mean", "non-septic-row coverage (target 0.95)", 0.95)]):
        ax.plot(n, x[col], ls, color=C_IN, lw=1.5, marker="o", ms=4, label=f"recalibrated on n target patients ({d})")
        ax.axhline(ref[rcol], color=C_OUT, ls=ls, lw=1.2, label=f"source-only ({d})")
        ax.axhline(tgt, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("labeled target patients"); ax.set_title(ttl, fontsize=9)
h, l = axs[0].get_legend_handles_labels(); f.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=8)
f.tight_layout(rect=(0, 0.1, 1, 1)); f.savefig(f"{fig}/fig5_remedies.png", dpi=200); plt.close(f)
print("figures ->", fig)
