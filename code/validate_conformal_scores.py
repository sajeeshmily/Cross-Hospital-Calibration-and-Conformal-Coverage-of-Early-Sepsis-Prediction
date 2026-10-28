#!/usr/bin/env python3
"""Validation of the fixed-time patient-level conformal score (run after conformal_unit_analysis.py / conformal_guarantee_check.py).

Unit: patients still in the ICU at the fixed prediction time T (default 24 h) with no sepsis onset by T. Score m_i = highest model score
over hours <= T, the same window for every patient. Outcome (evaluation label only, not the hourly training label): septic = onset in (T, T+H] (H = 24 h; --H 0 = any time after T); non-septic = no onset by T+H.

Checks
  1  deployable score   m is unchanged when (a) every score after T is overwritten and (b) the outcome labels are randomly permuted:
                        the score depends only on data up to T, never on the outcome.
  1b eligibility        eligible patients have a row at or after T; septic ones have onset > T; non-septic ones were never septic; classes are disjoint.
  2  prediction sets    sets formed at T from m alone reproduce the coverage numbers used in the tables.
  3  independent code   a plain-loop re-implementation of eligibility, score, quantile and coverage reproduces the vectorised result (seed 0, in-site).
  4  finite-sample      on synthetic exchangeable patients, class-conditional coverage >= 1 - alpha (Monte-Carlo tolerance), and
                        the quantile is +inf (full set) when calibration is too small.
  5  real exchangeable  results/conformal_guarantee_check.csv (random calibration/test splits of held-out eligible patients) >= target - 0.01 for every T.
  6  tables agree       conformal_unit_analysis.csv: in-site patient-level coverage (all T) is not below target by more than a Bonferroni-adjusted number of SE
                        over seeds (the guarantee is one-sided); cells beyond 2 SE are listed as INFO.
Exit status is non-zero if any check fails.

usage: python validate_conformal_scores.py --cache C --preds P --results R [--out TXT] [--T 24]"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_splits
from helpers2 import qhat, fixed_time_unit

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--results", required=True)
ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--T", type=int, default=24); ap.add_argument("--H", type=int, default=24)
a = ap.parse_args(); T = a.T; H = None if a.H <= 0 else a.H
log, fails = [], []


def check(name, ok, detail):
    log.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok: fails.append(name)
    print(log[-1], flush=True)


def prediction_set(m, q1, q0):
    """Set for each patient from the score alone: septic in set iff 1 - m <= q1; non-septic in set iff m <= q0."""
    return (1 - m) <= q1, m <= q0


S_, T_ = "A", "B"; tag = "AtoB"
meta = np.load(f"{a.cache}/{S_}_meta.npz", allow_pickle=True); loc, y, tt = meta["loc"].astype(int), meta["y"].astype(int), meta["t"].astype(int)
n = int(loc.max()) + 1; nt = int(np.load(f"{a.cache}/{T_}_meta.npz", allow_pickle=True)["loc"].max()) + 1
P = np.load(f"{a.preds}/{tag}_s{a.seed}.npz"); s = P["gbm_both_src"].astype(float); spl = make_splits(n, nt, a.seed)

# 1 ------------------------------------------------------------------ deployable score
m0, is_sep, is_non = fixed_time_unit(loc, tt, y, s, n, T, H)
s2 = s.copy(); late = tt > T; r = np.random.default_rng(1); s2[late] = r.random(late.sum())
m_late, _, _ = fixed_time_unit(loc, tt, y, s2, n, T, H)
y_perm = r.permutation(y); m_perm, _, _ = fixed_time_unit(loc, tt, y_perm, s, n, T, H)
whole = np.full(n, -np.inf); np.maximum.at(whole, loc, s); whole2 = np.full(n, -np.inf); np.maximum.at(whole2, loc, s2)
el = is_sep | is_non; n_changed = int(np.sum(np.abs(whole[el] - whole2[el]) > 1e-12))
check("1 deployable score", np.array_equal(m0, m_late) and np.array_equal(m0, m_perm) and n_changed > 0.2 * el.sum(),
      f"T={T}: m unchanged when all later scores are overwritten and when outcome labels are permuted (a whole-stay maximum changed for {n_changed} of {int(el.sum())} eligible patients)")
# 1b ------------------------------------------------------------------ eligibility
last = np.zeros(n); np.maximum.at(last, loc, tt.astype(float)); first = np.full(n, np.inf); k = y == 1; np.minimum.at(first, loc[k], tt[k].astype(float))
hor = np.inf if H is None else T + H
ok = bool(np.all(last[el] >= T) and np.all(first[is_sep] + 6 > T) and np.all(first[is_sep] + 6 <= hor)
          and np.all((~np.isfinite(first[is_non])) | (first[is_non] + 6 > hor)) and not np.any(is_sep & is_non) and np.all(np.isfinite(m0[el])))
check("1b eligibility and outcome", ok, f"T={T}, outcome horizon H={'rest of stay' if H is None else str(H) + ' h'}: eligible septic {int(is_sep.sum())}, eligible non-septic {int(is_non.sum())} of {n} patients; excluded: stay shorter than T or sepsis onset by T; septic = onset in (T, T+H], non-septic = no onset by T+H")
# 2/3 ------------------------------------------------------------------ sets from scores; independent re-implementation
cp, te = spl["cp"], spl["test_in"]
cs, cn = cp[is_sep[cp]], cp[is_non[cp]]; ts, tn = te[is_sep[te]], te[is_non[te]]
q1, q0 = qhat(1 - m0[cs], .20), qhat(m0[cn], .05)
in1, _ = prediction_set(m0[ts], q1, q0); _, in0 = prediction_set(m0[tn], q1, q0)
cov1, cov0 = in1.mean(), in0.mean()
rows_by = {}
for i in range(len(loc)): rows_by.setdefault(int(loc[i]), []).append((int(tt[i]), int(y[i]), float(s[i])))
def loop_unit(p):
    rs = rows_by[p]; lastp = max(h for h, _, _ in rs); fl = min([h for h, yy, _ in rs if yy == 1], default=None)
    if lastp < T: return None
    if fl is not None and fl + 6 <= T: return None
    hz = float("inf") if H is None else T + H
    cls = 1 if (fl is not None and fl + 6 <= hz) else 0
    return cls, max(sc for h, _, sc in rs if h <= T)
U = {p: loop_unit(p) for p in set(map(int, cp)) | set(map(int, te))}
L_cs = [p for p in map(int, cp) if U[p] and U[p][0] == 1]; L_cn = [p for p in map(int, cp) if U[p] and U[p][0] == 0]
L_ts = [p for p in map(int, te) if U[p] and U[p][0] == 1]; L_tn = [p for p in map(int, te) if U[p] and U[p][0] == 0]
sc1 = sorted(1 - U[p][1] for p in L_cs); sc0 = sorted(U[p][1] for p in L_cn)
ql1 = sc1[int(np.ceil((len(sc1) + 1) * .8)) - 1]; ql0 = sc0[int(np.ceil((len(sc0) + 1) * .95)) - 1]
l1 = np.mean([(1 - U[p][1]) <= ql1 for p in L_ts]); l0 = np.mean([U[p][1] <= ql0 for p in L_tn])
check("2/3 sets and independent code", len(L_cs) == len(cs) and len(L_tn) == len(tn) and abs(l1 - cov1) < 1e-12 and abs(l0 - cov0) < 1e-12 and abs(ql1 - q1) < 1e-12 and abs(ql0 - q0) < 1e-12,
      f"in-site seed {a.seed}, T={T}: septic {cov1:.4f} vs {l1:.4f}, non-septic {cov0:.4f} vs {l0:.4f} ({len(cs)} septic and {len(cn)} non-septic calibration patients)")
# 4 ------------------------------------------------------------------ synthetic finite-sample guarantee
r = np.random.default_rng(7); cov = {0.20: [], 0.05: []}
for _ in range(4000):
    nc = 200; sc = r.normal(size=nc); t_ = r.normal(size=50)
    for al in cov: cov[al].append(np.mean(t_ <= qhat(sc, al)))
check("4 synthetic coverage", all(np.mean(cov[al]) >= 1 - al - 0.003 for al in cov),
      "alpha 0.20: %.4f (>= 0.80), alpha 0.05: %.4f (>= 0.95)" % (np.mean(cov[0.20]), np.mean(cov[0.05])))
check("4b small calibration -> full set", np.isinf(qhat(np.arange(5.0), .05)) and np.isinf(qhat(np.array([]), .2)), "qhat returns +inf when ceil((n+1)(1-alpha)) > n")
# 5 ------------------------------------------------------------------ real exchangeable splits
g = pd.read_csv(f"{a.results}/conformal_guarantee_check.csv")
ok = bool((g.cov_septic_mean >= 0.80 - 0.01).all() and (g.cov_nonseptic_mean >= 0.95 - 0.01).all())
check("5 guarantee on held-out eligible patients", ok, "; ".join(f"{r.direction} {r.model} T={r.T}: septic {r.cov_septic_mean:.3f}, non-septic {r.cov_nonseptic_mean:.3f}" for r in g.itertuples()))
# 6 ------------------------------------------------------------------ table values
Rcu = pd.read_csv(f"{a.results}/conformal_unit_analysis.csv")
ins = Rcu[(Rcu.test == "in_site") & (Rcu.metric.str.startswith("patient_"))].copy()
ins["target"] = np.where(ins.metric.str.startswith("patient_pos"), .80, .95)
ins["dev"] = ins["mean"] - ins.target; ins["se"] = ins.sd_seeds / np.sqrt(ins.n_seeds)
from scipy.stats import norm
z = norm.ppf(1 - 0.05 / len(ins))                      # Bonferroni over all in-site cells (one-sided 5% family-wise)
under = ins[ins.dev < -z * ins.se]
check("6 in-site patient coverage not below target", under.empty,
      "guarantee is one-sided: no cell (%d cells, all T) below target by more than %.1f SE of the seed mean (Bonferroni); lowest margin %.3f, highest over-coverage %.3f" % (len(ins), z, ins.dev.min(), ins.dev.max()))
for r in ins[ins.dev < -2 * ins.se].itertuples():
    log.append(f"[INFO] {r.direction} {r.model} {r.metric}: in-site coverage {r.mean:.3f} is {-r.dev / r.se:.1f} SE below target (seed SD {r.sd_seeds:.3f}, {int(r.n_seeds)} seeds); expected by chance in a few of {len(ins)} cells")
txt = "\n".join(log) + f"\n\n{len(log) - len(fails)} of {len(log)} checks passed\n"
if a.out: open(a.out, "w").write(txt)
sys.exit(1 if fails else 0)
