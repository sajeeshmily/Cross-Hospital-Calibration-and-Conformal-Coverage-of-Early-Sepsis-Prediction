"""
Measurement-aware, uncertainty-calibrated early sepsis prediction
PhysioNet/CinC 2019 pipeline (baseline: gradient boosting; swap in GRU-D/transformer later).

Stages
  1. Load .psv files (training_setA / training_setB)
  2. Features: value (ffill), mask, time-since-observation, observation density
  3. Horizon labels
  4. Patient-level split: train / isotonic-cal / conformal-cal / in-site test; other site = external test
  5. Feature-set ablation (values / mask / both / both+admin)
  6. Calibration (isotonic) + stratified ECE/Brier by observation density
  7. Class-conditional Mondrian (density-stratified) split conformal, ONE ROW PER PATIENT PER CLASS
  8. Abstention vs. alert burden
  9. Test-time shuffle (mask vs values) for the shortcut study

Usage:
  python sepsis_pipeline.py --setA data/training_setA --setB data/training_setB \
      --H 0 --seeds 0 1 2 --a_pos 0.2 --a_neg 0.05 --out results.csv

Note: the Challenge Utility score is NOT reimplemented here. Use the official
evaluate_sepsis_score.py from the PhysioNet challenge for that.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

VARS = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2",
        "pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine",
        "Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium",
        "Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets"]
STATIC = ["Age", "Gender"]
ADMIN = ["Unit1", "Unit2", "HospAdmTime", "ICULOS"]   # possible proxies -> ablate
CAP = 48                                              # cap for time-since-observation (hours)

# ---------------------------------------------------------------- 1. load
def load_set(folder, site):
    frames = []
    for f in sorted(glob.glob(os.path.join(folder, "*.psv"))):
        d = pd.read_csv(f, sep="|")
        d["pid"] = f"{site}_{os.path.basename(f)[:-4]}"
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    num = [c for c in out.columns if c not in ("pid",)]
    out[num] = out[num].astype(np.float32)                 # memory: float32
    return out

# ---------------------------------------------------------------- 2-3. features + labels
def build(df, H=0):
    """H = extra hours of look-ahead on top of the Challenge label (which is already
    shifted 6 h before Sepsis-3 onset). H=0 reproduces the official label."""
    df = df.sort_values(["pid", "ICULOS"]).reset_index(drop=True)
    g = df.groupby("pid", sort=False)
    M = df[VARS].notna().astype(np.int8); M.columns = ["m_" + v for v in VARS]
    V = g[VARS].ffill().astype(np.float32); V.columns = ["v_" + v for v in VARS]
    hour = g.cumcount().values
    last = pd.DataFrame(np.where(M.values == 1, hour[:, None], np.nan), columns=VARS)
    last["pid"] = df["pid"].values
    last = last.groupby("pid", sort=False)[VARS].ffill()
    D = hour[:, None] - last.values
    D = np.log1p(np.where(np.isnan(D), CAP, np.minimum(D, CAP)))
    D = pd.DataFrame(D.astype(np.float32), columns=["d_" + v for v in VARS])
    dens = pd.Series(M.values.mean(1)).groupby(df["pid"].values) \
             .transform(lambda s: s.rolling(6, min_periods=1).mean())
    y = g["SepsisLabel"].transform(
        lambda s: s[::-1].rolling(H + 1, min_periods=1).max()[::-1]).astype(int)
    out = pd.concat([df[["pid"] + STATIC + ADMIN], V, M, D], axis=1)
    out["y"] = y.values; out["density"] = dens.values.astype(np.float32)
    del V, M, D, last, df
    return out

def feature_sets(cols):
    v = [c for c in cols if c.startswith("v_")]
    m = [c for c in cols if c.startswith("m_") or c.startswith("d_")]
    return {"values": v + STATIC, "mask": m + STATIC,
            "both": v + m + STATIC, "both+admin": v + m + STATIC + ADMIN}

# ---------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def cal_slope_intercept(y, p):
    if len(np.unique(y)) < 2: return np.nan, np.nan
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    if np.ptp(z) < 1e-9: return np.nan, np.nan
    lr = LogisticRegression(C=1e6, max_iter=300).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def metrics(y, p):
    ok = len(np.unique(y)) == 2
    sl, ic = cal_slope_intercept(y, p)
    return {"cal_slope": sl, "cal_intercept": ic,
            "auroc": roc_auc_score(y, p) if ok else np.nan,
            "auprc": average_precision_score(y, p) if ok else np.nan,
            "brier": brier_score_loss(y, p), "ece": ece(y, p), "n": len(y),
            "prevalence": y.mean()}

# ---------------------------------------------------------------- 7. class-conditional Mondrian conformal
# Guarantee targeted: P(y in C | y=k, density bin) >= 1 - alpha_k, with ONE calibration row per patient
# per class (the patient is the exchangeable unit). Class-conditional because rows are ~98% negative,
# so marginal coverage would be driven almost entirely by non-septic rows.
def qhat(scores, alpha):
    n = len(scores)
    if n == 0: return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]

def fit_cc(c1, c0, a_pos, a_neg, nb, by_density=True, min_n=20):
    g1, g0 = qhat(1 - c1["_p"].values, a_pos), qhat(c0["_p"].values, a_neg)
    q1, q0 = np.full(nb, g1), np.full(nb, g0)
    if by_density:
        for b in range(nb):
            s1, s0 = c1[c1["_b"] == b], c0[c0["_b"] == b]
            if len(s1) >= min_n: q1[b] = qhat(1 - s1["_p"].values, a_pos)
            if len(s0) >= min_n: q0[b] = qhat(s0["_p"].values, a_neg)
    return q1, q0

# ---------------------------------------------------------------- experiment
def run(dfa, dfb, fsname, seed, a_pos, a_neg, rows, shuffle=True):
    rng = np.random.default_rng(seed)
    pids = np.array(dfa["pid"].unique().tolist()); rng.shuffle(pids); n = len(pids)
    cut = [int(.6 * n), int(.7 * n), int(.8 * n)]
    part = {"train": pids[:cut[0]], "iso": pids[cut[0]:cut[1]],
            "cp": pids[cut[1]:cut[2]], "test_in": pids[cut[2]:]}
    sub = {k: dfa[dfa["pid"].isin(v)] for k, v in part.items()}
    sub["test_out"] = dfb
    feats = feature_sets(dfa.columns)[fsname]

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                         early_stopping=True, random_state=seed)
    clf.fit(sub["train"][feats], sub["train"]["y"])
    raw = {k: clf.predict_proba(d[feats])[:, 1] for k, d in sub.items() if k != "train"}

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(raw["iso"], sub["iso"]["y"])
    cal = {k: iso.predict(v) for k, v in raw.items()}

    edges = np.quantile(sub["train"]["density"], [1 / 3, 2 / 3]); nb = 3
    binof = lambda d: np.digitize(d["density"].values, edges)

    cdf = sub["cp"].assign(_p=cal["cp"], _b=binof(sub["cp"]))
    c1 = cdf[cdf.y == 1].groupby("pid").sample(1, random_state=seed)
    c0 = cdf[cdf.y == 0].groupby("pid").sample(1, random_state=seed)
    Q = {"mondrian": (*fit_cc(c1, c0, a_pos, a_neg, nb, True), True),
         "global": (*fit_cc(c1, c0, a_pos, a_neg, 1, False), False)}
    thr = np.quantile(cal["cp"][sub["cp"]["y"].values == 1], a_pos)   # threshold for (1-a_pos) sensitivity

    base = {"seed": seed, "featset": fsname, "a_pos": a_pos, "a_neg": a_neg}
    strata = ["all"] + [f"dens{b}" for b in range(nb)]
    for k in ["test_in", "test_out"]:
        d, y, bn = sub[k], sub[k]["y"].values, binof(sub[k])
        smask = lambda st: np.ones(len(y), bool) if st == "all" else bn == int(st[-1])
        for name, p in [("raw", raw[k]), ("isotonic", cal[k])]:
            for st in strata:
                m = smask(st)
                rows.append(dict(base, kind="calibration", test=k, calib=name, stratum=st,
                                 **metrics(y[m], p[m])))
        thr_alert = cal[k] >= thr
        for method, (q1, q0, use) in Q.items():
            b_ = bn if use else np.zeros_like(bn)
            inc1, inc0 = (1 - cal[k]) <= q1[b_], cal[k] <= q0[b_]
            abst, alert = inc1 & inc0, inc1 & ~inc0
            for st in strata:
                m = smask(st); yy = y[m]; pos, neg = yy == 1, yy == 0
                ok = lambda a: a[m][pos].mean() if pos.any() else np.nan
                fp = lambda a: 1000 * (a[m] & neg).sum() / max(m.sum(), 1)
                rows.append(dict(
                    base, kind="conformal", method=method, test=k, stratum=st,
                    n=int(m.sum()), n_pos=int(pos.sum()),
                    cov_pos=ok(inc1), cov_neg=inc0[m][neg].mean() if neg.any() else np.nan,
                    defer_rate=abst[m].mean(),
                    sens_conf_alert=ok(alert), fp_per_1000h_conf_alert=fp(alert),
                    sens_thr=ok(thr_alert), fp_per_1000h_thr=fp(thr_alert),
                    sens_thr_defer=ok(thr_alert & ~abst), fp_per_1000h_thr_defer=fp(thr_alert & ~abst)))

        if shuffle and k == "test_in":                        # RQ1: shortcut test
            for grp, cols in [("mask+delta", [c for c in feats if c[:2] in ("m_", "d_")]),
                              ("values", [c for c in feats if c.startswith("v_")])]:
                if not cols: continue
                x = d[feats].copy(); x[cols] = x[cols].values[rng.permutation(len(x))]
                pp = clf.predict_proba(x)[:, 1]
                rows.append(dict(base, kind="shuffle", test=k, calib="raw", stratum=f"shuffle_{grp}",
                                 **metrics(y, pp)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setA", required=True); ap.add_argument("--setB", required=True)
    ap.add_argument("--H", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--a_pos", type=float, default=0.2)   # 1 - target coverage on septic rows
    ap.add_argument("--a_neg", type=float, default=0.05)  # 1 - target coverage on non-septic rows
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    A = build(load_set(a.setA, "A"), a.H); B = build(load_set(a.setB, "B"), a.H)
    rows = []
    for src, dst, tag in [(A, B, "A->B"), (B, A, "B->A")]:
        for fs in ["values", "mask", "both", "both+admin"]:
            for s in a.seeds:
                n0 = len(rows); run(src, dst, fs, s, a.a_pos, a.a_neg, rows)
                for r in rows[n0:]: r["direction"] = tag
                print(f"done {tag} {fs} seed={s}")
    pd.DataFrame(rows).to_csv(a.out, index=False); print("saved", a.out)

if __name__ == "__main__":
    main()
