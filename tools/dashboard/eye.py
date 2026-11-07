"""Eye-tracking features per trial from gaze_preproc.csv.gz + blinks -> JSON for the dashboard."""
import json
import sys
import time

import numpy as np
import pandas as pd

gaze_path, blinks_path, behav_path, out_path = sys.argv[1:5]
t0 = time.time()

behav = pd.read_csv(behav_path, usecols=["Experiment", "Subject", "TrialNum", "Hero", "MW", "MW_Estimate", "Tracking", "Pref_relative"])

# ---- 1-second aggregates, accumulated chunk by chunk
parts = []
cols = ["Subject", "TrialNum", "TimePoint", "LeftX", "LeftY", "LeftPupil", "RightPupil"]
for i, ch in enumerate(pd.read_csv(gaze_path, usecols=cols, chunksize=2_000_000,
                                   dtype={"Subject": "category", "TrialNum": "int16"})):
    ch["sec"] = (ch.TimePoint // 1000).astype("int32")
    ch["pupil"] = ch.LeftPupil.where(ch.LeftPupil.notna(), ch.RightPupil)   # left eye, right as fallback
    ch["p2"] = ch.pupil ** 2
    ch["has_p"] = ch.pupil.notna().astype("int32")
    ch["x2"] = ch.LeftX ** 2
    ch["y2"] = ch.LeftY ** 2
    ch["has_g"] = ch.LeftX.notna().astype("int32")
    ch["n"] = 1
    g = ch.groupby(["Subject", "TrialNum", "sec"], observed=True).agg(
        n=("n", "sum"), has_p=("has_p", "sum"), p=("pupil", "sum"), p2=("p2", "sum"),
        has_g=("has_g", "sum"), x=("LeftX", "sum"), x2=("x2", "sum"), y=("LeftY", "sum"), y2=("y2", "sum"),
        tmin=("TimePoint", "min"))
    parts.append(g.reset_index())
    print(f"chunk {i} rows={len(ch):,} elapsed={time.time() - t0:.0f}s", flush=True)

agg = pd.concat(parts).groupby(["Subject", "TrialNum", "sec"], observed=True).sum().reset_index()
agg["Subject"] = agg.Subject.astype(str)
agg["t_start"] = agg.groupby(["Subject", "TrialNum"]).tmin.transform("min")
agg["rel"] = ((agg.tmin - agg.t_start) // 1000).astype(int)
agg["pupil_mean"] = agg.p / agg.has_p.replace(0, np.nan)
print("seconds table", agg.shape, f"{time.time() - t0:.0f}s", flush=True)

# ---- per-trial features
tr = agg.groupby(["Subject", "TrialNum"]).agg(n=("n", "sum"), has_p=("has_p", "sum"), p=("p", "sum"), p2=("p2", "sum"),
                                              has_g=("has_g", "sum"), x=("x", "sum"), x2=("x2", "sum"), y=("y", "sum"), y2=("y2", "sum"),
                                              dur=("rel", "max")).reset_index()
tr["pupil_mean"] = tr.p / tr.has_p.replace(0, np.nan)
tr["pupil_sd"] = np.sqrt((tr.p2 / tr.has_p.replace(0, np.nan)) - tr.pupil_mean ** 2)
tr["missing"] = 1 - tr.has_p / tr.n
tr["gaze_sd"] = np.sqrt(np.clip((tr.x2 / tr.has_g.replace(0, np.nan)) - (tr.x / tr.has_g.replace(0, np.nan)) ** 2, 0, None)
                        + np.clip((tr.y2 / tr.has_g.replace(0, np.nan)) - (tr.y / tr.has_g.replace(0, np.nan)) ** 2, 0, None))
tr.loc[tr.missing > 0.5, ["pupil_mean", "pupil_sd"]] = np.nan   # shortcut: >50 % missing -> trial dropped from pupil stats

# blinks: left eye, drop single-sample artefacts (Duration == 2)
bl = pd.read_csv(blinks_path)
bl = bl[(bl.Eye == "L") & (bl.Duration > 2)]
bc = bl.groupby(["Subject", "TrialNum"]).agg(blinks=("Duration", "size"), blink_dur=("Duration", "mean")).reset_index()
tr = tr.merge(bc, on=["Subject", "TrialNum"], how="left")
tr["blinks"] = tr.blinks.fillna(0)
tr["blink_rate"] = tr.blinks / ((tr.dur + 1) / 60)   # per minute

tr = tr.merge(behav, on=["Subject", "TrialNum"], how="inner")   # MIGO2607 has eye data but no behavioural row

# within-subject z-scores
for c in ["pupil_mean", "pupil_sd", "gaze_sd", "blink_rate"]:
    tr[c + "_z"] = tr.groupby("Subject")[c].transform(lambda x: (x - x.mean()) / x.std(ddof=1))


def f(x, d=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def r(a, b):
    m = pd.DataFrame({"a": a, "b": b}).dropna()
    return {"r": f(np.corrcoef(m.a, m.b)[0, 1]), "n": int(len(m))} if len(m) > 3 else None


out = {"n_trials": int(len(tr)), "n_subjects": int(tr.Subject.nunique()),
       "seconds_elapsed": int(time.time() - t0)}

# correlations with MW (pooled, within-subject z) and per-subject correlation distribution
feats = ["pupil_mean", "pupil_sd", "gaze_sd", "blink_rate"]
out["corr"] = {c: r(tr[c + "_z"], tr.MW_Estimate) for c in feats}
within = {}
for c in feats:
    rs = []
    for s, g in tr.groupby("Subject"):
        m = g[[c + "_z", "MW_Estimate"]].dropna()
        if len(m) >= 10:
            rs.append({"id": s, "r": f(np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1])})
    rs.sort(key=lambda x: x["r"])
    within[c] = {"per_subject": rs, "mean_r": f(np.mean([x["r"] for x in rs])), "n_neg": int(sum(1 for x in rs if x["r"] < 0)), "n": len(rs)}
out["within"] = within

# features by MW bin (raw MW)
bins = [-1, 10, 25, 50, 75, 100]
labels = ["0-10", "11-25", "26-50", "51-75", "76-99"]
tr["mwbin"] = pd.cut(tr.MW, bins)
out["by_mw"] = []
for (iv, g), lab in zip(tr.groupby("mwbin", observed=False), labels):
    row = {"label": lab, "n": int(len(g))}
    for c in feats:
        v = g[c + "_z"].dropna()
        row[c] = f(v.mean()); row[c + "_sem"] = f(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
    out["by_mw"].append(row)

# tracked vs untracked and over trials
e1 = tr[tr.Experiment == "Exp1_Cued"]
out["tracking"] = {c: {k: f(e1[e1.Tracking == k][c + "_z"].mean()) for k in ["TRACKED", "UNTRACKED"]} for c in feats}
out["over_trials"] = {c: [f(v) for v in tr.groupby("TrialNum")[c + "_z"].mean()] for c in feats}
out["missing_by_subject"] = sorted([{"id": s, "missing": f(g.missing.mean()), "exp": g.Experiment.iloc[0]} for s, g in tr.groupby("Subject")], key=lambda x: -x["missing"])

# ---- average pupil timecourse (z within subject on 1-s means) for high vs low MW trials
agg = agg.merge(tr[["Subject", "TrialNum", "MW_Estimate", "MW", "missing"]], on=["Subject", "TrialNum"], how="left")
agg = agg[agg.missing <= 0.5]
agg["pz"] = agg.groupby("Subject").pupil_mean.transform(lambda x: (x - x.mean()) / x.std(ddof=1))
agg["mwgrp"] = np.where(agg.MW_Estimate > 0.5, "high", np.where(agg.MW_Estimate < -0.5, "low", "mid"))
tc = {}
for grp, g in agg[agg.rel <= 50].groupby("mwgrp"):
    s = g.groupby("rel").pz
    tc[grp] = {"mean": [f(v) for v in s.mean()], "sem": [f(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0 for _, x in s],
               "n_trials": int(g.groupby(["Subject", "TrialNum"]).ngroups)}
out["timecourse"] = tc

# ---- per-trial pupil at 1 Hz for the trial viewer (raw a.u., rounded) + blink times
pt = {}
for (s, t), g in agg.groupby(["Subject", "TrialNum"]):
    g = g.sort_values("rel")
    arr = [None] * (int(g.rel.max()) + 1)
    for rel, v in zip(g.rel, g.pupil_mean):
        arr[int(rel)] = f(v, 0)
    pt[f"{s}|{t}"] = arr
out["pupil_1hz"] = pt
blk = {}
starts = agg.groupby(["Subject", "TrialNum"]).t_start.min().reset_index()
bl = bl.merge(starts, on=["Subject", "TrialNum"], how="inner")
for (s, t), g in bl.groupby(["Subject", "TrialNum"]):
    blk[f"{s}|{t}"] = [[f((b - st) / 1000, 1), f(d / 1000, 2)] for b, d, st in zip(g.Beg, g.Duration, g.t_start)]
out["blinks_1"] = blk
out["trial_feats"] = {f"{r_.Subject}|{r_.TrialNum}": {"pm": f(r_.pupil_mean, 0), "pz": f(r_.pupil_mean_z, 2), "gsd": f(r_.gaze_sd, 0), "br": f(r_.blink_rate, 1), "miss": f(r_.missing, 2)}
                      for r_ in tr.itertuples()}

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print("done", f"{time.time() - t0:.0f}s", "->", out_path)
print("corr", out["corr"])
print("within", {c: {k: v for k, v in w.items() if k != "per_subject"} for c, w in within.items()})
print("by_mw", out["by_mw"])
print("tracking", out["tracking"])
print("timecourse n", {k: v["n_trials"] for k, v in tc.items()})
print("worst missing", out["missing_by_subject"][:6])
