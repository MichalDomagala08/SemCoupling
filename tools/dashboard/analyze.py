"""Behavioural analysis of behavioral_cleaned.csv -> JSON for the HTML dashboard."""
import ast
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

csv_path, stop_path, out_path = sys.argv[1:4]

df = pd.read_csv(csv_path)
stop = {l.strip().lower() for l in open(stop_path, encoding="utf-8") if l.strip()}


def as_list(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return []


df["LemmasL"] = df["Lemmas"].apply(as_list)
df["StoryLemmasL"] = df["StoryLemmas"].apply(as_list)
df["has"] = df["has_thoughts"] == 1
df["words"] = df["Utterance"].apply(lambda u: len(str(u).split()) if isinstance(u, str) else np.nan)

PRON = {"ja", "mnie", "mi", "mną", "moim", "mój", "moją", "moimi", "moja", "moje", "moich", "mojego", "mojej", "sobie", "siebie"}
PAST_1SG = re.compile(r"\w+(łem|łam)$")


def self_refs(u):
    if not isinstance(u, str):
        return np.nan
    toks = [t.strip("';:.,!?\"()").lower() for t in u.split()]
    toks = [t for t in toks if t]
    n = sum(1 for t in toks if t in PRON or PAST_1SG.match(t))
    return n


def self_rate(u):
    if not isinstance(u, str):
        return np.nan
    n = len(u.split())
    return self_refs(u) / n if n else np.nan


df["self_n"] = df["Utterance"].apply(self_refs)
df["self_rate"] = df["Utterance"].apply(self_rate)


def content(lems):
    return {l.lower() for l in lems if l.lower() not in stop and len(l) > 2}


def coupling(row):
    if not row["has"]:
        return np.nan
    u = content(row["LemmasL"])
    s = content(row["StoryLemmasL"])
    if not u:
        return np.nan
    return len(u & s) / len(u)


df["coupling"] = df.apply(coupling, axis=1)


def sem(x):
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def r(a, b):
    m = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(m) < 3:
        return None
    return {"r": float(np.corrcoef(m.a, m.b)[0, 1]), "n": int(len(m))}


def f(x, d=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


out = {}

# ---- meta
out["meta"] = {
    "n_rows": int(len(df)),
    "n_subjects": int(df.Subject.nunique()),
    "n_trials": int(df.TrialNum.nunique()),
    "n_recalls": int(df.has.sum()),
    "exp": {},
}
for e, g in df.groupby("Experiment"):
    out["meta"]["exp"][e] = {
        "subjects": int(g.Subject.nunique()),
        "recalls": int(g.has.sum()),
        "subjects_with_recalls": int(g[g.has].Subject.nunique()),
        "mean_mw": f(g.MW.mean(), 1),
        "median_words": f(g.words.median(), 1),
    }

# ---- MW over trials, per experiment and per hero
curves = {}
for e, g in df.groupby("Experiment"):
    s = g.groupby("TrialNum").MW
    curves[e] = {"mean": [f(v, 2) for v in s.mean()], "sem": [f(sem(x), 2) for _, x in s]}
    for h, gh in g.groupby("Hero"):
        sh = gh.groupby("TrialNum").MW
        curves[f"{e}|{h}"] = {"trials": [int(t) for t in sh.mean().index], "mean": [f(v, 2) for v in sh.mean()]}
out["curves"] = curves

# ---- Per-fragment table (story + aggregates)
frags = []
for (t, fr, h), g in df.groupby(["TrialNum", "Fragment", "Hero"]):
    e1 = g[g.Experiment == "Exp1_Cued"]
    e2 = g[g.Experiment == "Exp2_Preference"]
    frags.append({
        "trial": int(t), "fragment": fr, "hero": h,
        "story": g.StoryFrag.iloc[0],
        "story_words": len(str(g.StoryFrag.iloc[0]).split()),
        "mw_all": f(g.MW.mean(), 1), "mw_z": f(g.MW_Estimate.mean(), 3),
        "mw_exp1": f(e1.MW.mean(), 1), "mw_exp2": f(e2.MW.mean(), 1),
        "mw_tracked": f(e1[e1.Tracking == "TRACKED"].MW.mean(), 1),
        "mw_untracked": f(e1[e1.Tracking == "UNTRACKED"].MW.mean(), 1),
        "pref": f(e2.Preference.mean(), 1),
        "pref_z": f(e2.Pref_relative.mean(), 3),
        "n_recalls": int(g.has.sum()),
        "words": f(g.words.mean(), 1),
        "coupling": f(g.coupling.mean(), 3),
        "self_rate": f(g.self_rate.mean(), 3),
        "no_thoughts": f(g.no_discernable_thoughts.mean(), 3),
    })
out["fragments"] = frags

# ---- Subjects
subs = []
for s, g in df.groupby("Subject"):
    g = g.sort_values("TrialNum")
    subs.append({
        "id": s, "exp": g.Experiment.iloc[0],
        "mean_mw": f(g.MW.mean(), 1), "sd_mw": f(g.MW.std(), 1),
        "n_recalls": int(g.has.sum()),
        "mw": [f(v, 0) for v in g.MW],
        "words": f(g.words.mean(), 1),
        "coupling": f(g.coupling.mean(), 3),
        "tracked_hero": (g[g.Tracking == "TRACKED"].Hero.iloc[0] if (g.Tracking == "TRACKED").any() else None),
    })
subs.sort(key=lambda x: x["mean_mw"])
out["subjects"] = subs

# ---- Exp1: tracked vs untracked, paired per subject
e1 = df[df.Experiment == "Exp1_Cued"]
pairs = []
for s, g in e1.groupby("Subject"):
    tr = g[g.Tracking == "TRACKED"].MW.mean()
    un = g[g.Tracking == "UNTRACKED"].MW.mean()
    pairs.append({"id": s, "tracked": f(tr, 1), "untracked": f(un, 1), "diff": f(un - tr, 1),
                  "tracked_hero": g[g.Tracking == "TRACKED"].Hero.iloc[0]})
diffs = np.array([p["diff"] for p in pairs])
from scipy import stats
t_tr = stats.ttest_rel([p["untracked"] for p in pairs], [p["tracked"] for p in pairs])
w_tr = stats.wilcoxon([p["untracked"] for p in pairs], [p["tracked"] for p in pairs])
out["tracking"] = {
    "pairs": pairs,
    "mean_tracked": f(np.mean([p["tracked"] for p in pairs]), 1),
    "mean_untracked": f(np.mean([p["untracked"] for p in pairs]), 1),
    "mean_diff": f(diffs.mean(), 2), "sem_diff": f(sem(diffs), 2),
    "n_positive": int((diffs > 0).sum()), "n": len(pairs),
    "t": f(t_tr.statistic, 2), "p_t": f(t_tr.pvalue, 4),
    "p_wilcoxon": f(w_tr.pvalue, 4),
    "cohen_dz": f(diffs.mean() / diffs.std(ddof=1), 2),
}

# ---- Exp2: preference vs MW
e2 = df[df.Experiment == "Exp2_Preference"]
corr_pref = r(e2.Pref_relative, e2.MW_Estimate)
# within-subject correlations
within = []
for s, g in e2.groupby("Subject"):
    c = r(g.Pref_relative, g.MW_Estimate)
    if c:
        within.append({"id": s, "r": f(c["r"], 3)})
within.sort(key=lambda x: x["r"])
# bins of pref_relative
bins = [-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3]
e2b = e2.dropna(subset=["Pref_relative"]).copy()
e2b["bin"] = pd.cut(e2b.Pref_relative, bins)
binned = [{"label": f"{iv.left:g}..{iv.right:g}", "mean": f(g.MW_Estimate.mean(), 3), "sem": f(sem(g.MW_Estimate), 3), "n": int(len(g))}
          for iv, g in e2b.groupby("bin", observed=True)]
out["preference"] = {
    "points": [{"x": f(a, 3), "y": f(b, 3), "s": s, "t": int(t), "h": h}
               for a, b, s, t, h in zip(e2.Pref_relative, e2.MW_Estimate, e2.Subject, e2.TrialNum, e2.Hero) if not np.isnan(a)],
    "r": f(corr_pref["r"], 3), "n": corr_pref["n"],
    "within": within, "within_mean_r": f(np.mean([w["r"] for w in within]), 3),
    "within_n_negative": int(sum(1 for w in within if w["r"] < 0)),
    "binned": binned,
    "pref_by_hero": {h: f(g.Preference.mean(), 1) for h, g in e2.groupby("Hero")},
}

# ---- Hero effect (per subject paired)
hero = []
for s, g in df.groupby("Subject"):
    hero.append({"id": s, "exp": g.Experiment.iloc[0],
                 "KAROLINA": f(g[g.Hero == "KAROLINA"].MW.mean(), 1),
                 "JANEK": f(g[g.Hero == "JANEK"].MW.mean(), 1)})
hd = np.array([h["KAROLINA"] - h["JANEK"] for h in hero])
t_h = stats.ttest_rel([h["KAROLINA"] for h in hero], [h["JANEK"] for h in hero])
out["hero"] = {"pairs": hero, "mean_diff": f(hd.mean(), 2), "sem_diff": f(sem(hd), 2),
               "t": f(t_h.statistic, 2), "p": f(t_h.pvalue, 4),
               "mean": {h: f(g.MW.mean(), 1) for h, g in df.groupby("Hero")}}

# ---- Time on task: first vs last 10 trials, per subject
tot = []
for s, g in df.groupby("Subject"):
    tot.append({"id": s, "first10": f(g[g.TrialNum <= 10].MW.mean(), 1), "last10": f(g[g.TrialNum > 30].MW.mean(), 1)})
td = np.array([x["last10"] - x["first10"] for x in tot])
t_t = stats.ttest_rel([x["last10"] for x in tot], [x["first10"] for x in tot])
slope = r(df.TrialNum, df.MW_Estimate)
out["time_on_task"] = {"pairs": tot, "mean_diff": f(td.mean(), 2), "sem_diff": f(sem(td), 2),
                       "t": f(t_t.statistic, 2), "p": f(t_t.pvalue, 4), "r_trial_mwz": f(slope["r"], 3),
                       "n_increase": int((td > 0).sum()), "n": len(tot)}

# ---- Recall features vs MW (all rows with a recall)
rec = df[df.has].copy()
out["recall_corr"] = {
    "words": r(rec.words, rec.MW_Estimate),
    "coupling": r(rec.coupling, rec.MW_Estimate),
    "self_rate": r(rec.self_rate, rec.MW_Estimate),
}
# MW bins (raw 0-99) -> mean features
mwbins = [-1, 10, 25, 50, 75, 100]
rec["mwbin"] = pd.cut(rec.MW, mwbins)
labels = ["0-10", "11-25", "26-50", "51-75", "76-99"]
feat_by_mw = []
for (iv, g), lab in zip(rec.groupby("mwbin", observed=False), labels):
    feat_by_mw.append({"label": lab, "n": int(len(g)),
                       "words": f(g.words.mean(), 1), "words_sem": f(sem(g.words), 2),
                       "coupling": f(g.coupling.mean(), 3), "coupling_sem": f(sem(g.coupling), 3),
                       "self_rate": f(g.self_rate.mean(), 4), "self_sem": f(sem(g.self_rate), 4),
                       "no_thoughts": f(g.no_discernable_thoughts.mean(), 3),
                       "attn": f((g.has_attention_indicator > 0).mean(), 3)})
out["feat_by_mw"] = feat_by_mw

# flags
out["flags"] = {
    "no_discernable": {"n": int((rec.no_discernable_thoughts == 1).sum()),
                       "mw_yes": f(rec[rec.no_discernable_thoughts == 1].MW.mean(), 1),
                       "mw_no": f(rec[rec.no_discernable_thoughts == 0].MW.mean(), 1)},
    "attention_indicator": {"n": int((rec.has_attention_indicator > 0).sum()),
                            "mw_yes": f(rec[rec.has_attention_indicator > 0].MW.mean(), 1),
                            "mw_no": f(rec[rec.has_attention_indicator == 0].MW.mean(), 1)},
    "recall_first": {k: f(g.MW.mean(), 1) for k, g in e2.dropna(subset=["recall_first"]).groupby("recall_first")},
}

# ---- MW histogram (raw)
hist, edges = np.histogram(df.MW, bins=20, range=(0, 100))
out["mw_hist"] = {"counts": [int(c) for c in hist], "edges": [float(e) for e in edges],
                  "by_exp": {e: [int(c) for c in np.histogram(g.MW, bins=20, range=(0, 100))[0]] for e, g in df.groupby("Experiment")}}

# ---- Words: high vs low MW (median split of MW_Estimate within recalls)
med = rec.MW_Estimate.median()
hi = rec[rec.MW_Estimate > med]
lo = rec[rec.MW_Estimate <= med]


def counts(g):
    c = Counter()
    for lems in g.LemmasL:
        c.update(content(lems))  # document frequency
    return c


ch, cl = counts(hi), counts(lo)
nh, nl = len(hi), len(lo)
vocab = {w for w, c in (ch + cl).items() if c >= 15}
words = []
for w in vocab:
    ph = (ch[w] + 0.5) / (nh + 1)
    pl = (cl[w] + 0.5) / (nl + 1)
    words.append({"w": w, "hi": ch[w], "lo": cl[w], "lor": f(np.log2(ph / pl), 3),
                  "rate_hi": f(ch[w] / nh, 3), "rate_lo": f(cl[w] / nl, 3)})
words.sort(key=lambda x: x["lor"])
out["words"] = {"n_hi": nh, "n_lo": nl, "median_split": f(med, 3),
                "low_mw": list(reversed(words[:20])) and words[:20], "high_mw": list(reversed(words[-20:]))}
# overall top lemmas
call = counts(rec)
out["words"]["top"] = [{"w": w, "n": c, "rate": f(c / len(rec), 3)} for w, c in call.most_common(30)]

# ---- Recalls (full list for the browser)
recs = []
for _, x in rec.iterrows():
    recs.append({
        "s": x.Subject, "e": "1" if x.Experiment == "Exp1_Cued" else "2", "t": int(x.TrialNum),
        "mw": int(x.MW), "z": f(x.MW_Estimate, 2),
        "tr": (x.Tracking[0] if isinstance(x.Tracking, str) else None),
        "pref": (None if pd.isna(x.Preference) else int(x.Preference)),
        "u": x.Utterance, "w": int(x.words), "c": f(x.coupling, 2), "sr": f(x.self_rate, 2),
        "nt": int(x.no_discernable_thoughts) if not pd.isna(x.no_discernable_thoughts) else None,
        "ai": int(x.has_attention_indicator) if not pd.isna(x.has_attention_indicator) else None,
    })
out["recalls"] = recs

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print("rows", len(df), "recalls", len(recs), "->", out_path)
print(json.dumps({k: out[k] for k in ["meta", "tracking", "time_on_task", "hero", "recall_corr", "flags"]}, ensure_ascii=False, indent=1, default=str)[:6000])
print("pref r", out["preference"]["r"], "within mean r", out["preference"]["within_mean_r"], "neg", out["preference"]["within_n_negative"], "/", len(within))
print("binned pref", out["preference"]["binned"])
print("feat_by_mw", out["feat_by_mw"])
print("HIGH", [(w["w"], w["lor"]) for w in out["words"]["high_mw"]])
print("LOW", [(w["w"], w["lor"]) for w in out["words"]["low_mw"]])
