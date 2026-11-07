"""Which seconds of the fragment came back in the recall, against the pupil; and trial pupil x MW x coupling.

usage: timing.py behavioral_cleaned.csv semantic.json embeddings.npz eye.json timing.json [words.json]
Story sentence onsets are MEASURED on the audio when words.json (align.py) is given;
without it they are ESTIMATED from the character share of each sentence in the text.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

csv_path, sem_path, npz_path, eye_path, out_path = sys.argv[1:6]
words_path = sys.argv[6] if len(sys.argv) > 6 else None
MODELS = ["pl", "mini"]
MW_BINS, MW_LABELS = [-1, 10, 25, 50, 75, 100], ["0-10", "11-25", "26-50", "51-75", "76-99"]
LAGS = list(range(-3, 9))   # seconds around a sentence onset in the aligned time course
NULL_Q = 0.95


def f(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


sem = json.load(open(sem_path, encoding="utf-8"))
E = np.load(npz_path)
eye = json.load(open(eye_path, encoding="utf-8"))
df = pd.read_csv(csv_path, usecols=["Subject", "TrialNum", "Fragment", "Length"])

# ---- trial length: own Length, else the same fragment's median (Exp1 has none; same audio), else 45 s
frag_len = df.groupby("Fragment").Length.median()
own_len = {(r.Subject, int(r.TrialNum)): r.Length for r in df.itertuples()}


def trial_len(s, t, frag):
    v = own_len.get((s, t))
    if v is None or not np.isfinite(v):
        v = frag_len.get(frag, np.nan)
    return (v if np.isfinite(v) else 45000.0) / 1000


# ---- sentence intervals as fractions 0..1 of the trial: measured onsets (words.json, ticket 8)
# over the recording's length when given, else the character share of the fragment text
FR = sem["fragments"]
edges = {}
for fr in FR:
    c = np.cumsum([0] + [len(s) for s in fr["ss"]])
    edges[fr["name"]] = c / c[-1]
MEASURED = words_path is not None
if MEASURED:
    for wf in json.load(open(words_path, encoding="utf-8"))["fragments"].values():
        first, last_end = {}, 0.0
        for x in wf["words"]:
            if "t0" in x:
                first.setdefault(x["s"], x["t0"])
                last_end = x["t1"]
            else:   # a punctuation-only "sentence" (KAROLINA_8: a lone quote) starts in the pause before it
                first.setdefault(x["s"], last_end)
        n = len(edges[wf["fragment"]]) - 1
        assert sorted(first) == list(range(n)), f"{wf['fragment']}: sentences in words.json do not match semantic.json"
        edges[wf["fragment"]] = np.array([0.0] + [first[s] / wf["dur"] for s in range(1, n)] + [1.0])

story_frag = E["story_frag"]
fidx = {fr["name"]: j for j, fr in enumerate(FR)}
hero_of = {j: fr["hero"] for j, fr in enumerate(FR)}
ridx = {s: i for i, s in enumerate(E["recall_sents"].tolist())}
R = sem["recalls"]

# ---- similarity of every story sentence to a recall: max over the recall's sentences
sims, thr, null_share = {}, {}, {}
for k in MODELS:
    Es, Er = E[f"{k}_story"].astype(np.float64), E[f"{k}_recall"].astype(np.float64)
    own, null = [], []
    for r in R:
        best = (Er[[ridx[s] for s in r["rs"]]] @ Es.T).max(0)          # [252]
        j = fidx[r["f"]]
        own.append(best[story_frag == j])
        same_hero_other = np.array([hero_of[x] == hero_of[j] and x != j for x in story_frag])
        null.append(best[same_hero_other])
    thr[k] = float(np.quantile(np.concatenate(null), NULL_Q))
    sims[k] = own
    null_share[k] = float(np.mean(np.concatenate(own) >= thr[k]))
    print(f"{k}: threshold {thr[k]:.3f} (q{NULL_Q} of other fragments of the same story); "
          f"own story sentences above it {null_share[k]:.1%}", flush=True)

# ---- pupil: z within subject over the seconds inside the fragment; detrended = minus the mean z at that second
P = eye["pupil_1hz"]
rows = []
for key, arr in P.items():
    s, t = key.split("|")
    t = int(t)
    frag = df.loc[(df.Subject == s) & (df.TrialNum == t), "Fragment"]
    if frag.empty:
        continue
    L = trial_len(s, t, frag.iloc[0])
    for sec, v in enumerate(arr):
        if v is not None and sec + 0.5 < L:
            rows.append((s, t, sec, v))
pu = pd.DataFrame(rows, columns=["s", "t", "sec", "v"])
pu["z"] = pu.groupby("s").v.transform(lambda x: (x - x.mean()) / x.std(ddof=1))
pu["dz"] = pu.z - pu.groupby("sec").z.transform("mean")
pz = {(s, t): g.set_index("sec")[["z", "dz"]] for (s, t), g in pu.groupby(["s", "t"])}
print(f"pupil: {len(pz)} trials, {len(pu)} seconds", flush=True)

out = {"meta": {"null_q": NULL_Q, "thr": {k: f(thr[k]) for k in MODELS},
                "own_above_thr": {k: f(null_share[k]) for k in MODELS}, "lags": LAGS,
                "onsets": "measured" if MEASURED else "estimated",
                "length_source": "Length (ms) of the trial; Exp1 has none, so the median Length of the same fragment"},
       "edges": {n: [f(x, 4) for x in e] for n, e in edges.items()},
       "per_recall": [], "summary": {}}

trial_rows = []   # one per recall x model with pupil
events = []       # aligned time course: one per story sentence onset
for i, r in enumerate(R):
    key = (r["s"], r["t"])
    L = trial_len(r["s"], r["t"], r["f"])
    e = edges[r["f"]] * L
    # one sentence ends where the next starts (the shared edge), the first at 0, the last at the trial length
    assert e[0] == 0 and np.isclose(e[-1], L) and np.all(np.diff(e) > 0), f"{r['f']}: intervals not contiguous"
    p = pz.get(key)
    out["per_recall"].append({"L": f(L, 2), "sim": {k: [f(v) for v in sims[k][i]] for k in MODELS},
                              "pz": None if p is None else [f(p.z.get(sec, np.nan), 2) for sec in range(int(np.ceil(L - 0.5)))]})
    if p is None:
        continue
    sent_of_sec = np.searchsorted(e, p.index.values + 0.5, side="right") - 1
    for k in MODELS:
        rec = sims[k][i] >= thr[k]
        m = rec[sent_of_sec]
        trial_rows.append({"model": k, "s": r["s"], "i": i, "mw": r["mw"], "n_rec_sent": int(rec.sum()), "n_sent": len(rec),
                           "rz": p.z.values[m].mean() if m.any() else np.nan, "nz": p.z.values[~m].mean() if (~m).any() else np.nan,
                           "rdz": p.dz.values[m].mean() if m.any() else np.nan, "ndz": p.dz.values[~m].mean() if (~m).any() else np.nan,
                           "tdz": p.dz.values.mean()})
        if not (m.any() and (~m).any()):
            continue   # same trials as the paired test, or the curve mixes in the between-trial difference
        for j in range(1, len(rec)):   # the first sentence starts with the trial: its onset is the pupil's onset response
            o = int(round(e[j]))
            vals = [p.dz.get(o + lag, np.nan) for lag in LAGS]
            events.append({"model": k, "s": r["s"], "rec": bool(rec[j]), "pos": j, "vals": vals})

tr = pd.DataFrame(trial_rows)
tr["bin"] = pd.cut(tr.mw, MW_BINS, labels=MW_LABELS)
ev = pd.DataFrame(events)


def paired(g, a, b):
    both = g.dropna(subset=[a, b])
    subj = both.groupby("s")[[a, b]].mean()
    if len(subj) < 3:
        return {"rec": None, "not": None, "diff": None, "sem": None, "t": None, "p": None, "n_subj": len(subj), "n_trials": len(both)}
    d = subj[a] - subj[b]
    t = stats.ttest_rel(subj[a], subj[b])
    return {"rec": f(subj[a].mean()), "not": f(subj[b].mean()), "diff": f(d.mean()), "sem": f(d.sem()),
            "t": f(t.statistic, 2), "p": f(t.pvalue, 4), "n_subj": len(subj), "n_trials": len(both)}


for k in MODELS:
    g = tr[tr.model == k]
    e_ = ev[ev.model == k]
    aligned = {}
    for flag, name in [(True, "rec"), (False, "not")]:
        sub = e_[e_.rec == flag]
        M = pd.DataFrame(sub.vals.tolist(), columns=LAGS)
        M["s"] = sub.s.values
        per_s = M.groupby("s").mean()
        aligned[name] = {"mean": [f(x) for x in per_s.mean()], "sem": [f(x) for x in per_s.sem()],
                         "n_events": len(sub), "n_subj": len(per_s)}
    flags = [(j, n, v >= thr[k]) for sm in sims[k] for n in [len(sm)] for j, v in enumerate(sm)]
    last = np.mean([x for j, n, x in flags if j == n - 1])
    rest = np.mean([x for j, n, x in flags if j < n - 1])
    ga = g.assign(any_=np.where(g.n_rec_sent > 0, g.tdz, np.nan), none_=np.where(g.n_rec_sent == 0, g.tdz, np.nan))
    between = ga.groupby("s")[["any_", "none_"]].mean().dropna()
    bt = stats.ttest_rel(between.any_, between.none_)
    out["summary"][k] = {
        "z": paired(g, "rz", "nz"), "dz": paired(g, "rdz", "ndz"),
        "between": {"any": f(between.any_.mean()), "none": f(between.none_.mean()), "t": f(bt.statistic, 2),
                    "p": f(bt.pvalue, 4), "n_subj": len(between)},
        "by_mw": [{"label": lab, "n": int(len(gb)), **paired(gb, "rdz", "ndz")} for lab, gb in g.groupby("bin", observed=False)],
        "aligned": aligned,
        "share_recalled": f(g.n_rec_sent.sum() / g.n_sent.sum()),
        "trials_none": int((g.n_rec_sent == 0).sum()), "trials_all": int((g.n_rec_sent == g.n_sent).sum()), "n_trials": len(g),
        "last_vs_rest": {"last": f(last), "rest": f(rest)},   # all recalls, with or without pupil
    }
    s = out["summary"][k]
    print(f"{k}: {s['share_recalled']:.1%} of story sentences recalled; trials with none {s['trials_none']}, "
          f"all {s['trials_all']} of {s['n_trials']}; last sentence recalled {last:.1%} vs others {rest:.1%}", flush=True)
    for v in ["z", "dz"]:
        x = s[v]
        print(f"{k} paired {v}: recalled {x['rec']} vs not {x['not']}, diff {x['diff']} ± {x['sem']}, "
              f"t = {x['t']}, p = {x['p']}, {x['n_subj']} subjects, {x['n_trials']} trials", flush=True)
    b = s["between"]
    print(f"{k} between trials (detrended): any recalled {b['any']} vs none {b['none']}, t = {b['t']}, p = {b['p']}, "
          f"{b['n_subj']} subjects", flush=True)

# ---- triad: trial pupil x coupling x MW, all z within subject (a pooled between-subject r would mix in who talks more)
TF = eye["trial_feats"]
TRIAD_VARS = {"pupil": "pupil", "mw": "MW", "cmax": "coupling"}
PAIRS = [("pupil", "mw"), ("pupil", "cmax"), ("mw", "cmax")]


def triad(k):
    rows = []
    for r in R:
        tf = TF.get(f"{r['s']}|{r['t']}")
        if tf and tf.get("pz") is not None and tf.get("miss", 1) <= 0.5 and r["mwz"] is not None:
            rows.append({"s": r["s"], "pupil": tf["pz"], "mw": r["mwz"], "cmax": r["m"][k]["cmax"]})
    d = pd.DataFrame(rows)
    d = d[d.groupby("s").s.transform("size") >= 8]
    zs = d.groupby("s")[list(TRIAD_VARS)].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    zs["s"] = d.s
    zs = zs.dropna()   # a subject with a constant variable has no within-subject variance
    res = {"n_trials": len(zs), "n_subj": int(zs.s.nunique()), "pairs": []}
    for a, b in PAIRS:
        pr = stats.pearsonr(zs[a], zs[b])
        per = np.array([np.corrcoef(g[a], g[b])[0, 1] for _, g in zs.groupby("s")])
        per = per[np.isfinite(per)]
        t1 = stats.ttest_1samp(per, 0)
        res["pairs"].append({"a": a, "b": b, "r": f(pr[0]), "p": f(pr[1], 4), "mean_r": f(per.mean()),
                             "t": f(t1.statistic, 2), "p_subj": f(t1.pvalue, 4), "n_subj": len(per)})
    X = np.column_stack([np.ones(len(zs)), zs.pupil, zs.mw])
    y = zs.cmax.values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    se = np.sqrt((y - X @ b) @ (y - X @ b) / dof * np.diag(np.linalg.inv(X.T @ X)))
    res["reg"] = [{"x": n, "b": f(b[j]), "se": f(se[j], 4), "t": f(b[j] / se[j], 2), "p": f(2 * stats.t.sf(abs(b[j] / se[j]), dof), 4)}
                  for j, n in [(1, "pupil"), (2, "mw")]]
    return res


out["triad"] = {"vars": TRIAD_VARS, **{k: triad(k) for k in MODELS}}
for k in MODELS:
    tr_ = out["triad"][k]
    print(f"{k} triad ({tr_['n_trials']} trials, {tr_['n_subj']} subjects): " + "; ".join(
        f"r({p['a']},{p['b']}) = {p['r']} p = {p['p']}, per-subject {p['mean_r']} p = {p['p_subj']}" for p in tr_["pairs"]), flush=True)
    print(f"{k} triad regression coupling ~ pupil + MW: " + "; ".join(
        f"b_{c['x']} = {c['b']} ± {c['se']}, t = {c['t']}, p = {c['p']}" for c in tr_["reg"]), flush=True)

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("wrote", out_path)
