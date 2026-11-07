"""Pupil around each story word (measured onsets, ticket 8), high vs low MW within subject.

usage: wordpupil.py gaze_preproc.csv.gz behavioral_cleaned.csv words.json polish.stopwords.txt wordpupil.json

Pupil: left eye, right as fallback, 100 Hz, z within subject over its kept trials
(trials with > 50 % missing pupil dropped, as in eye.py). Epoch: -0.5..+2 s around a
word onset minus the mean of -0.5..0 s. The gaze trial starts with its recording:
trial 1 spans 42.57 s of gaze against a 42.55 s recording.
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

gaze_path, behav_path, words_path, stop_path, out_path = sys.argv[1:6]
HZ, PRE, POST = 100, 50, 200            # samples: 0.5 s before, 2 s after
SKIP_S = 3.0                            # words in the first 3 s ride on the pupil's response to the trial start
LOW_SC = 0.5                            # aligner score below this: word left out (ticket 8)
MIN_TRIALS = 5                          # per subject and MW half
WINDOWS = [(0, 0.5), (0.5, 1.0), (1.0, 2.0)]
CATS = ["all", "content", "function", "sent_first", "random"]
rng = np.random.default_rng(7)
t_start = time.time()

stop = {w.strip() for w in open(stop_path, encoding="utf-8") if w.strip()}
W = json.load(open(words_path, encoding="utf-8"))["fragments"]
behav = pd.read_csv(behav_path, usecols=["Subject", "TrialNum", "Fragment", "MW", "MW_Estimate"])
behav["hi"] = behav.MW_Estimate > behav.groupby("Subject").MW_Estimate.transform("median")
half = {(r.Subject, int(r.TrialNum)): bool(r.hi) for r in behav.itertuples()}


def word_cat(w):
    k = re.sub(r"[^\w]", "", w.lower())
    return "function" if k in stop else "content"


onsets = {}   # trial -> list of (sample, category list)
for t, f in W.items():
    ev, seen_s = [], set()
    for x in f["words"]:
        if "t0" not in x or x["sc"] < LOW_SC or x["t0"] < SKIP_S:
            seen_s.add(x["s"])
            continue
        cats = ["all", word_cat(x["w"])]
        if x["s"] not in seen_s:
            cats.append("sent_first")
        seen_s.add(x["s"])
        ev.append((int(round(x["t0"] * HZ)), cats))
    onsets[int(t)] = (ev, f["dur"])

# ---- pupil per trial at 100 Hz
chunks = {}
cols = ["Subject", "TrialNum", "TimePoint", "LeftPupil", "RightPupil"]
for i, ch in enumerate(pd.read_csv(gaze_path, usecols=cols, chunksize=2_000_000, dtype={"Subject": "category", "TrialNum": "int16"})):
    ch["p"] = ch.LeftPupil.where(ch.LeftPupil.notna(), ch.RightPupil)
    for (s, t), g in ch.groupby(["Subject", "TrialNum"], observed=True):
        chunks.setdefault((str(s), int(t)), []).append(g[["TimePoint", "p"]].to_numpy(np.float64))
    print(f"gaze chunk {i}, {time.time() - t_start:.0f} s", flush=True)

trials = {}
for key, parts in chunks.items():
    a = np.concatenate(parts)
    idx = np.round((a[:, 0] - a[:, 0].min()) / (1000 / HZ)).astype(int)
    p = np.full(idx.max() + 1, np.nan)
    p[idx] = a[:, 1]
    if np.isnan(p).mean() <= 0.5 and key in half and key[1] in onsets:
        trials[key] = p
del chunks
subj_stats = {}
for (s, t), p in trials.items():
    subj_stats.setdefault(s, []).append(p[~np.isnan(p)])
subj_stats = {s: (np.concatenate(v).mean(), np.concatenate(v).std()) for s, v in subj_stats.items()}
print(f"{len(trials)} trials with pupil, {len(subj_stats)} subjects, {time.time() - t_start:.0f} s", flush=True)

# ---- epochs, accumulated as nan-aware sums per subject x half x category
L = PRE + POST + 1
acc = {}
n_ep = {c: 0 for c in CATS}
for (s, t), p in trials.items():
    mu, sd = subj_stats[s]
    z = (p - mu) / sd
    ev, dur = onsets[t]
    n_rand = sum(1 for _, c in ev)
    lo_s, hi_s = int(SKIP_S * HZ), min(len(z), int(dur * HZ)) - POST - 1
    rand = [(int(c), ["random"]) for c in rng.integers(lo_s, max(lo_s + 1, hi_s), n_rand)]
    for c0, cats in ev + rand:
        if c0 - PRE < 0 or c0 + POST >= len(z):
            continue
        seg = z[c0 - PRE:c0 + POST + 1]
        base = seg[:PRE]
        if np.isnan(base).mean() > 0.5 or np.isnan(seg[PRE:]).mean() > 0.5:
            continue
        seg = seg - np.nanmean(base)
        ok = ~np.isnan(seg)
        for c in cats:
            k = (s, half[(s, t)], c)
            if k not in acc:
                acc[k] = [np.zeros(L), np.zeros(L), set()]
            acc[k][0][ok] += seg[ok]
            acc[k][1][ok] += 1
            acc[k][2].add(t)
            n_ep[c] += 1
print("epochs", n_ep, f"{time.time() - t_start:.0f} s", flush=True)

tax = (np.arange(L) - PRE) / HZ
win_idx = [(tax >= a) & (tax < b) for a, b in WINDOWS]


def f(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


out = {"meta": {"hz": HZ, "pre_s": PRE / HZ, "post_s": POST / HZ, "skip_s": SKIP_S, "low_sc": LOW_SC,
                "windows": WINDOWS, "mw": "MW_Estimate, median split within subject",
                "n_trials": len(trials), "n_epochs": n_ep, "elapsed_s": round(time.time() - t_start),
                # how big one within-subject SD is in the pupil's own units, for reading the z axis
                "sd_pct_median": f(100 * np.median([sd / mu for mu, sd in subj_stats.values()]), 1),
                "raw_mean_range": [f(min(mu for mu, _ in subj_stats.values()), 0), f(max(mu for mu, _ in subj_stats.values()), 0)]},
       "t": [f(x, 2) for x in tax[::2]], "cats": {}}
for c in CATS:
    subs = sorted({s for (s, h, cc) in acc if cc == c and (s, True, c) in acc and (s, False, c) in acc
                   and len(acc[(s, True, c)][2]) >= MIN_TRIALS and len(acc[(s, False, c)][2]) >= MIN_TRIALS})
    cur = {h: np.array([acc[(s, h, c)][0] / np.maximum(acc[(s, h, c)][1], 1) for s in subs]) for h in (True, False)}
    d = cur[True] - cur[False]
    sem = lambda m: m.std(0, ddof=1) / np.sqrt(len(m))
    res = {"n_subjects": len(subs),
           "hi": {"mean": [f(v) for v in cur[True].mean(0)[::2]], "sem": [f(v) for v in sem(cur[True])[::2]]},
           "lo": {"mean": [f(v) for v in cur[False].mean(0)[::2]], "sem": [f(v) for v in sem(cur[False])[::2]]},
           "windows": [], "per_subject": []}
    for (a, b), wi in zip(WINDOWS, win_idx):
        wh, wl = cur[True][:, wi].mean(1), cur[False][:, wi].mean(1)
        tt = stats.ttest_rel(wh, wl)
        res["windows"].append({"from": a, "to": b, "hi": f(wh.mean(), 4), "lo": f(wl.mean(), 4), "diff": f((wh - wl).mean(), 4),
                               "sem": f((wh - wl).std(ddof=1) / np.sqrt(len(wh)), 4), "t": f(tt.statistic, 2), "p": f(tt.pvalue, 4),
                               "dz": f((wh - wl).mean() / (wh - wl).std(ddof=1), 2), "n_pos": int((wh > wl).sum())})
        print(f"{c:10s} {a:.1f}-{b:.1f} s  hi {wh.mean():+.3f}  lo {wl.mean():+.3f}  diff {(wh - wl).mean():+.3f}  "
              f"t({len(wh) - 1}) = {tt.statistic:+.2f}  p = {tt.pvalue:.4f}", flush=True)
    for j, s in enumerate(subs):
        res["per_subject"].append({"id": s, "n_hi": len(acc[(s, True, c)][2]), "n_lo": len(acc[(s, False, c)][2]),
                                   "w": [f(cur[True][j, wi].mean() - cur[False][j, wi].mean()) for wi in win_idx]})
    out["cats"][c] = res

# word response beyond arbitrary moments: (word - random) per subject, high vs low
diffs = {}
for h in (True, False):
    subs = [x["id"] for x in out["cats"]["all"]["per_subject"] if x["id"] in {y["id"] for y in out["cats"]["random"]["per_subject"]}]
    diffs[h] = np.array([acc[(s, h, "all")][0] / np.maximum(acc[(s, h, "all")][1], 1) - acc[(s, h, "random")][0] / np.maximum(acc[(s, h, "random")][1], 1) for s in subs])
out["word_minus_random"] = []
for (a, b), wi in zip(WINDOWS, win_idx):
    row = {"from": a, "to": b}
    for h, nm in ((True, "hi"), (False, "lo")):
        v = diffs[h][:, wi].mean(1)
        tt = stats.ttest_1samp(v, 0)
        row[nm] = {"mean": f(v.mean(), 4), "sem": f(v.std(ddof=1) / np.sqrt(len(v)), 4), "t": f(tt.statistic, 2), "p": f(tt.pvalue, 4)}
    v = diffs[True][:, wi].mean(1) - diffs[False][:, wi].mean(1)
    tt = stats.ttest_1samp(v, 0)
    row["hi_minus_lo"] = {"mean": f(v.mean(), 4), "sem": f(v.std(ddof=1) / np.sqrt(len(v)), 4), "t": f(tt.statistic, 2), "p": f(tt.pvalue, 4)}
    out["word_minus_random"].append(row)
    print(f"word-random {a:.1f}-{b:.1f} s  hi {row['hi']['mean']:+.3f} (p {row['hi']['p']})  lo {row['lo']['mean']:+.3f} (p {row['lo']['p']})  "
          f"hi-lo {row['hi_minus_lo']['mean']:+.3f} t = {row['hi_minus_lo']['t']} p = {row['hi_minus_lo']['p']}", flush=True)
out["meta"]["n_subjects_wr"] = len(diffs[True])

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"{out_path}, {len(trials)} trials, {out['cats']['all']['n_subjects']} subjects, {time.time() - t_start:.0f} s")
