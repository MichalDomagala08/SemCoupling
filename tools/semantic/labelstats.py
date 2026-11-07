"""Statistics for the thought categories (ticket 3) and Haiku's distraction rating (ticket 11).

usage: labelstats.py semantic.json labels.json eye.json out.json [mwrate.json]
Every comparison that could mix people is within subject: z inside the subject, per-subject means or r,
then a test across subjects.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sem_p, lab_p, eye_p, out_p = sys.argv[1:5]
rate_p = sys.argv[5] if len(sys.argv) > 5 else None
# same order as label.py::CATS (not imported: label.py parses its own argv at import)
CATS = ["story", "association", "task", "environment", "self_past", "self_present", "self_future", "blank"]
MW_BINS, MW_LABELS = [-1, 10, 25, 50, 75, 100], ["0-10", "11-25", "26-50", "51-75", "76-99"]
EYE = {"pz": "źrenica", "gsd": "rozrzut spojrzenia", "br": "mrugnięcia"}
MIN_TRIALS = 8


def f(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def zin(d, cols):
    if d.empty:   # groupby-transform on no rows raises instead of returning nothing
        return d[cols]
    return d.groupby("s")[cols].transform(lambda x: (x - x.mean()) / x.std(ddof=1))


sem = json.load(open(sem_p, encoding="utf-8"))
labels = json.load(open(lab_p, encoding="utf-8"))
TF = json.load(open(eye_p, encoding="utf-8"))["trial_feats"]
R = sem["recalls"]
rows = []
for i, r in enumerate(R):
    k = f"{r['s']}|{r['t']}"
    lab, tf = labels.get(k, {}), TF.get(k) or {}
    rows.append({"i": i, "s": r["s"], "mw": r["mw"], "cmax_pl": r["m"]["pl"]["cmax"], "cmax_mini": r["m"]["mini"]["cmax"],
                 "cg": lab.get("category"), "also": lab.get("also", []), "conf": lab.get("confidence"), "topic": lab.get("topic"),
                 **{e: tf.get(e) for e in EYE}, "miss": tf.get("miss")})
d = pd.DataFrame(rows)
bad = set(d.cg.dropna()) - set(CATS)
assert not bad, f"categories outside CATS: {bad}"
missing = int(d.cg.isna().sum())
print(f"{len(d)} recalls, {len(d) - missing} labelled, {missing} without a label", flush=True)
d["bin"] = pd.cut(d.mw, MW_BINS, labels=MW_LABELS).astype(str)
L = d.dropna(subset=["cg"])

out = {"cats": CATS, "per_recall": [[r.cg, r.also, f(r.conf, 2), r.topic] if r.cg else None for r in d.itertuples()],
       "n_missing": missing}

# share per MW bin (primary category) and how often each category appears at all (primary or also)
out["by_mw"] = [{"label": b, "n": int((L.bin == b).sum()), "counts": {c: int(((L.bin == b) & (L.cg == c)).sum()) for c in CATS}}
                for b in MW_LABELS]
out["totals"] = {c: {"primary": int((L.cg == c).sum()), "any": int(L.apply(lambda r: r.cg == c or c in r.also, axis=1).sum())}
                 for c in CATS}

# coupling and self-reported MW per category: descriptive over recalls
out["by_cat"] = []
for c in CATS:
    g = L[L.cg == c]
    sem_ = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    out["by_cat"].append({"cat": c, "n": len(g), "n_subj": int(g.s.nunique()),
                          "cmax_pl": f(g.cmax_pl.mean()), "cmax_pl_sem": f(sem_(g.cmax_pl), 4),
                          "cmax_mini": f(g.cmax_mini.mean()), "cmax_mini_sem": f(sem_(g.cmax_mini), 4),
                          "mw": f(g.mw.mean(), 1), "mw_sem": f(sem_(g.mw), 2)})

# eye features per category, within subject, each category against story
E = L[(L.miss <= 0.5) & L.pz.notna()].copy()
E = E[E.groupby("s").s.transform("size") >= MIN_TRIALS]
E[list(EYE)] = zin(E, list(EYE))
E = E.dropna(subset=list(EYE))
out["eye"] = {"n_trials": len(E), "n_subj": int(E.s.nunique()), "features": EYE, "rows": []}
for c in CATS:
    row = {"cat": c}
    for e in EYE:
        per = E[E.cg == c].groupby("s")[e].mean()
        st = E[E.cg == "story"].groupby("s")[e].mean()
        both = per.index.intersection(st.index)
        diff = per[both] - st[both]
        t = stats.ttest_1samp(diff, 0) if c != "story" and len(diff) > 2 else None
        row[e] = {"mean": f(per.mean()), "sem": f(per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan, 4),
                  "n_subj": len(per), "vs_story": f(diff.mean()) if t else None,
                  "t": f(t.statistic, 2) if t else None, "p": f(t.pvalue, 4) if t else None, "n_pair": len(diff) if t else None}
    out["eye"]["rows"].append(row)

for b in out["by_mw"]:
    top = sorted(CATS, key=lambda c: -b["counts"][c])[:3]
    print(f"MW {b['label']:>6} n={b['n']:>4}: " + ", ".join(f"{c} {b['counts'][c] / max(1, b['n']):.0%}" for c in top), flush=True)
for r in out["by_cat"]:
    print(f"{r['cat']:<13} n={r['n']:>4} subj={r['n_subj']:>2} cmax_pl {r['cmax_pl']} mini {r['cmax_mini']} MW {r['mw']}", flush=True)
for r in out["eye"]["rows"]:
    print(f"eye {r['cat']:<13} " + "; ".join(f"{e} {r[e]['mean']} (vs story {r[e]['vs_story']}, p {r[e]['p']})" for e in EYE), flush=True)


# ---- ticket 11: Haiku's 0-100 rating against the probe
def within_r(z, a, b):
    pr = stats.pearsonr(z[a], z[b])
    per = np.array([np.corrcoef(g[a], g[b])[0, 1] for _, g in z.groupby("s")])
    per = per[np.isfinite(per)]
    t = stats.ttest_1samp(per, 0)
    return {"r": f(pr[0]), "p": f(pr[1], 4), "mean_r": f(per.mean()), "t": f(t.statistic, 2), "p_subj": f(t.pvalue, 4), "n_subj": len(per)}


def partial(x, y, c):
    rxy, rxc, ryc = np.corrcoef(x, y)[0, 1], np.corrcoef(x, c)[0, 1], np.corrcoef(y, c)[0, 1]
    return (rxy - rxc * ryc) / np.sqrt((1 - rxc ** 2) * (1 - ryc ** 2))


if rate_p:
    rate = json.load(open(rate_p, encoding="utf-8"))
    d["model"] = [rate.get(f"{r['s']}|{r['t']}", {}).get("mw") for r in R]
    out["rate"] = {"per_recall": [None if pd.isna(m) else int(m) for m in d.model],
                   "reason": [rate.get(f"{r['s']}|{r['t']}", {}).get("reason") for r in R]}
    Q = d.dropna(subset=["model"]).copy()
    Q["model"] = Q.model.astype(float)
    res = {"n": len(Q), "pooled": {k: f(stats.pearsonr(Q.mw, Q[k])[0]) for k in ["model", "cmax_pl", "cmax_mini"]}}
    # the pooled scatter mixes who (subject's habitual scale use) with when; these numbers separate the two
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    edges, labs = [-1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 100], [f"{a}-{a + 9}" for a in range(0, 100, 10)]
    Q["b10"] = pd.cut(Q.mw, edges, labels=labs)
    res["bins"] = [{"label": l, "n": len(g), "x": f(g.mw.mean(), 1), "model": f(g.model.mean(), 1), "ci": f(ci(g.model), 1)}
                   for l, g in Q.groupby("b10", observed=False) if len(g)]
    share = lambda k: Q.groupby("s")[k].transform("mean").var() / Q[k].var()
    res["between_share"] = {"self": f(share("mw"), 2), "model": f(share("model"), 2)}
    res["top_values"] = [int(v) for v in Q.model.value_counts().index[:5]]
    Z = Q[Q.groupby("s").s.transform("size") >= MIN_TRIALS].copy()
    Z["model_raw"] = Z.model
    cols = ["mw", "model", "cmax_pl", "cmax_mini"]
    Z[cols] = zin(Z, cols)
    Z = Z.dropna(subset=cols)
    Z["q5"] = pd.qcut(Z.mw, 5, labels=False)
    res["quint"] = [{"label": f"{q + 1}", "n": len(g), "self_z": f(g.mw.mean(), 2), "model": f(g.model_raw.mean(), 1), "ci": f(ci(g.model_raw), 1)}
                    for q, g in Z.groupby("q5")]
    res.update({"n_within": len(Z), "n_subj": int(Z.s.nunique()),
                "within": {k: within_r(Z, "mw", k) for k in ["model", "cmax_pl", "cmax_mini"]},
                "model_vs_cmax": within_r(Z, "model", "cmax_pl")})
    for k in ["pl", "mini"]:
        pr = partial(Z.mw, Z.model, Z[f"cmax_{k}"])
        n = len(Z)
        tt = pr * np.sqrt((n - 3) / (1 - pr ** 2))
        per = np.array([partial(g.mw, g.model, g[f"cmax_{k}"]) for _, g in Z.groupby("s")])
        per = per[np.isfinite(per)]
        t1 = stats.ttest_1samp(per, 0)
        res[f"partial_{k}"] = {"r": f(pr), "p": f(2 * stats.t.sf(abs(tt), n - 3), 4), "mean_r": f(per.mean()),
                               "t": f(t1.statistic, 2), "p_subj": f(t1.pvalue, 4), "n_subj": len(per)}
    res["by_cat"] = [{"cat": c, "n": int((Q.cg == c).sum()), "self": f(Q[Q.cg == c].mw.mean(), 1),
                      "model": f(Q[Q.cg == c].model.mean(), 1)} for c in CATS]
    out["rate"]["stats"] = res
    print(f"rate: {res['n']} rated; pooled r(self, model) {res['pooled']['model']}, r(self, cmax pl) {res['pooled']['cmax_pl']}", flush=True)
    for k, v in res["within"].items():
        print(f"rate within r(self, {k}) = {v['r']} p = {v['p']}; per-subject mean r {v['mean_r']}, t = {v['t']}, p = {v['p_subj']}, {v['n_subj']} subjects", flush=True)
    for k in ["pl", "mini"]:
        v = res[f"partial_{k}"]
        print(f"rate partial r(self, model | cmax {k}) = {v['r']} p = {v['p']}; per-subject {v['mean_r']}, t = {v['t']}, p = {v['p_subj']}", flush=True)
    print("rate bins (self MW -> mean Haiku): " + ", ".join(f"{b['label']} {b['model']}" for b in res["bins"]), flush=True)
    print("rate within-subject fifths -> mean Haiku: " + ", ".join(str(q["model"]) for q in res["quint"]) +
          f"; between-subject share of variance self {res['between_share']['self']}, Haiku {res['between_share']['model']}", flush=True)
    for c in res["by_cat"]:
        print(f"rate {c['cat']:<13} n={c['n']:>4} self {c['self']} model {c['model']}", flush=True)

json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("wrote", out_p)
