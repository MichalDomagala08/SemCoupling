"""Semantic coupling of every recall with its story fragment (sBERT, two models).

usage: embed.py behavioral_cleaned.csv semantic.json
Writes semantic.json (per-recall measures + summaries for the view) and
embeddings.npz beside it (unit sentence vectors, reused by timing.py).
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

MODELS = [
    ("pl", "sdadas/st-polish-paraphrase-from-mpnet"),
    ("mini", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
]
MW_BINS, MW_LABELS = [-1, 10, 25, 50, 75, 100], ["0-10", "11-25", "26-50", "51-75", "76-99"]

csv_path, out_path = sys.argv[1:3]


def as_list(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return [s.strip() for s in ast.literal_eval(x) if str(s).strip()]
        except (ValueError, SyntaxError):
            return []
    return []


def f(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


df = pd.read_csv(csv_path)
df["rs"] = df["Sentences"].apply(as_list)
df["ss"] = df["StorySentences"].apply(as_list)
n_rows = len(df)
rec = df[df.rs.str.len() > 0].reset_index(drop=True)   # ASSUMPTION: rows without sentences get no measures

frag = df.drop_duplicates("Fragment").copy()
frag["k"] = frag.Fragment.str.split("_").str[1].astype(int)
frag = frag.sort_values(["Hero", "k"]).reset_index(drop=True)
FR = list(frag.Fragment)
fidx = {n: i for i, n in enumerate(FR)}
# previous fragment of the same story (KAROLINA_3 -> KAROLINA_2); the first one has none
prev_of = {r.Fragment: (f"{r.Hero}_{r.k - 1}" if r.k > 1 else None) for r in frag.itertuples()}

story_sents = [s for ss in frag.ss for s in ss]
story_frag = np.array([fidx[n] for n, ss in zip(frag.Fragment, frag.ss) for _ in ss])
recall_sents = sorted({s for rs in rec.rs for s in rs})
ridx = {s: i for i, s in enumerate(recall_sents)}
verbatim = len(set(recall_sents) & set(story_sents))


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def coupling_max(U, F):
    # bertSentenceCosineSim(aggregation='max', mode=0) without its identical-token exclusion:
    # for sentences a verbatim retelling is the strongest coupling, not a trivial match
    S = U @ F.T
    return float(S.max(1).mean()), S.argmax(1), S.max(1)


def measures(U, E_story, centroids):
    """U: unit vectors of one recall's sentences. Returns cmax per fragment and centroid sims."""
    S = U @ E_story.T                                     # [u, all story sentences]
    cmax = np.array([S[:, story_frag == j].max(1).mean() for j in range(len(FR))])
    cent = centroids @ unit(U.mean(0))
    return S, cmax, cent


out = {"meta": {"n_rows": n_rows, "n_recalls": len(rec), "n_no_sentences": n_rows - len(rec),
                "n_story_sentences": len(story_sents), "n_recall_sentences": len(recall_sents),
                "n_fragments": len(FR), "verbatim_sentences": verbatim, "models": []},
       "fragments": [{"name": n, "hero": h, "k": int(k), "prev": prev_of[n], "ss": ss}
                     for n, h, k, ss in zip(frag.Fragment, frag.Hero, frag.k, frag.ss)],
       "recalls": [{"s": r.Subject, "e": 1 if r.Experiment.startswith("Exp1") else 2, "t": int(r.TrialNum),
                    "f": r.Fragment, "mw": f(r.MW, 0), "mwz": f(r.MW_Estimate, 3),
                    "tr": None if pd.isna(r.Tracking) else str(r.Tracking)[0], "rs": r.rs, "m": {}}
                   for r in rec.itertuples()],
       "summary": {}}
npz = {"story_frag": story_frag}

for key, name in MODELS:
    model = SentenceTransformer(name)
    E_story = unit(model.encode(story_sents, batch_size=64, convert_to_numpy=True).astype(np.float64))
    E_rec = unit(model.encode(recall_sents, batch_size=64, convert_to_numpy=True).astype(np.float64))
    npz[f"{key}_story"], npz[f"{key}_recall"] = E_story.astype(np.float32), E_rec.astype(np.float32)
    centroids = unit(np.vstack([E_story[story_frag == j].mean(0) for j in range(len(FR))]))

    # sanity: a "recall" that is a story sentence of its own fragment must couple at ~1
    j0 = fidx["KAROLINA_5"]
    probe = E_story[np.flatnonzero(story_frag == j0)[:1]]
    _, cm, _ = measures(probe, E_story, centroids)
    assert cm[j0] >= 0.95, f"{key}: verbatim story sentence couples at {cm[j0]:.3f} (model or column mix-up)"

    offsets = np.concatenate([[0], np.cumsum([len(ss) for ss in frag.ss])])
    other = []
    for r, o in zip(rec.itertuples(), out["recalls"]):
        U = E_rec[[ridx[s] for s in r.rs]]
        S, cmax, cent = measures(U, E_story, centroids)
        j = fidx[r.Fragment]
        own = S[:, offsets[j]:offsets[j + 1]]
        p = prev_of[r.Fragment]
        o["m"][key] = {
            "cmax": f(cmax[j]), "cent": f(cent[j]),
            "spec": int((cent > cent[j]).sum()) + 1,          # 1 = own fragment is the closest of all 40
            "prev": f(cmax[fidx[p]]) if p else None,
            "best": [[int(b), f(v, 3)] for b, v in zip(own.argmax(1), own.max(1))],  # per recall sentence
        }
        other.append(np.delete(cmax, j).mean())
    out["meta"]["models"].append({"key": key, "name": name, "dim": int(E_story.shape[1])})

    # ---- summaries for the view
    m = pd.DataFrame([{**o["m"][key], "s": o["s"], "e": o["e"], "mw": o["mw"], "mwz": o["mwz"]} for o in out["recalls"]])
    m["other"] = other
    m["bin"] = pd.cut(m.mw, MW_BINS, labels=MW_LABELS)
    by_mw = [{"label": lab, "n": len(g), "cmax": f(g.cmax.mean()), "cmax_sem": f(g.cmax.sem()),
              "cent": f(g.cent.mean()), "cent_sem": f(g.cent.sem()), "spec1": f((g.spec == 1).mean(), 3)}
             for lab, g in m.groupby("bin", observed=False)]
    rr = stats.pearsonr(m.cmax, m.mwz)
    m["cmax_wz"] = m.groupby("s").cmax.transform(lambda x: (x - x.mean()) / x.std())
    ws = [stats.pearsonr(g.cmax, g.mwz)[0] for _, g in m.groupby("s") if len(g) >= 5 and g.mwz.std() > 0]
    wt = stats.ttest_1samp(ws, 0)
    pv = m.dropna(subset=["prev"])
    pt = stats.ttest_rel(pv.cmax, pv.prev)
    edges = np.linspace(0, 1, 41)
    out["summary"][key] = {
        "by_mw": by_mw,
        "r_mwz": {"r": f(rr[0], 3), "p": f(rr[1], 5), "n": len(m)},
        "within": {"mean_r": f(np.mean(ws), 3), "t": f(wt.statistic, 2), "p": f(wt.pvalue, 5), "n": len(ws)},
        "by_exp": {str(e): {"hist": np.histogram(g.cmax, edges)[0].tolist(), "mean": f(g.cmax.mean()),
                            "sd": f(g.cmax.std()), "n": len(g)} for e, g in m.groupby("e")},
        "hist_edges": [f(x, 3) for x in edges],
        "spec_hist": np.bincount(m.spec, minlength=41)[1:].tolist(),
        "spec1": f((m.spec == 1).mean(), 3),
        "own_vs_other": {"own": f(m.cmax.mean()), "other": f(m.other.mean()),
                         "share_own_higher": f((m.cmax > m.other).mean(), 3)},
        "own_vs_prev": {"own": f(pv.cmax.mean()), "prev": f(pv.prev.mean()), "t": f(pt.statistic, 2),
                        "p": f(pt.pvalue, 5), "n": len(pv)},
        "mean": f(m.cmax.mean()), "sd": f(m.cmax.std()),
    }
    print(f"{key}: cmax {m.cmax.mean():.3f} own vs {m.other.mean():.3f} other, spec1 {(m.spec == 1).mean():.2f}, "
          f"r(cmax, MW z) {rr[0]:.3f}", flush=True)

a = np.array([o["m"]["pl"]["cmax"] for o in out["recalls"]])
b = np.array([o["m"]["mini"]["cmax"] for o in out["recalls"]])
out["summary"]["models_r"] = {"r": f(stats.pearsonr(a, b)[0], 3), "rho": f(stats.spearmanr(a, b)[0], 3)}

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(out_path)), "embeddings.npz"),
                    story_sents=np.array(story_sents), recall_sents=np.array(recall_sents), **npz)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
print(f"{len(rec)} recalls embedded ({n_rows - len(rec)} rows without sentences), {len(story_sents)} story sentences, "
      f"models: {', '.join(n for _, n in MODELS)}; r(pl, mini) = {out['summary']['models_r']['r']}; "
      f"verbatim recall=story sentences: {verbatim}")
