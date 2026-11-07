"""Inject semantic.json (and timing.json, wordpupil.json, labelstats.json, if given) into template.html -> output html.

usage: build.py template.html semantic.json out.html [timing.json [wordpupil.json [labelstats.json]]] [--share]
--share: results only (no recall or story texts, no model summaries of them), asserted on the output.
"""
import json
import os
import sys

args = [a for a in sys.argv[1:] if a != "--share"]
share = "--share" in sys.argv
template, data_path, out = args[:3]
load = lambda p: json.load(open(p, encoding="utf-8"), parse_constant=lambda c: None)
d = load(data_path)
if len(args) > 3:
    d["timing"] = load(args[3])
if len(args) > 4:
    d["wordpupil"] = load(args[4])
if len(args) > 5:
    d["labels"] = load(args[5])

if share:
    # results only: recall texts, model summaries of them and story texts leave the file; subject codes
    # stay on purpose (Madara 2026-11-07: a reader with the data checks a trial in the CSV by its code)
    secret_texts = {s for r in d["recalls"] for s in r["rs"] if len(s) >= 25}
    for r in d["recalls"]:
        r["rs"] = []
        for m in r["m"].values():
            m.pop("best", None)
    for fr in d["fragments"]:
        secret_texts |= {s for s in fr.get("ss", []) if len(s) >= 25}
        fr["ss"] = []
    if "timing" in d:
        d["timing"]["per_recall"] = []
    if "labels" in d:
        d["labels"]["per_recall"] = [[x[0], x[1], x[2], None] if x else None for x in d["labels"]["per_recall"]]
        if "rate" in d["labels"]:
            d["labels"]["rate"]["reason"] = [None] * len(d["labels"]["rate"]["reason"])
    d["share"] = True
blob = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = open(template, encoding="utf-8").read().replace("/*__DATA__*/", blob)
if share:
    leaked = [t[:40] for t in secret_texts if t in html]
    assert not leaked, f"shared build still holds {len(leaked)} texts, e.g. {leaked[:2]}"
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out) / 1e6:.1f} MB")
