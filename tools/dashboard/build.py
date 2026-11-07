"""Inject data.json (+ eye.json if present) into template.html -> output html.

usage: build.py template.html data.json eye.json out.html [--share]
--share: results only, for the password-protected copy: no recall or story texts, no per-trial
eye traces; sections 5 and 7 (the browsers built on them) are left out. Asserted on the output.
"""
import json
import os
import sys

share = "--share" in sys.argv
template, data_path, eye_path, out = [a for a in sys.argv[1:] if a != "--share"][:4]
d = json.load(open(data_path, encoding="utf-8"), parse_constant=lambda c: None)
if os.path.exists(eye_path):
    d["eye"] = json.load(open(eye_path, encoding="utf-8"), parse_constant=lambda c: None)
if share:
    secret = {r["u"][:60] for r in d["recalls"] if len(r.get("u") or "") >= 40}
    secret |= {f["story"][:60] for f in d["fragments"] if f.get("story")}
    d["recalls"] = []
    for f in d["fragments"]:
        f["story"] = ""
    if "eye" in d:
        for k in ["pupil_1hz", "blinks_1", "trial_feats"]:
            d["eye"][k] = {}
    d["share"] = True
blob = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = open(template, encoding="utf-8").read().replace("/*__DATA__*/", blob)
if share:
    leaked = [t for t in secret if t in html]
    assert not leaked, f"shared build still holds {len(leaked)} texts, e.g. {leaked[:1]}"
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out) / 1e6:.1f} MB", "eye" if "eye" in d else "no eye")
