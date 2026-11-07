"""Inject data.json (+ eye.json if present) into template.html -> output html."""
import json
import os
import sys

template, data_path, eye_path, out = sys.argv[1:5]
d = json.load(open(data_path, encoding="utf-8"), parse_constant=lambda c: None)
if os.path.exists(eye_path):
    d["eye"] = json.load(open(eye_path, encoding="utf-8"), parse_constant=lambda c: None)
blob = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = open(template, encoding="utf-8").read().replace("/*__DATA__*/", blob)
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out) / 1e6:.1f} MB", "eye" if "eye" in d else "no eye")
