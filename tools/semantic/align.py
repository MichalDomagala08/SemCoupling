"""Measured onset of every story word: forced alignment of the known fragment
text to its recording (torchaudio MMS_FA), plus a waveform envelope for the viewer.

usage: align.py behavioral_cleaned.csv audio_dir words.json [TrialNum ...]

Recording of trial N is <audio_dir>/segm_<N-1>.wav (durations match `Length`
within ~50 ms). The aligner knows only a-z and ', so Polish letters are folded;
each original word keeps its own entry, punctuation-only tokens get no times.
"""
import ast
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from torchaudio.pipelines import MMS_FA

FOLD = str.maketrans("ąćęłńóśźż", "acelnoszz")
ENV_HZ = 50

csv_path, audio_dir, out_path = sys.argv[1:4]
only = {int(a) for a in sys.argv[4:]}

df = pd.read_csv(csv_path)
frags = df.drop_duplicates("TrialNum").set_index("TrialNum").sort_index()
model, tok, aligner = MMS_FA.get_model(with_star=False).eval(), MMS_FA.get_tokenizer(), MMS_FA.get_aligner()
SR = MMS_FA.sample_rate

out = {"model": "torchaudio MMS_FA", "env_hz": ENV_HZ, "fragments": {}}
for trial, row in frags.iterrows():
    trial = int(trial)
    if only and trial not in only:
        continue
    t_start = time.time()
    words = []
    # empty sentences dropped as in embed.py::as_list, so `s` indexes the same list (KAROLINA_10 has one)
    for si, sent in enumerate(s for s in ast.literal_eval(row["StorySentences"]) if s.strip()):
        for w in sent.split():
            words.append({"w": w, "s": si, "k": re.sub(r"[^a-z']", "", w.lower().translate(FOLD))})
    keyed = [x for x in words if x["k"]]

    wav, sr = sf.read(f"{audio_dir}/segm_{trial - 1}.wav", dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    dur = len(wav) / sr
    env = np.sqrt(np.add.reduceat(wav.numpy() ** 2, np.arange(0, len(wav), sr // ENV_HZ)) / (sr // ENV_HZ))
    w16 = AF.resample(wav, sr, SR).unsqueeze(0)
    with torch.inference_mode():
        emission, _ = model(w16)
    spans = aligner(emission[0], tok([x["k"] for x in keyed]))
    sec = w16.size(1) / emission.size(1) / SR
    for x, sp in zip(keyed, spans):
        n = sum(s.end - s.start for s in sp)
        x.update(t0=round(sp[0].start * sec, 3), t1=round(sp[-1].end * sec, 3),
                 sc=round(sum(s.score * (s.end - s.start) for s in sp) / n, 3))
    for x in words:
        del x["k"]

    t0s = [x["t0"] for x in keyed]
    assert all(a <= b for a, b in zip(t0s, t0s[1:])), f"trial {trial}: onsets not monotonic"
    assert keyed[-1]["t1"] <= dur + 0.05, f"trial {trial}: last word ends after the recording"
    out["fragments"][str(trial)] = {"fragment": row["Fragment"], "dur": round(dur, 3), "words": words,
                                    "env": [round(float(v), 4) for v in env]}
    sc = np.mean([x["sc"] for x in keyed])
    print(f"{trial:2d} {row['Fragment']:12s} {len(keyed):3d} words, {dur:5.1f} s, "
          f"first {keyed[0]['t0']:.2f} s, last ends {keyed[-1]['t1']:.2f} s, mean score {sc:.3f}, "
          f"{time.time() - t_start:.1f} s", flush=True)

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"{len(out['fragments'])} fragments -> {out_path}")
