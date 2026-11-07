# Dashboard: behavioural + eye-tracking overview of the Strategic MW experiment

One self-contained HTML page (d3, no server) with: MW over the 40 fragments,
tracked vs untracked (Exp1), interest vs MW (Exp2), hero and time-on-task
effects, recall features and typical words, a fragment/recall browser, pupil
and blink features per trial, and a per-subject trial viewer. Every chart has
a one-paragraph interpretation and a table view.

**No data lives in this folder.** `template.html` renders an empty page until
you build it against your own copy of the data. The built file embeds the
recalls and the pupil traces, so it is gitignored (`drafts/*.html`,
`tools/dashboard/*.json`) and must never be committed or shared.

## Build

Requires Python 3.11+ with `pandas>=3`, `numpy`, `scipy` (the core stack from
`requirements.txt` is enough). Data as described in `Data/README.md`:

```
Data/Strategic_MW_Experiment/behavioral_cleaned.csv
Data/Strategic_MW_Experiment/gaze_preproc.csv.gz     (optional: eye section)
Data/Strategic_MW_Experiment/blinks.csv.gz           (optional: eye section)
```

From the repository root:

```bash
# 1. behavioural metrics -> JSON (seconds)
python tools/dashboard/analyze.py Data/Strategic_MW_Experiment/behavioral_cleaned.csv Analysis_Codes/polish.stopwords.txt tools/dashboard/data.json

# 2. pupil + blink features per trial -> JSON (reads the 184 MB gaze file in chunks, ~1 min)
python tools/dashboard/eye.py Data/Strategic_MW_Experiment/gaze_preproc.csv.gz Data/Strategic_MW_Experiment/blinks.csv.gz Data/Strategic_MW_Experiment/behavioral_cleaned.csv tools/dashboard/eye.json

# 3. inject the JSON into the template -> one HTML file
mkdir -p drafts
python tools/dashboard/build.py tools/dashboard/template.html tools/dashboard/data.json tools/dashboard/eye.json drafts/semcoupling-dashboard.html
```

Open `drafts/semcoupling-dashboard.html` in a browser. Skip step 2 if you do
not have the gaze files: step 3 then builds the page without the eye section.

With [uv](https://docs.astral.sh/uv/) and no virtualenv:
`uv run --with pandas --with numpy --with scipy python tools/dashboard/analyze.py …`.

## What the scripts compute

- `analyze.py`: per-trial, per-fragment and per-subject MW aggregates; paired
  tests for tracked vs untracked, Karolina vs Janek, trials 1–10 vs 31–40;
  interest vs MW correlation (pooled and within subject); recall features
  (length, first-person references, lexical overlap with the story fragment),
  log-odds of lemmas in high- vs low-MW recalls (median split of `MW_Estimate`).
- `eye.py`: 1-second means of pupil (left eye, right as fallback) and gaze;
  per-trial mean/SD pupil, gaze dispersion, blink rate (left eye, blinks
  longer than one sample); within-subject z-scores and their relation to MW;
  average pupil time course for high vs low MW trials. Trials with more than
  50 % missing pupil samples are excluded from pupil statistics; a subject with
  eye data but no behavioural rows (MIGO2607) is dropped.
- `build.py`: replaces the `/*__DATA__*/` placeholder in the template with
  the JSON (NaN becomes null).
