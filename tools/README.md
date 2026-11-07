# tools/ — dashboard, semantic analysis, eye tracking

Two self-contained HTML views built from your local copy of the Strategic MW data:

- **Dashboard** (`tools/dashboard/`): MW over the 40 fragments, the hypotheses
  (tracked vs untracked, interest, hero, time on task), recall features, pupil
  and blink features per trial, a fragment/recall browser and a per-subject
  trial viewer.
- **Semantic analysis** (`tools/semantic/`): how close each recall is to the
  fragment just heard (two sBERT models), which story sentences came back and
  what the pupil did meanwhile, pupil · MW · text within subject, pupil around
  every story word (word onsets measured on the audio), thought categories and a
  distraction rating from Claude Haiku.

Each folder's README describes its measures. This file is the run order.

**No data is in the repository and none may enter it.** Every script reads
`Data/` (gitignored) and writes JSON, embeddings and HTML that are gitignored
too (`tools/dashboard/*.json`, `tools/semantic/*.json`, `tools/semantic/*.np[yz]`,
`drafts/**/*.html`). The built pages embed participants' recalls: keep them on
your machine.

## What you need

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (every command below pulls
  its own packages with `uv run --with …`; no virtualenv to set up). With the
  repo's `.venv` instead, drop the `uv run --with …` prefix.
- The data, as distributed by the team lead:

  ```
  Data/Strategic_MW_Experiment/behavioral_cleaned.csv
  Data/Strategic_MW_Experiment/gaze_preproc.csv.gz   (eye sections)
  Data/Strategic_MW_Experiment/blinks.csv.gz         (eye sections)
  Data/audioFiles/expAudio/segm_0.wav … segm_39.wav  (word onsets, optional)
  ```

- For the Claude steps only: an Anthropic API key of your own, and the study
  lead's OK to send the recall texts to the Anthropic API (only the fragment
  text and the recall text are sent; the scripts assert that no subject code is
  in any request).

## Run order

From the repository root. First runs download models (~1 GB sBERT, ~1.2 GB for
the aligner). Measured on a laptop: step 1 and 2 under half a minute each, step 3
(audio alignment) about 8 minutes, step 4 under 15 s, step 5 about 4 and 6 minutes
for the two `--all` runs.

```bash
mkdir -p drafts

# --- 1. dashboard (behaviour + eye) ---
uv run --with pandas --with numpy --with scipy python tools/dashboard/analyze.py \
  Data/Strategic_MW_Experiment/behavioral_cleaned.csv Analysis_Codes/polish.stopwords.txt tools/dashboard/data.json
uv run --with pandas --with numpy --with scipy python tools/dashboard/eye.py \
  Data/Strategic_MW_Experiment/gaze_preproc.csv.gz Data/Strategic_MW_Experiment/blinks.csv.gz \
  Data/Strategic_MW_Experiment/behavioral_cleaned.csv tools/dashboard/eye.json
python tools/dashboard/build.py tools/dashboard/template.html tools/dashboard/data.json tools/dashboard/eye.json \
  drafts/semcoupling-dashboard.html

# --- 2. semantic coupling (sBERT) ---
uv run --with sentence-transformers --with pandas --with numpy --with scipy python tools/semantic/embed.py \
  Data/Strategic_MW_Experiment/behavioral_cleaned.csv tools/semantic/semantic.json

# --- 3. word onsets from the audio (optional; without it sentence onsets are estimated) ---
uv run --with torch --with torchaudio --with soundfile --with pandas --with numpy python tools/semantic/align.py \
  Data/Strategic_MW_Experiment/behavioral_cleaned.csv Data/audioFiles/expAudio tools/semantic/words.json
python tools/semantic/build.py tools/semantic/align.html tools/semantic/words.json drafts/semcoupling-align.html

# --- 4. eye tracking against the text (needs tools/dashboard/eye.json from step 1) ---
uv run --with pandas --with numpy --with scipy python tools/semantic/timing.py \
  Data/Strategic_MW_Experiment/behavioral_cleaned.csv tools/semantic/semantic.json tools/semantic/embeddings.npz \
  tools/dashboard/eye.json tools/semantic/timing.json tools/semantic/words.json
uv run --with pandas --with numpy --with scipy python tools/semantic/wordpupil.py \
  Data/Strategic_MW_Experiment/gaze_preproc.csv.gz Data/Strategic_MW_Experiment/behavioral_cleaned.csv \
  tools/semantic/words.json Analysis_Codes/polish.stopwords.txt tools/semantic/wordpupil.json

# --- 5. Claude Haiku: thought categories and distraction rating (optional, costs money) ---
export ANTHROPIC_API_KEY=...      # or put ANTHROPIC_API_KEY=... in a .env at the repo root (gitignored)
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/labels.json --dry-run
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/labels.json --sample 50
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/labels.json --all
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/mwrate.json --rate --sample 50
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/mwrate.json --rate --all
uv run --with pandas --with numpy --with scipy python tools/semantic/labelstats.py \
  tools/semantic/semantic.json tools/semantic/labels.json tools/dashboard/eye.json tools/semantic/labelstats.json tools/semantic/mwrate.json

# --- 6. the semantic view ---
python tools/semantic/build.py tools/semantic/template.html tools/semantic/semantic.json drafts/semcoupling-semantic.html \
  tools/semantic/timing.json tools/semantic/wordpupil.json tools/semantic/labelstats.json
```

Open `drafts/semcoupling-dashboard.html`; its header links to the semantic view
when both files sit in `drafts/`.

Every step after 2 is optional, and the view leaves out what it lacks:
`build.py` takes its JSON files in this order and stops reading at the first one
you leave off (no `timing.json`: no sections 6–7; no `wordpupil.json`: no
section 8; no `labelstats.json`: no sections 9–10).

## Notes

- **Claude cost and caching.** Model `claude-haiku-5-5`. `--dry-run` prints one
  request and checks every payload for subject codes, without calling the API.
  `--sample 50` prints 50 answers with the measured cost and stops; look at them
  before `--all`. A full run was ~$0.45 (categories) and ~$0.26 (ratings) for
  1920 recalls. Answers are cached in the JSON, so a rerun only asks for what is
  missing.
- **Sharing.** `--share` on either `build.py` writes a results-only copy
  (no recall or story texts; the browsers built on them left out) and asserts it.
  Build both into `drafts/share/`. It is still derived from participant data:
  show it behind a login, never commit it.
- **Within subject.** Every relation between pupil, MW and text is computed
  within subject (z inside the subject, then per-subject statistics); pooled
  between-subject numbers are shown only as context.
