# Semantic coupling of recalls (sBERT)

A separate view from the dashboard: how close in meaning each thought recall
is to the story fragment the participant had just heard. Every recall sentence
(`Sentences`) and every story sentence (`StorySentences`) is embedded with two
sentence-transformers models:

- `sdadas/st-polish-paraphrase-from-mpnet` (Polish, 768 dims)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (the model in
  `Analysis_Codes/storyDataExtraction.py`, 384 dims)

**No data lives in this folder.** `semantic.json`, `embeddings.npz` and the
built `drafts/semcoupling-semantic.html` hold the participants' recalls, so
they are gitignored and must never be committed or shared.

## Build

From the repository root, with [uv](https://docs.astral.sh/uv/) (the models
download on the first run, about 1 GB together; then ~20 s on a laptop):

```bash
uv run --with sentence-transformers --with pandas --with numpy --with scipy \
  python tools/semantic/embed.py Data/Strategic_MW_Experiment/behavioral_cleaned.csv tools/semantic/semantic.json
uv run --with pandas --with numpy --with scipy \
  python tools/semantic/timing.py Data/Strategic_MW_Experiment/behavioral_cleaned.csv tools/semantic/semantic.json \
  tools/semantic/embeddings.npz tools/dashboard/eye.json tools/semantic/timing.json tools/semantic/words.json
uv run --with pandas --with numpy --with scipy \
  python tools/semantic/wordpupil.py Data/Strategic_MW_Experiment/gaze_preproc.csv.gz Data/Strategic_MW_Experiment/behavioral_cleaned.csv \
  tools/semantic/words.json Analysis_Codes/polish.stopwords.txt tools/semantic/wordpupil.json
uv run --with pandas --with numpy --with scipy \
  python tools/semantic/labelstats.py tools/semantic/semantic.json tools/semantic/labels.json \
  tools/dashboard/eye.json tools/semantic/labelstats.json tools/semantic/mwrate.json
python tools/semantic/build.py tools/semantic/template.html tools/semantic/semantic.json drafts/semcoupling-semantic.html \
  tools/semantic/timing.json tools/semantic/wordpupil.json tools/semantic/labelstats.json
```

`labels.json` and `mwrate.json` come from Claude Haiku (sections 9–10). Making
them sends the recall and fragment texts to the Anthropic API (mentor's OK,
2026-11-07); the key is `ANTHROPIC_API_KEY` or the gitignored `.env`. Answers are
cached, so a rerun costs nothing; the full runs cost ~$0.45 and ~$0.26:

```bash
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/labels.json --sample 50   # then --all
uv run --with anthropic python -I -u tools/semantic/label.py tools/semantic/semantic.json tools/semantic/mwrate.json --rate --sample 50   # then --all
```

`build.py … --share` (and the same flag on `tools/dashboard/build.py`) writes the
results-only copy: no recall or story texts and no Haiku summaries of them, subject
codes kept so a reader with the data can check a trial; the browsers built on the
texts are left out, and the build asserts no text is left in the output. Build both
into `drafts/share/` (gitignored). Even this copy is derived from participant data:
host it only behind a login, never in a repository.

`label.py --open N` is the open-coding pass the taxonomy was checked against
(free codes for N recalls, then grouped); it writes to the file you give it.

`timing.py` needs the dashboard's `eye.json` built first (`tools/dashboard/README.md`);
`words.json` comes from `align.py` (below) and is optional: without it sentence
onsets fall back to the character-share estimate. `build.py` leaves sections 6–7
out without `timing.json` and section 8 out without `wordpupil.json`.

Open `drafts/semcoupling-semantic.html`. The dashboard header links to it when
both files sit in `drafts/`.

## Measures (per recall, per model)

- `cmax` — coupling max: for every recall sentence the cosine of the closest
  sentence of its own fragment, averaged over recall sentences. This is
  `storyTools.bertSentenceCosineSim(aggregation="max", mode=0)` **without** its
  identical-item exclusion: for whole sentences a verbatim retelling is the
  strongest coupling, not a trivial match (none occur in the data anyway).
- `cent` — coupling centroid: cosine of the mean recall vector and the mean
  fragment vector.
- `spec` — specificity: rank of the own fragment among all 40 by centroid
  similarity (1 = the recall is closest to the fragment just heard).
- `prev` — `cmax` against the previous fragment of the same story
  (KAROLINA_5 → KAROLINA_4); null for the first fragment of each story.
- `best` — per recall sentence, the index of its best-matching story sentence
  in the own fragment and the cosine.

## Recalled seconds vs pupil (`timing.py`, section 6)

- Story sentence intervals are **estimated**: each sentence's share of the
  fragment's characters times the trial length (`Length`; Exp1 has none, so the
  median `Length` of the same fragment — the same recording). Not measured on audio.
- A story sentence is *recalled* when some recall sentence is closer to it than
  the 95th percentile of the same similarity to sentences of the other
  fragments of the same story (threshold per model, in `timing.json`).
- Pupil: `pupil_1hz` from `eye.json`, z within subject over seconds inside the
  fragment; *detrended* = minus the mean z of all trials at that second. The
  paired test compares per-subject means of recalled vs not-recalled seconds,
  on trials that have both.

## Pupil x MW x coupling per trial (`timing.py`, section 7)

- Trials with a recall and pupil (`eye.json::trial_feats.pz`, missing ≤ 50 %),
  subjects with at least 8 of them. Mean trial pupil, `MW_Estimate` and `cmax`
  are z-scored within subject, so every number is a within-subject relation.
- Per pair: pooled r over those z-scores, and the mean of per-subject r with a
  one-sample t. Regression `cmax ~ pupil + MW` (OLS on the z-scores).

## Thought categories and the model's distraction rating (`label.py`, `labelstats.py`, sections 9–10)

- Category per recall: taxonomy v2 after Stawarczyk et al. 2011 (story, association,
  task, environment, self_past / self_present / self_future, blank), the content
  that takes most of the report, plus `also`. Only the fragment and recall texts
  are sent; `label.py` asserts no subject ID is in any payload.
- Rating per recall: the participants' own probe question and scale, asked of the
  model in a separate request; the self-report is never in it.
- Eye features per category and every model-vs-probe correlation are within
  subject (z inside the subject, subjects with ≥ 8 trials); the partial r asks
  whether the rating adds anything beyond coupling max.

Rows without recall sentences (400, `has_thoughts == 0`) are left out.
`embeddings.npz` keeps the unit sentence vectors for both models
(`<model>_story`, `<model>_recall`, with `story_sents`, `recall_sents`,
`story_frag`) so later steps do not re-embed.

## Measured word onsets (`align.py`)

```bash
uv run --with torch --with torchaudio --with soundfile --with pandas --with numpy \
  python tools/semantic/align.py Data/Strategic_MW_Experiment/behavioral_cleaned.csv Data/audioFiles/expAudio tools/semantic/words.json
python tools/semantic/build.py tools/semantic/align.html tools/semantic/words.json drafts/semcoupling-align.html
```

- Forced alignment of the known fragment text (`StorySentences`) to its
  recording with torchaudio `MMS_FA` (~1.2 GB on the first run, then ~2 min for
  all 40). Trial N is `segm_<N-1>.wav`; durations match `Length` within ~50 ms.
- `words.json`: per trial, every word with its sentence index `s`, onset `t0`,
  offset `t1` (seconds from the start of the recording) and the aligner score
  `sc`; punctuation-only tokens have no times. Words with `sc < 0.5` (~0.5 %,
  mostly one-letter words merged into a neighbour) are best left out.
- `drafts/semcoupling-align.html` plays `../Data/audioFiles/expAudio/*.wav`, so
  it works only from `drafts/` next to the data; the audio is not embedded.

## Pupil around each story word (`wordpupil.py`, section 8)

- Pupil from `gaze_preproc` at 100 Hz (left eye, right as fallback), z within
  subject; trials with > 50 % missing pupil dropped. The gaze trial starts with
  its recording (trial 1: 42.57 s of gaze, 42.55 s of audio).
- Epoch −0.5..+2 s around every word onset from `words.json`, minus the mean of
  −0.5..0 s; words in the first 3 s and words with aligner score < 0.5 left out.
- Groups: all words, content vs function (`polish.stopwords.txt`), the first word
  of each sentence, and a control of as many random moments in the same trials.
- MW: `MW_Estimate` split at each subject's median. Curves are averaged within
  subject first; windows 0–0.5, 0.5–1, 1–2 s compared with paired t over subjects,
  plus word minus random moment per MW half.
