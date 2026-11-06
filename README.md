# SemCoupling

**Linking brain and behavior to cognitive states during naturalistic experience**

BrainHack Kraków 2026 · Michał Domagała (team lead) · Marcin Leszczyński (mentor)
Brain and Cognition Lab, Cognitive Science Center, Jagiellonian University

---

## About

Traditional paradigms study cognitive functions in isolation, with simple, tightly controlled stimuli. But functions such as attention, memory and perception interact, and simple tasks cannot show how they work together in real life. **Naturalistic paradigms** (listening to stories, reading, watching films) bring the task closer to everyday experience.

Our case is **mind wandering**: a private, largely unconscious state in which thoughts drift away from the task at hand, towards the past, the future, daydreams, or nothing at all (**cognitive decoupling**). In naturalistic settings it is rich in content and strongly depends on the surroundings.

Naturalistic paradigms are not a skeleton key, though. They bring problems of their own.

> **Overarching goal:** gain new insights into the nature of cognitive decoupling by tackling the problems of naturalistic paradigms.

## Challenges and research directions

| Challenge                                                                                                                                                                                                                                                | Research direction                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **The cognitive state is still a theoretical, latent construct.** Its functions (memory consolidation, future planning, creative incubation) are postulated from behavioral indicators only.                                                       | **A closer look at the function of mind wandering:** model it externally (e.g. in LLMs) and test the predictions of the major theories.                                     |
| **Indicators of the state are sparse** compared with the stimulus (one thought probe per ~45 s fragment), so the onset of an episode is unknown. Yet mind wandering has correlates in continuous measures (pupil, saccades, reaction times, ERPs). | **Continuous detection:** treat the state as a latent variable and combine continuous behavioral indicators with thought probes, using ML, to detect the onset of episodes. |
| **Complex stimuli mean complex cognitive states.** Free thought recalls describe what participants were thinking.                                                                                                                                  | **Characterize episodes from thought recalls:** use NLP (e.g. semantic similarity between recalls and the stimulus) to describe decoupling episodes.                        |

---

## Repository structure

```
SemCoupling/
├── README.md
├── Experiments_and_data_description.docx  ← detailed description of experiments and data
├── requirements.txt                       ← core stack (data, plotting, Jupyter)
├── requirements-nlp.txt                   ← optional NLP stack (torch, SBERT, gensim, spaCy, Morfeusz2)
├── setup_env.ps1                          ← environment setup (Windows)
│
├── Strategic_MW_Viz.ipynb                 ← start here: our story-listening dataset
├── ROAMM_Viz.ipynb                        ← start here: ROAMM (EEG + eye tracking during reading)
│
├── Analysis_Codes/
│   ├── analysisFunctions.py               ← eye-tracking plots and preprocessing helpers
│   ├── behavDataExtraction.py             ← EyeLink .asc → trial-wise behavioral and ocular tables
│   ├── storyDataExtraction.py             ← thought recalls + story text → lexical and semantic metrics
│   └── storyTools.py                      ← NLP tools: lemmatization, lexical diversity, SBERT similarity, WMD
│
└── Data/                                  ← not in git (see below)
    ├── Strategic_MW_Experiment/
    │   ├── behavioral.csv, behavioral_cleaned.csv
    │   ├── gaze_raw.csv.gz, gaze_preproc.csv.gz
    │   ├── fixations.csv.gz, saccades.csv.gz, blinks.csv.gz
    │   ├── audioFiles/ (expAudio/segm_*.wav, practice/)
    │   ├── Stories/
    │   └── polish.stopwords.txt
    └── ROAMM/
        ├── sub-XXXXX/*_mldata.pkl
        ├── stimuli/ (*_coordinates.csv, ROAMM_README.md)
        └── v1.0.0.zip                         ← full dataset archive (~22 GB)
```

## Data

`Data/` is excluded from git (size and privacy). Place the files as shown above.

**Strategic mind-wandering experiment** (ours). 58 participants (2 × 29) listened to a Polish story about two protagonists, split into 40 fragments of ~45 s. After each fragment: a thought probe (0–100, "How distracted are you from the story?") and a free recall of thoughts. EyeLink 1000, 500 Hz, central fixation.

- `Exp1_Cued`: participants tracked one protagonist (tracked vs. untracked fragments).
- `Exp2_Preference`: no cue; participants rated how interesting each fragment was.

All tables share the keys `Experiment, Subject, TrialNum, Fragment, Hero`. The thought recalls are in `behavioral_cleaned.csv` (`Utterance`).

**ROAMM** (external, [DOI 10.82901/nemar.on007629](https://doi.org/10.82901/nemar.on007629)). 44 participants read 5 texts with simultaneous EEG and eye tracking. They pressed a key when they caught themselves mind wandering and marked the words the episode covered (span-level labels, at most one per page). One `.pkl` per run holds EEG, eye tracking and the `is_mw` label on a common timeline.

## Getting started

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1        # core
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Nlp   # + NLP stack (~2–3 GB)
.\.venv\Scripts\Activate.ps1
```

Then open a notebook with the kernel **Python (SemCoupling)**.

## Known gaps

- `behavDataExtraction.py` and `storyDataExtraction.py` still point to the original paths (`analysis/RawData`) and need to be redirected to `Data/Strategic_MW_Experiment/`.
- There is no feature-extraction or modeling module yet. **That is the core of the hackathon.**

## Contact

Michał Domagała · michal.domagala@doctoral.uj.edu.pl
