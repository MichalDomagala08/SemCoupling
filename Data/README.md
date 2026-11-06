# Data

This folder is intentionally empty in git (data size and participant privacy).
Put the data here so that the notebooks find them:

```
Data/
├── Strategic_MW_Experiment/          ← our story-listening experiment (shared separately by the organisers)
│   ├── behavioral_cleaned.csv        ← one row per trial: MW ratings, thought recalls, lemmas, story text
│   ├── behavioral.csv
│   ├── gaze_preproc.csv.gz           ← gaze & pupil, 100 Hz (used by Strategic_MW_Viz.ipynb)
│   ├── gaze_raw.csv.gz               ← gaze & pupil, 500 Hz (optional)
│   ├── fixations.csv.gz, saccades.csv.gz, blinks.csv.gz
│   ├── Stories/story3.txt
│   └── audioFiles/                   ← optional
└── ROAMM/                            ← external dataset, https://doi.org/10.82901/nemar.on007629
    ├── sub-10014/sub-10014_task-ReMind_run-01_mldata.pkl   ← one subject is enough to start
    └── stimuli/*_coordinates.csv
```

The full ROAMM archive is ~22 GB, so download just one or two subjects before the workshop.
