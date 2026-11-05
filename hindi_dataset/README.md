# Hindi Dataset Drop Folder

Put the dataset here in **speaker subfolders**. Each video must have a transcript with the
same base name.

```text
hindi_dataset/
├── spk01/
│   ├── spk01_001.mp4
│   ├── spk01_001.txt
│   ├── spk01_002.mp4
│   └── spk01_002.txt
├── spk02/
│   ├── spk02_001.mp4
│   └── spk02_001.txt
└── spk03/
    └── ...
```

Rules:

- At least **two speaker folders** are required (the validation split is speaker-disjoint).
- Clips should be **1.5–3 seconds**, frontal, mouth unobstructed.
- Transcripts must be **UTF-8 Devanagari only** (no English letters or digits; spell numbers out).
- Do not place videos directly in `hindi_dataset/`; they must be inside `spkXX/`.
- Dataset contents are gitignored; they are never committed.

Validate the drop before training:

```bash
python scripts/preflight.py --data_dir ./hindi_dataset --output ./preflight_report.json
```

Add `--crop_mode none` if the clips are already mouth-cropped (otherwise preflight expects
MediaPipe for on-the-fly cropping).

Optional: crop raw videos into mouth ROIs first:

```bash
python scripts/preprocess_lips.py --input_dir ./raw_videos --output_dir ./hindi_dataset --max_frames 96 --frame_size 224
```

See `TEAM_GUIDE.md` Part 1 for the full dataset instructions.
