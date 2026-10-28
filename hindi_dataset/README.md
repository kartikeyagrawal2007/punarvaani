# Hindi Dataset Directory

Drop your paired video and transcript files directly into this directory:

```text
hindi_dataset/
├── data001.mp4
├── data001.txt       <-- Contains Devanagari text (e.g., "नमस्ते भारत")
├── data002.mp4
├── data002.txt       <-- Contains Devanagari text
├── ...
├── data999.mp4
└── data999.txt
```

### Notes:
- Each video file (`.mp4`, `.avi`, `.mov`, `.mkv`) must have a corresponding `.txt` file with the exact same base name.
- The `.txt` file should contain UTF-8 encoded Hindi Devanagari text.
- By default, `--val_split 0.1` will automatically split 10% of these files for validation.
