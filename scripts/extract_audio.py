"""
Per-video-frame audio features for every clip, saved next to its mouth crop.

    python scripts/extract_audio.py --input_dir <raw clips with audio> --output_dir <cropped tree>

For <input_dir>/<speaker>/<clip>.mp4 this writes <output_dir>/<speaker>/<clip>.audio.npy: an array of shape
(frames, 4, 80), float16. Each video frame (25 fps) gets the four 10 ms log-mel frames (80 bands, 16 kHz)
that fall inside it, so audio frame i lines up exactly with video frame i of the crop. Only the first
--max_frames frames are kept, as in preprocess_lips.py.

The features serve two purposes: an audio-only Stage-1 model with the same labels (`--frontend audio`), which
shows whether the transcripts themselves can be learned, and the audio target of audio-guided lip training
(`--audio_weight`), which gives the lip model a dense per-frame signal from the clip's own sound.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from functools import partial

import numpy as np

SAMPLE_RATE = 16000
HOP = 160          # 10 ms
WIN = 400          # 25 ms
N_FFT = 512
N_MELS = 80
MELS_PER_FRAME = 4  # 100 mel frames per second / 25 video frames per second
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular HTK-style mel filters."""
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (bins - lo) / max(mid - lo, 1e-9)
        falling = (hi - bins) / max(hi - mid, 1e-9)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


_FILTERS = mel_filterbank()
_WINDOW = np.hanning(WIN + 1)[:-1]


def log_mel(samples: np.ndarray) -> np.ndarray:
    """Mono float samples at 16 kHz -> (n_frames, 80) log-mel, one frame every 10 ms (frame t covers t*10 ms)."""
    samples = np.asarray(samples, dtype=np.float32)
    n = int(np.ceil(len(samples) / HOP))
    padded = np.pad(samples, (WIN // 2, WIN // 2 + n * HOP), mode="constant")
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = padded[idx] * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    return np.log(power @ _FILTERS.T + 1e-6).astype(np.float32)


def frame_features(mel: np.ndarray, num_frames: int) -> np.ndarray:
    """(n_mel_frames, 80) -> (num_frames, 4, 80): the four mel frames inside each video frame (zero-padded)."""
    need = num_frames * MELS_PER_FRAME
    if len(mel) < need:
        mel = np.concatenate([mel, np.full((need - len(mel), mel.shape[1]), mel.min() if len(mel) else 0.0,
                                           dtype=mel.dtype)])
    return mel[:need].reshape(num_frames, MELS_PER_FRAME, mel.shape[1])


def read_audio(path: str, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode a clip's audio to 16 kHz mono float32 with ffmpeg."""
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def _video_frames(path: str) -> tuple[int, float]:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return n, fps


def process_clip(video_path: str, input_dir: str, output_dir: str, max_frames: int, skip_existing: bool,
                 ffmpeg: str = "ffmpeg"):
    """Returns (video_path, None) on success or (video_path, reason)."""
    rel = os.path.relpath(video_path, input_dir)
    out = os.path.join(output_dir, os.path.splitext(rel)[0] + ".audio.npy")
    if skip_existing and os.path.exists(out):
        return video_path, None
    try:
        n, fps = _video_frames(video_path)
        if n <= 0:
            return video_path, "no video frames"
        if fps and abs(fps - 25.0) > 0.5:
            return video_path, f"video is {fps:.2f} fps, expected 25"
        samples = read_audio(video_path, ffmpeg)
        if len(samples) < HOP:
            return video_path, "no audio"
        feats = frame_features(log_mel(samples), min(n, max_frames))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out[:-4] + ".part.npy"
        np.save(tmp, feats.astype(np.float16))
        os.replace(tmp, out)
        return video_path, None
    except subprocess.CalledProcessError as exc:
        return video_path, f"ffmpeg failed: {exc.stderr.decode(errors='ignore')[:200]}"
    except Exception as exc:  # one bad clip must not stop the run
        return video_path, f"error: {exc}"


def main(argv=None) -> int:
    from tqdm import tqdm

    ap = argparse.ArgumentParser(description="Per-video-frame log-mel features for every clip")
    ap.add_argument("--input_dir", required=True, help="Raw clips WITH audio (e.g. dataset\\splits\\train)")
    ap.add_argument("--output_dir", required=True, help="The cropped tree to write .audio.npy files into")
    ap.add_argument("--max_frames", type=int, default=96)
    ap.add_argument("--workers", type=int, default=0, help="Clips in parallel; 0 = half the CPU cores (max 8)")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg if it is not on PATH")
    args = ap.parse_args(argv)

    videos = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.input_dir) for f in fs
                    if f.lower().endswith(VIDEO_EXTENSIONS) and ".part." not in f)
    if not videos:
        print(f"[Error] No videos under {args.input_dir}")
        return 1
    workers = args.workers if args.workers > 0 else max(1, min(8, (os.cpu_count() or 2) // 2))
    task = partial(process_clip, input_dir=args.input_dir, output_dir=args.output_dir, max_frames=args.max_frames,
                   skip_existing=args.skip_existing, ffmpeg=args.ffmpeg)
    failed = []
    print(f"Found {len(videos)} clips. Extracting audio features with {workers} workers...")
    if workers == 1:
        results = map(task, videos)
        pool = None
    else:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(task, videos, chunksize=8)
    try:
        for path, reason in tqdm(results, total=len(videos), desc="Audio"):
            if reason:
                failed.append((path, reason))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    ok = len(videos) - len(failed)
    print(f"\nDone: {ok}/{len(videos)} ({100.0 * ok / len(videos):.1f}%)")
    for path, reason in failed[:10]:
        print(f"  failed: {path}: {reason}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
