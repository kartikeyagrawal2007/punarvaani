"""Audio-visual dataset read from a JSONL manifest, plus collation and dynamic batching.

Manifest line (written by :mod:`punarvaani.data_engine` or the recording tool)::

    {"id": "news_0001_003", "video": "crops/news_0001_003.npy", "text": "...",
     "units": "units/news_0001_003.npy", "audio": "wav/news_0001_003.wav",
     "speaker": "ch12_anchor3", "mode": "spoken", "num_frames": 87, "source": "news"}

``video`` is a (T, 96, 96) uint8 array from :class:`LipCropper`; ``units`` is a
(2T,) int array of mHuBERT cluster ids at 50 Hz; ``mode`` is "spoken" or
"silent" (silent mouthing from volunteers/patients). Paths are relative to the
manifest's directory. Media never lives in git.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from punarvaani.models.encoder.avhubert import normalize_video
from punarvaani.text.vocab import CharVocab


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for r in rows:
        r["_root"] = str(path.parent)
    return rows


def write_manifest(rows: Sequence[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n"
            )


@dataclass
class VideoAugment:
    """Train: random crop + horizontal flip (+ optional brightness jitter). Eval: centre crop.

    Flipping is safe for lips (speech is left-right symmetric). Random crops
    simulate imperfect mouth tracking.
    """

    crop: int = 88
    train: bool = True
    flip_prob: float = 0.5
    brightness: float = 0.0

    def __call__(self, video: np.ndarray) -> np.ndarray:
        t, h, w = video.shape
        c = min(self.crop, h, w)
        if self.train:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        video = video[:, y : y + c, x : x + c]
        if self.train and random.random() < self.flip_prob:
            video = video[:, :, ::-1]
        if self.train and self.brightness > 0:
            gain = 1.0 + random.uniform(-self.brightness, self.brightness)
            video = np.clip(video.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(video)


class AVManifestDataset(Dataset):
    def __init__(self, manifest: str | Path | list[dict], vocab: CharVocab, augment: VideoAugment | None = None,
                 load_units: bool = False, max_frames: int | None = None, modes: Sequence[str] | None = None):  # fmt: skip
        rows = read_manifest(manifest) if not isinstance(manifest, list) else manifest
        if modes is not None:
            rows = [r for r in rows if r.get("mode", "spoken") in modes]
        if max_frames is not None:
            rows = [r for r in rows if r.get("num_frames", 0) <= max_frames]
        self.rows, self.vocab = rows, vocab
        self.augment = augment or VideoAugment(train=False)
        self.load_units = load_units

    def __len__(self) -> int:
        return len(self.rows)

    def num_frames(self, i: int) -> int:
        return int(self.rows[i].get("num_frames", 0))

    def _path(self, row: dict, key: str) -> Path:
        p = Path(row[key])
        return p if p.is_absolute() else Path(row.get("_root", ".")) / p

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        video = np.load(self._path(row, "video"))
        video = normalize_video(torch.from_numpy(self.augment(video)))
        item = {
            "id": row["id"],
            "video": video,
            "text": row.get("text", ""),
            "mode": row.get("mode", "spoken"),
        }
        if row.get("text"):
            item["tokens"] = torch.tensor(self.vocab.encode(row["text"]), dtype=torch.long)
        if self.load_units and "units" in row:
            units = torch.from_numpy(np.load(self._path(row, "units")).astype(np.int64))
            item["units"] = _fit_length(units, 2 * video.shape[0], fill=-100)
        return item


def _fit_length(x: torch.Tensor, n: int, fill: int) -> torch.Tensor:
    """Crop or pad a 1-D target to exactly ``n`` frames (audio/video lengths differ by a few frames)."""
    if x.shape[0] >= n:
        return x[:n]
    return torch.cat([x, torch.full((n - x.shape[0],), fill, dtype=x.dtype)])


def collate_av(items: list[dict], pad_id: int = 1) -> dict:
    lengths = torch.tensor([it["video"].shape[0] for it in items])
    t, h, w = int(lengths.max()), *items[0]["video"].shape[1:]
    video = torch.zeros(len(items), t, h, w)
    for i, it in enumerate(items):
        video[i, : it["video"].shape[0]] = it["video"]
    batch = {"ids": [it["id"] for it in items], "video": video, "lengths": lengths,
             "texts": [it["text"] for it in items], "modes": [it["mode"] for it in items]}  # fmt: skip
    if all("tokens" in it for it in items):
        tl = torch.tensor([len(it["tokens"]) for it in items])
        tokens = torch.full((len(items), max(1, int(tl.max()))), pad_id, dtype=torch.long)
        for i, it in enumerate(items):
            tokens[i, : len(it["tokens"])] = it["tokens"]
        batch["tokens"], batch["token_lengths"] = tokens, tl
    if all("units" in it for it in items):
        units = torch.full((len(items), 2 * t), -100, dtype=torch.long)
        for i, it in enumerate(items):
            units[i, : len(it["units"])] = it["units"]
        batch["units"] = units
    if all("mel" in it for it in items):
        mel = torch.zeros(len(items), 4 * t, items[0]["mel"].shape[-1])
        for i, it in enumerate(items):
            mel[i, : len(it["mel"])] = it["mel"]
        batch["mel"] = mel
    return batch


class DynamicBatchSampler(Sampler[list[int]]):
    """Batches of similar-length clips capped by total *frames*, not by count.

    A fixed batch size wastes memory on padding (one 15 s clip padded next to
    1 s clips) or runs out of memory on long batches. Sorting into length
    buckets and capping ``max_frames`` per batch (fairseq's ``max_tokens``)
    keeps GPU memory roughly constant per step.
    """

    def __init__(self, lengths: Sequence[int], max_frames: int, shuffle: bool = True, seed: int = 0,
                 bucket_size: int = 100):  # fmt: skip
        self.lengths, self.max_frames, self.shuffle = list(lengths), max_frames, shuffle
        self.seed, self.epoch, self.bucket_size = seed, 0, bucket_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        # sort within large random chunks: similar lengths together, still random overall
        chunk = self.bucket_size * 8
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s : s + chunk], key=lambda i: self.lengths[i])
            batch, longest = [], 0
            for i in part:
                longest_new = max(longest, self.lengths[i])
                if batch and longest_new * (len(batch) + 1) > self.max_frames:
                    batches.append(batch)
                    batch, longest_new = [], self.lengths[i]
                batch.append(i)
                longest = longest_new
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
