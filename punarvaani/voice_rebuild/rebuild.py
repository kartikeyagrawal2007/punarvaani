"""Rebuild a voice profile from scraps: old voice notes / family videos -> what both paths need.

Steps (once per user, offline)::

    files ─► load 16 kHz ─► denoise ─► VAD speech regions ─► 1.5-4 s segments
          ─► quality score per segment (SNR, clipping, bandwidth)
          ─► ECAPA embedding per segment ─► cluster speakers (relatives talk too!)
          ─► keep the patient's cluster (closest to an enrolment clip, else the largest)
          ─► outputs:
               * speaker embedding  = quality-weighted mean of kept embeddings  (Path B vocoder)
               * reference clip     = best contiguous 5-12 s region of the patient (IndicF5, Path A)
               * clean segments     = everything kept                          (vocoder fine-tuning)

Why cluster? A family video of a birthday has five voices. Averaging all of
them would produce a voice that is nobody's. Agglomerative clustering with a
cosine-distance threshold needs no "number of speakers" up front.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from punarvaani.preprocessing.audio import SAMPLE_RATE, energy_vad, rms_normalize, save_audio
from punarvaani.voice_rebuild.denoise import denoise, quality_report


@dataclass
class Segment:
    source: str
    start_s: float
    end_s: float
    quality: float
    snr_db: float
    cluster: int = -1
    wav: torch.Tensor | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


def speech_regions(
    wav: torch.Tensor, min_gap_s: float = 0.3, min_len_s: float = 0.8
) -> list[tuple[int, int]]:
    """Sample ranges of speech from the energy VAD, merging gaps shorter than ``min_gap_s``."""
    vad = energy_vad(wav).tolist()
    regions, start = [], None
    for i, v in enumerate([*vad, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append([start, i])
            start = None
    merged: list[list[int]] = []
    for r in regions:
        if merged and (r[0] - merged[-1][1]) * 0.01 < min_gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    hop = 160
    return [(a * hop, min(b * hop, wav.numel())) for a, b in merged if (b - a) * 0.01 >= min_len_s]


def split_region(start: int, end: int, max_len_s: float = 4.0) -> list[tuple[int, int]]:
    n = max(1, int(np.ceil((end - start) / (max_len_s * SAMPLE_RATE))))
    edges = np.linspace(start, end, n + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def agglomerative_cosine(embeddings: torch.Tensor, threshold: float = 0.5) -> list[int]:
    """Average-linkage clustering; merge while the closest pair's cosine *distance* < threshold."""
    x = F.normalize(embeddings.float(), dim=-1)
    clusters = [[i] for i in range(len(x))]
    sim = x @ x.T
    while len(clusters) > 1:
        best, pair = -2.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = float(sim[clusters[a]][:, clusters[b]].mean())
                if s > best:
                    best, pair = s, (a, b)
        if 1 - best >= threshold:
            break
        a, b = pair
        clusters[a] += clusters.pop(b)
    labels = [0] * len(x)
    for c, members in enumerate(sorted(clusters, key=len, reverse=True)):
        for i in members:
            labels[i] = c
    return labels


@dataclass
class VoiceProfile:
    embedding: list[float]
    reference_text: str
    reference_path: str
    segments: list[dict]
    report: dict

    def save(self, out_dir: str | Path) -> None:
        Path(out_dir, "profile.json").write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8"
        )


class VoiceRebuilder:
    def __init__(self, embedder: Callable[[torch.Tensor], torch.Tensor], denoise_backend: str = "auto",
                 cluster_threshold: float = 0.5, min_quality: float = 0.2,
                 transcriber: Callable[[torch.Tensor], str] | None = None):  # fmt: skip
        self.embedder, self.denoise_backend = embedder, denoise_backend
        self.cluster_threshold, self.min_quality = cluster_threshold, min_quality
        self.transcriber = transcriber

    def segments(self, wavs: Sequence[tuple[str, torch.Tensor]]) -> list[Segment]:
        out = []
        for name, wav in wavs:
            clean = denoise(wav, SAMPLE_RATE, self.denoise_backend) if self.denoise_backend != "none" else wav
            for a, b in speech_regions(clean):
                for s, e in split_region(a, b):
                    piece = clean[s:e]
                    q = quality_report(piece)
                    out.append(Segment(name, s / SAMPLE_RATE, e / SAMPLE_RATE, q.score, q.snr_db, wav=piece))
        return out

    def build(self, wavs: Sequence[tuple[str, torch.Tensor]], out_dir: str | Path,
              enrollment: torch.Tensor | None = None) -> VoiceProfile:  # fmt: skip
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        segs = [s for s in self.segments(wavs) if s.quality >= self.min_quality]
        if not segs:
            raise ValueError("no usable speech found; record a few sentences or lower min_quality")
        embs = torch.stack([self.embedder(s.wav) for s in segs])
        labels = agglomerative_cosine(embs, self.cluster_threshold)
        for s, lab in zip(segs, labels):
            s.cluster = lab
        if enrollment is not None:
            e = F.normalize(self.embedder(enrollment).float(), dim=-1)
            centroids = {
                c: F.normalize(embs[[i for i, lab in enumerate(labels) if lab == c]].mean(0), dim=-1)
                for c in set(labels)
            }
            target = max(centroids, key=lambda c: float(centroids[c] @ e))
        else:
            target = 0  # labels are sorted by cluster size
        keep = [i for i, lab in enumerate(labels) if lab == target]
        weights = torch.tensor([max(segs[i].quality, 1e-3) for i in keep])
        embedding = F.normalize((F.normalize(embs[keep], dim=-1) * weights[:, None]).sum(0), dim=-1)

        # Reference for zero-shot TTS: the best-quality kept segments, concatenated up to ~10 s.
        ranked = sorted(keep, key=lambda i: segs[i].quality, reverse=True)
        ref_parts, total = [], 0.0
        for i in ranked:
            if total >= 10.0:
                break
            ref_parts.append(segs[i].wav)
            total += segs[i].duration
        silence = torch.zeros(int(0.25 * SAMPLE_RATE))
        ref = rms_normalize(torch.cat([p for part in ref_parts for p in (part, silence)]))
        ref_path = out_dir / "reference.wav"
        try:
            save_audio(ref_path, ref)
        except ImportError:
            torch.save(ref, ref_path.with_suffix(".pt"))
            ref_path = ref_path.with_suffix(".pt")
        ref_text = self.transcriber(ref) if self.transcriber else ""
        torch.save(embedding, out_dir / "speaker_embedding.pt")

        report = {
            "segments_total": len(segs),
            "clusters": int(max(labels) + 1),
            "target_cluster": target,
            "target_segments": len(keep),
            "target_minutes": round(sum(segs[i].duration for i in keep) / 60, 2),
            "mean_snr_db": round(float(np.mean([segs[i].snr_db for i in keep])), 1),
            "reference_seconds": round(total, 1),
        }
        seg_rows = [{k: v for k, v in asdict(s).items() if k != "wav"} for s in segs]
        profile = VoiceProfile(embedding.tolist(), ref_text, str(ref_path), seg_rows, report)
        profile.save(out_dir)
        return profile
