"""Speech units: self-supervised speech features -> k-means cluster IDs at 50 Hz.

Pipeline: 16 kHz audio -> mHuBERT-147 (a HuBERT trained on 147 languages,
including Hindi) -> features from a middle layer -> nearest k-means centroid.
Middle layers are used because HuBERT's top layers specialise toward its own
pretraining targets while middle layers are the most phonetic.

We fit our own k-means on Hindi features (the GPU k-means below, k-means++
init) instead of relying on an English codebook, so the unit inventory covers
Hindi sounds (retroflexes, aspirates). The same units are the targets for
Path B and the input of the unit vocoder.
"""

from __future__ import annotations

from pathlib import Path

import torch

from punarvaani.preprocessing.audio import SAMPLE_RATE


class KMeans:
    """Mini-batch-free Lloyd's k-means with k-means++ initialisation, on torch tensors."""

    def __init__(self, k: int, iters: int = 50, seed: int = 0, tol: float = 1e-4):
        self.k, self.iters, self.seed, self.tol = k, iters, seed, tol
        self.centroids: torch.Tensor | None = None

    @staticmethod
    def _sq_dists(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (x.pow(2).sum(1, keepdim=True) - 2 * x @ c.T + c.pow(2).sum(1)).clamp(min=0)

    def fit(self, x: torch.Tensor) -> KMeans:
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        x = x.float()
        n = x.shape[0]
        # k-means++: spread initial centroids out, picking each with probability ∝ D(x)^2
        idx = [int(torch.randint(n, (1,), generator=g))]
        d2 = self._sq_dists(x, x[idx]).squeeze(1)
        for _ in range(1, self.k):
            probs = (d2 / d2.sum().clamp(min=1e-12)).cpu()
            nxt = (
                int(torch.multinomial(probs, 1, generator=g))
                if probs.sum() > 0
                else int(torch.randint(n, (1,), generator=g))
            )
            idx.append(nxt)
            d2 = torch.minimum(d2, self._sq_dists(x, x[nxt : nxt + 1]).squeeze(1))
        c = x[idx].clone()
        prev = None
        for _ in range(self.iters):
            assign = self._sq_dists(x, c).argmin(1)
            sums = torch.zeros_like(c).index_add_(0, assign, x)
            counts = torch.bincount(assign, minlength=self.k).float().unsqueeze(1)
            empty = counts.squeeze(1) == 0
            c = torch.where(empty.unsqueeze(1), c, sums / counts.clamp(min=1))
            inertia = self._sq_dists(x, c).gather(1, assign.unsqueeze(1)).sum()
            if prev is not None and abs(prev - inertia) <= self.tol * max(prev, 1e-12):
                break
            prev = inertia
        self.centroids = c
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("fit() first")
        return self._sq_dists(x.float(), self.centroids.to(x.device)).argmin(1)

    def save(self, path: str | Path) -> None:
        torch.save({"centroids": self.centroids, "k": self.k}, path)

    @classmethod
    def load(cls, path: str | Path) -> KMeans:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        km = cls(blob["k"])
        km.centroids = blob["centroids"]
        return km


def dedup_units(units: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse runs: [5,5,5,9,9,2] -> units [5,9,2], durations [3,2,1].

    Speech-to-speech systems dedup and then predict durations. Path B doesn't
    need to (video gives timing), but dedup is useful for analysis and for
    training a text-to-unit model on top.
    """
    if units.numel() == 0:
        return units, units
    change = torch.ones_like(units, dtype=torch.bool)
    change[1:] = units[1:] != units[:-1]
    starts = torch.nonzero(change).squeeze(1)
    durations = torch.diff(torch.cat([starts, torch.tensor([units.numel()], device=units.device)]))
    return units[starts], durations


class HuBERTUnitExtractor:
    """mHuBERT-147 features (Hugging Face ``transformers``) + our k-means -> units at 50 Hz."""

    def __init__(self, kmeans: KMeans | str | Path, model_name: str = "utter-project/mHuBERT-147", layer: int = 9,
                 device: str = "cpu"):  # fmt: skip
        from transformers import HubertModel

        self.model = HubertModel.from_pretrained(model_name).eval().to(device)
        self.layer, self.device = layer, device
        self.kmeans = KMeans.load(kmeans) if isinstance(kmeans, (str, Path)) else kmeans

    @torch.no_grad()
    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """(S,) 16 kHz -> (frames, D). HuBERT's conv stack has stride 320 = 20 ms."""
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        out = self.model(wav.view(1, -1).to(self.device), output_hidden_states=True)
        return out.hidden_states[self.layer][0].cpu()

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return self.kmeans.predict(self.features(wav))

    @staticmethod
    def expected_frames(num_samples: int) -> int:
        return num_samples // (SAMPLE_RATE // 50)
