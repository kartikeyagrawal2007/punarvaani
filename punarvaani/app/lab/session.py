"""The lab's brain: frames in, takes and live predictions out.

Per frame (25 fps): landmarks -> stabilised mouth crop (the same causal cropper
the real-time app uses) + lip-shape features -> lip-activity end-pointing ->
buffer. When an utterance ends, what happens depends on the mode:

* ``record`` - save it as a training take of the phrase being prompted, then
  prompt the phrase with the fewest takes;
* ``test``   - predict it, score it against the prompted phrase, log the result;
* ``free``   - predict whatever was said.

Everything recorded lands in ``data/users/<name>/lab/`` as manifests in the same
format as the rest of the project (``video`` = 96x96 crops), so the takes can
later personalise the real lip reader (``configs/personalize.yaml``).
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from punarvaani.app.lab.classifiers import DTWClassifier, NeuralClassifier, Prediction, cross_validate
from punarvaani.app.lab.features import canonical_lips, lip_features
from punarvaani.app.streaming import LipActivityDetector, OnlineLipCropper
from punarvaani.utils import get_logger

log = get_logger(__name__)

MODES = ("record", "test", "free")
NOT_SURE = "(not sure)"


class TakeStore:
    """Phrases, takes (crops + lip points) and test results for one user, on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "takes").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # --- phrases
    def load_phrases(self, default: list[str]) -> list[str]:
        path = self.root / "phrases.txt"
        if path.exists():
            return [p.strip() for p in path.read_text(encoding="utf-8").splitlines() if p.strip()]
        self.save_phrases(default)
        return list(default)

    def save_phrases(self, phrases: list[str]) -> None:
        (self.root / "phrases.txt").write_text("\n".join(phrases) + "\n", encoding="utf-8")

    # --- takes
    def add(
        self, split: str, text: str, mode: str, speaker: str, crops: np.ndarray, lips: np.ndarray
    ) -> dict:
        tid = f"{split}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
        np.save(self.root / "takes" / f"{tid}_crop.npy", crops.astype(np.uint8))
        np.save(self.root / "takes" / f"{tid}_lips.npy", lips.astype(np.float32))
        row = {"id": tid, "video": f"takes/{tid}_crop.npy", "lips": f"takes/{tid}_lips.npy", "text": text,
               "speaker": speaker, "mode": mode, "num_frames": len(crops), "source": "lab"}  # fmt: skip
        with self._lock, open(self.root / f"{split}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def rows(self, split: str = "train") -> list[dict]:
        path = self.root / f"{split}.jsonl"
        if not path.exists():
            return []
        with self._lock:
            return [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]

    def lips(self, row: dict) -> np.ndarray:
        return np.load(self.root / row["lips"])

    def delete(self, split: str, tid: str) -> dict | None:
        rows = self.rows(split)
        gone = next((r for r in rows if r["id"] == tid), None)
        if gone is None:
            return None
        with self._lock:
            with open(self.root / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    if r["id"] != tid:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for key in ("video", "lips"):
                (self.root / gone[key]).unlink(missing_ok=True)
        return gone

    # --- live test results
    def log_result(self, result: dict) -> None:
        with self._lock, open(self.root / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def results(self) -> list[dict]:
        path = self.root / "results.jsonl"
        if not path.exists():
            return []
        with self._lock:
            return [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]

    def archive_results(self) -> None:
        path = self.root / "results.jsonl"
        if path.exists():
            path.rename(self.root / f"results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def _pred_dict(p: Prediction | None) -> dict | None:
    if p is None:
        return None
    return {"label": p.label, "confidence": round(p.confidence, 3), "rejected": p.rejected,
            "top": [[lab, round(c, 3)] for lab, c in p.top]}  # fmt: skip


def summarize(results: list[dict]) -> dict:
    """Live test numbers: accuracy, how often it stayed silent, per phrase, silent vs spoken, confusions."""

    def block(rs):
        n = len(rs)
        correct = sum(r["correct"] for r in rs)
        answered = sum(not r["rejected"] for r in rs)
        return {"n": n, "correct": correct, "accuracy": correct / n if n else None,
                "coverage": answered / n if n else None,
                "answered_accuracy": correct / answered if answered else None}  # fmt: skip

    per_phrase: dict[str, list] = {}
    by_mode: dict[str, list] = {}
    confusion: dict[str, Counter] = {}
    for r in results:
        per_phrase.setdefault(r["target"], []).append(r)
        by_mode.setdefault(r["speech_mode"], []).append(r)
        confusion.setdefault(r["target"], Counter())[r["predicted"] or NOT_SURE] += 1
    lat = sorted(r["latency_ms"] for r in results)
    return {
        "overall": block(results),
        "by_mode": {k: block(v) for k, v in by_mode.items()},
        "per_phrase": {k: block(v) for k, v in per_phrase.items()},
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "latency_p50": lat[len(lat) // 2] if lat else None,
        "recent": results[-12:][::-1],
    }


class LabSession:
    def __init__(self, store: TakeStore, landmarker: Callable[[np.ndarray], np.ndarray | None], *, speaker: str,
                 phrases: list[str], open_vocab: Callable | None = None, sync: bool = False, neural_steps: int = 300,
                 min_frames: int = 8, max_frames: int = 250):  # fmt: skip
        self.store, self.landmarker, self.speaker = store, landmarker, speaker
        self.open_vocab, self.sync, self.neural_steps = open_vocab, sync, neural_steps
        self.min_frames, self.max_frames = min_frames, max_frames
        self.phrases = store.load_phrases(phrases)
        self.lock = threading.RLock()
        self.cropper = OnlineLipCropper(out_size=96)
        self.sensitivity = 2.5
        self.detector = self._new_detector()
        self.mode, self.speech_mode, self.trigger = "record", "silent", "auto"
        self.hold = False
        self._was_active = False
        self.target: str | None = None
        self._lips_ema: np.ndarray | None = None
        self._prev_feat: np.ndarray | None = None
        self.pre_roll: deque = deque(maxlen=5)
        self.buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self.motion: deque = deque(maxlen=150)
        self.face = False
        self.frames = 0
        self._fps_t: deque = deque(maxlen=30)
        self.last: dict | None = None
        self.events: deque = deque(maxlen=14)
        self.models: dict[str, object] = {}
        self.training = {"status": "idle", "step": 0, "total": 0, "losses": [], "cv": {}, "primary": None,
                         "trained_on": 0, "message": ""}  # fmt: skip
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="lab-utterance")
        self._pick_target()
        self._refit_dtw()
        if not sync:
            self.train()  # takes from earlier sessions: get the neural reader back without a click

    # ------------------------------------------------------------------ settings
    def _new_detector(self) -> LipActivityDetector:
        return LipActivityDetector(on_ratio=self.sensitivity, off_ratio=0.65 * self.sensitivity, min_on_frames=3,
                                   hang_frames=10, min_floor=0.1, warmup=25)  # fmt: skip

    def configure(self, mode: str | None = None, speech_mode: str | None = None, trigger: str | None = None,
                  sensitivity: float | None = None) -> None:  # fmt: skip
        with self.lock:
            if mode in MODES and mode != self.mode:
                self.mode = mode
                self.buffer = []
                self._pick_target()
            if speech_mode in ("silent", "spoken"):
                self.speech_mode = speech_mode
            if trigger in ("auto", "hold"):
                self.trigger = trigger
                self.buffer, self.hold = [], False
            if sensitivity:
                self.sensitivity = float(np.clip(sensitivity, 1.5, 6.0))
                self.detector = self._new_detector()

    def set_hold(self, down: bool) -> None:
        with self.lock:
            self.hold = bool(down)

    def set_phrases(self, phrases: list[str]) -> None:
        phrases = list(dict.fromkeys(p.strip() for p in phrases if p.strip()))
        with self.lock:
            self.phrases = phrases
            self.store.save_phrases(phrases)
            if self.target not in phrases:
                self._pick_target()

    def skip(self) -> None:
        with self.lock:
            self._pick_target(avoid=self.target)

    def take_counts(self) -> Counter:
        return Counter(r["text"] for r in self.store.rows("train"))

    def _pick_target(self, avoid: str | None = None) -> None:
        if self.mode == "free" or not self.phrases:
            self.target = None
            return
        if self.mode == "record":
            counts = self.take_counts()
            order = sorted(self.phrases, key=lambda p: (counts[p], p == avoid, self.phrases.index(p)))
            self.target = order[0]
            return
        known = [p for p in self.phrases if p in self._known_classes()] or self.phrases
        choices = [p for p in known if p != avoid] or known
        self.target = random.choice(choices)

    def _known_classes(self) -> set[str]:
        return {c for m in self.models.values() for c in getattr(m, "classes", [])}

    # ------------------------------------------------------------------ frames
    def push_frame(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Process one frame; returns the landmarks (for drawing) or None if no face."""
        lm = self.landmarker(frame_rgb)
        with self.lock:
            self.frames += 1
            self._fps_t.append(time.perf_counter())
            self.face = lm is not None
            crop = self.cropper(frame_rgb, lm)
            if lm is not None:
                lips = canonical_lips(lm)
                self._lips_ema = lips if self._lips_ema is None else 0.7 * lips + 0.3 * self._lips_ema
            if crop is None or self._lips_ema is None:
                return lm
            self.last_crop = crop
            feat = lip_features(self._lips_ema)
            motion = (
                0.0
                if self._prev_feat is None or lm is None
                else 100 * float(np.abs(feat - self._prev_feat).mean())
            )
            self._prev_feat = feat
            if self.trigger == "auto":
                event = self.detector.update(motion)
                active, ended = self.detector.active, bool(event and event.kind == "end")
            else:
                active = self.hold
                ended = self._was_active and not active
                self.detector.history.append(motion)  # keep the noise floor meaningful for the chart
            self._was_active = active
            floor = self.detector.floor()
            self.motion.append((round(motion, 3), active))
            self.thresholds = (floor * self.detector.on_ratio, floor * self.detector.off_ratio)
            item = (crop, self._lips_ema.copy())
            if active or ended:
                if not self.buffer:
                    self.buffer.extend(self.pre_roll)
                self.buffer.append(item)
            else:
                self.pre_roll.append(item)
            if self.buffer and (ended or len(self.buffer) >= self.max_frames):
                utt, self.buffer = self.buffer, []
                job = (np.stack([c for c, _ in utt]), np.stack([x for _, x in utt]), self.mode, self.target,
                       self.speech_mode, time.perf_counter())  # fmt: skip
                if self.sync:
                    self._handle(*job)
                else:
                    self.pool.submit(self._safe_handle, *job)
        return lm

    @property
    def fps(self) -> float:
        t = self._fps_t
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    # ------------------------------------------------------------------ utterances
    def _event(self, text: str, kind: str = "info") -> None:
        self.events.appendleft({"t": time.strftime("%H:%M:%S"), "text": text, "kind": kind})

    def _safe_handle(self, *job) -> None:
        try:
            self._handle(*job)
        except Exception as e:  # keep the camera loop alive whatever happens
            log.exception("utterance failed")
            self._event(f"error: {e}", "bad")

    def _handle(self, crops, lips, mode, target, speech_mode, t_end) -> None:
        if len(crops) < self.min_frames:
            self._event(f"ignored a {len(crops)}-frame blip (too short)", "muted")
            return
        if mode == "record":
            if target is None:
                return
            self.store.add("train", target, speech_mode, self.speaker, crops, lips)
            n = self.take_counts()[target]
            self._event(f"saved take {n} of “{target}” ({speech_mode}, {len(crops)} frames)", "good")
            with self.lock:
                self.last = {"kind": "take", "target": target, "frames": len(crops)}
                if self.mode == "record":
                    self._pick_target(avoid=target)
            self._refit_dtw()
            return
        feats = lip_features(lips)
        preds = {name: m.predict(feats) for name, m in list(self.models.items())}
        primary_name = self.training.get("primary") or ("dtw" if "dtw" in preds else None)
        primary = preds.get(primary_name)
        text = None
        if self.open_vocab is not None:
            try:
                o = crops.shape[1] // 2 - 44
                text = self.open_vocab(crops[:, o : o + 88, o : o + 88])
            except Exception as e:
                text = f"(open-vocabulary model failed: {e})"
        latency = (time.perf_counter() - t_end) * 1e3
        result = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": mode, "target": target,
                  "speech_mode": speech_mode, "frames": len(crops), "primary": primary_name,
                  "predicted": primary.label if primary else None,
                  "rejected": primary.rejected if primary else True,
                  "predictions": {k: _pred_dict(v) for k, v in preds.items()}, "open_vocab": text,
                  "latency_ms": round(latency, 1)}  # fmt: skip
        if primary is None:
            self._event("no model yet: record at least 2 takes of 2 phrases", "bad")
        if mode == "test" and target is not None:
            result["correct"] = bool(primary and primary.label == target)
            self.store.log_result(result)
            self.store.add("test", target, speech_mode, self.speaker, crops, lips)
            verdict = "✓" if result["correct"] else ("silent" if result["rejected"] else "✗")
            self._event(f"{verdict}  said “{target}”, heard “{result['predicted'] or NOT_SURE}”",
                        "good" if result["correct"] else "bad")  # fmt: skip
            with self.lock:
                if self.mode == "test":
                    self._pick_target(avoid=target)
        else:
            self._event(f"heard “{result['predicted'] or NOT_SURE}”", "info")
        with self.lock:
            self.last = {"kind": "prediction", **result}

    # ------------------------------------------------------------------ models
    def _training_data(self) -> tuple[list[np.ndarray], list[str]]:
        rows = [r for r in self.store.rows("train") if r["text"] in self.phrases]
        return [lip_features(self.store.lips(r)) for r in rows], [r["text"] for r in rows]

    def _refit_dtw(self) -> None:
        seqs, labels = self._training_data()
        if len(set(labels)) >= 2:
            dtw = DTWClassifier().fit(seqs, labels)
            with self.lock:
                self.models["dtw"] = dtw
                if self.training.get("primary") is None:
                    self.training["primary"] = "dtw"

    def train(self) -> bool:
        """Cross-validate both readers, then fit them on every take (in the background unless ``sync``)."""
        if self.training["status"] == "training":
            return False
        seqs, labels = self._training_data()
        counts = Counter(labels)  # a phrase with a single take can't be held out yet, so it waits
        keep = [i for i, lab in enumerate(labels) if counts[lab] >= 2]
        seqs, labels = [seqs[i] for i in keep], [labels[i] for i in keep]
        if len(set(labels)) < 2:
            self.training["message"] = "record at least 2 takes of at least 2 phrases first"
            return False
        self.training.update(status="training", step=0, losses=[], message="cross-validating DTW")
        if self.sync:
            self._train(seqs, labels)
        else:
            threading.Thread(target=self._train, args=(seqs, labels), daemon=True, name="lab-train").start()
        return True

    def _train(self, seqs, labels) -> None:
        try:
            folds = 4
            total = (folds + 1) * self.neural_steps
            self.training["total"] = total
            dtw_cv = cross_validate(DTWClassifier, seqs, labels)
            losses, fits_done = self.training["losses"], [0]

            def progress(step, loss):  # one bar across the 4 held-out fits and the final fit
                self.training["step"] = fits_done[0] * self.neural_steps + step
                if fits_done[0] == folds:
                    losses.append(round(loss, 4))
                if step == self.neural_steps:
                    fits_done[0] += 1

            def make_neural():
                return NeuralClassifier(steps=self.neural_steps, progress=progress)

            self.training["message"] = "cross-validating the neural reader"
            neural_cv = cross_validate(make_neural, seqs, labels, folds=folds)
            fits_done[0] = folds
            self.training["message"] = "training the neural reader on every take"
            neural = make_neural().fit(seqs, labels)
            dtw = DTWClassifier().fit(seqs, labels)
            cv = {
                "dtw": dtw_cv.to_dict() if dtw_cv else None,
                "neural": neural_cv.to_dict() if neural_cv else None,
            }
            primary = "neural" if neural_cv and dtw_cv and neural_cv.accuracy > dtw_cv.accuracy else "dtw"
            with self.lock:
                self.models = {"dtw": dtw, "neural": neural}
                self.training.update(status="ready", step=total, cv=cv, primary=primary, trained_on=len(seqs),
                                     message=f"trained on {len(seqs)} takes of {len(set(labels))} phrases")  # fmt: skip
            self._event(
                f"trained: DTW {dtw_cv.accuracy:.0%}, neural {neural_cv.accuracy:.0%} held-out accuracy",
                "good",
            )
        except Exception as e:
            log.exception("training failed")
            self.training.update(status="error", message=str(e))

    def delete_last_take(self) -> None:
        rows = self.store.rows("train")
        if rows:
            gone = self.store.delete("train", rows[-1]["id"])
            self._event(f"deleted the last take of “{gone['text']}”", "muted")
            with self.lock:
                if self.mode == "record":
                    self._pick_target()
            self._refit_dtw()

    def reset_results(self) -> None:
        self.store.archive_results()
        self._event("started a fresh test run (old results archived)", "muted")

    # ------------------------------------------------------------------ dashboard
    def snapshot(self) -> dict:
        counts_all = Counter((r["text"], r["mode"]) for r in self.store.rows("train"))
        with self.lock:
            phrases = [{"text": p, "silent": counts_all[(p, "silent")], "spoken": counts_all[(p, "spoken")]}
                       for p in self.phrases]  # fmt: skip
            on, off = getattr(self, "thresholds", (0.0, 0.0))
            training = {k: v for k, v in self.training.items() if k != "losses"}
            losses = self.training["losses"]
            training["losses"] = losses[:: max(1, len(losses) // 60)]
            return {
                "user": self.speaker, "mode": self.mode, "speech_mode": self.speech_mode, "trigger": self.trigger,
                "sensitivity": self.sensitivity, "target": self.target, "phrases": phrases,
                "camera": {"fps": round(self.fps, 1), "face": self.face, "frames": self.frames,
                           "calibrating": self.trigger == "auto" and self.detector.t < self.detector.warmup},
                "activity": {"active": bool(self.buffer), "buffered": len(self.buffer), "hold": self.hold,
                             "motion": [m for m, _ in self.motion], "speaking": [a for _, a in self.motion],
                             "on": round(on, 3), "off": round(off, 3)},
                "last": self.last, "events": list(self.events), "training": training,
                "models": sorted(self.models), "open_vocab": self.open_vocab is not None,
                "stats": summarize(self.store.results()),
            }  # fmt: skip
