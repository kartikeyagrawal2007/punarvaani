import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from punarvaani.app.lab.classifiers import DTWClassifier, NeuralClassifier, cross_validate, dtw_distances
from punarvaani.app.lab.features import canonical_lips, lip_features
from punarvaani.app.lab.session import LabSession, TakeStore, summarize
from punarvaani.app.lab.synthetic import SyntheticFace, lip_points, mouth_track
from punarvaani.preprocessing.lip_crop import CANONICAL_STABLE, LIPS_INNER, LIPS_OUTER, STABLE_POINTS

PHRASES = ["हाँ", "नहीं", "मुझे पानी चाहिए", "दर्द हो रहा है"]


def _naive_dtw(q, t):
    n, m = len(q), len(t)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = np.linalg.norm(q[i - 1] - t[j - 1]) + min(
                acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            )
    return acc[n, m] / (n + m)


def test_batched_dtw_matches_textbook_dtw():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(17, 4))
    templates = [rng.normal(size=(m, 4)) for m in (5, 17, 31)]
    np.testing.assert_allclose(dtw_distances(q, templates), [_naive_dtw(q, t) for t in templates])


def _face(lips, scale=1.0, angle=0.0, shift=(0.0, 0.0)):
    rot = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    lm = np.zeros((478, 2))
    lm[list(STABLE_POINTS)] = CANONICAL_STABLE @ rot.T + shift
    lm[list(LIPS_OUTER + LIPS_INNER)] = lips @ rot.T + shift
    return lm


def test_lip_features_ignore_head_position_size_and_tilt():
    lips = lip_points(12.0, 50.0)
    a = lip_features(canonical_lips(_face(lips)))
    b = lip_features(canonical_lips(_face(lips, scale=1.7, angle=0.2, shift=(140.0, -60.0))))
    np.testing.assert_allclose(a, b, atol=1e-5)
    opened = lip_features(canonical_lips(_face(lip_points(18.0, 50.0))))
    assert np.abs(opened - a).max() > 0.1  # but they do see the mouth opening


def _takes(per_phrase=3, seed=0):
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for p in PHRASES:
        for _ in range(per_phrase):
            track = mouth_track(p, rng)
            seqs.append(lip_features(np.stack([lip_points(o, w) for o, w in track])))
            labels.append(p)
    return seqs, labels


def test_dtw_reader_learns_from_three_takes_and_rejects_unknown_phrases():
    seqs, labels = _takes()
    report = cross_validate(DTWClassifier, seqs, labels)
    assert report.accuracy >= 0.8 and report.n == len(seqs)
    clf = DTWClassifier().fit(seqs, labels)
    rng = np.random.default_rng(9)
    new = lip_features(np.stack([lip_points(o, w) for o, w in mouth_track(PHRASES[2], rng)]))
    assert clf.predict(new).label == PHRASES[2]
    still = lip_features(np.stack([lip_points(1.0, 50.0)] * 30))  # lips never move
    assert clf.predict(still).rejected


def test_neural_reader_learns_and_reports_progress():
    seqs, labels = _takes(per_phrase=4)
    steps = []
    clf = NeuralClassifier(steps=120, progress=lambda s, loss: steps.append(s)).fit(seqs, labels)
    assert steps[-1] == 120 and clf.losses[-1] < clf.losses[0]
    rng = np.random.default_rng(5)
    hits = sum(
        clf.predict(lip_features(np.stack([lip_points(o, w) for o, w in mouth_track(p, rng)]))).label == p
        for p in PHRASES
    )
    assert hits >= 3


def test_cross_validation_needs_two_takes_of_two_phrases():
    seqs, labels = _takes(per_phrase=1)
    assert cross_validate(DTWClassifier, seqs, labels) is None


def test_summary_counts_silence_separately_from_mistakes():
    rs = [{"target": "a", "predicted": "a", "rejected": False, "correct": True, "speech_mode": "silent", "latency_ms": 5},
          {"target": "a", "predicted": None, "rejected": True, "correct": False, "speech_mode": "silent", "latency_ms": 7},
          {"target": "b", "predicted": "a", "rejected": False, "correct": False, "speech_mode": "spoken", "latency_ms": 9}]  # fmt: skip
    s = summarize(rs)
    assert s["overall"]["accuracy"] == pytest.approx(1 / 3)
    assert s["overall"]["coverage"] == pytest.approx(2 / 3)
    assert s["overall"]["answered_accuracy"] == pytest.approx(1 / 2)
    assert s["confusion"]["a"] == {"a": 1, "(not sure)": 1} and s["by_mode"]["spoken"]["n"] == 1


def _session(tmp_path, **kw):
    pytest.importorskip("cv2")
    holder = {}
    face = SyntheticFace(lambda: holder["s"].target or PHRASES[0], seed=3)
    s = LabSession(TakeStore(tmp_path), face.landmarks, speaker="syn", phrases=PHRASES[:3], sync=True, **kw)
    holder["s"] = s
    return s, face


def _run_until(s, face, cond, limit=3000):
    for _ in range(limit):
        if cond():
            return True
        s.push_frame(face.read())
    return False


@pytest.mark.slow
def test_lab_records_trains_and_scores_itself(tmp_path):
    s, face = _session(tmp_path, neural_steps=40)
    assert _run_until(s, face, lambda: len(s.store.rows("train")) >= 9)
    counts = s.take_counts()
    assert set(counts) == set(PHRASES[:3]) and min(counts.values()) >= 2  # record mode rotates the prompts
    row = s.store.rows("train")[0]
    crops, lips = np.load(tmp_path / row["video"]), np.load(tmp_path / row["lips"])
    assert crops.shape[1:] == (96, 96) and crops.dtype == np.uint8 and lips.shape == (len(crops), 40, 2)
    assert row["num_frames"] == len(crops) and row["mode"] == "silent"  # project manifest format

    assert s.train() and s.training["status"] == "ready" and set(s.models) == {"dtw", "neural"}
    assert s.training["cv"]["dtw"]["accuracy"] >= 0.6

    s.configure(mode="test")
    assert _run_until(s, face, lambda: len(s.store.results()) >= 6)
    stats = s.snapshot()["stats"]["overall"]
    assert stats["n"] >= 6 and stats["accuracy"] >= 0.6
    assert s.last["task"] == "test" and s.last["kind"] == "prediction"

    before = len(s.store.rows("train"))
    s.delete_last_take()
    assert len(s.store.rows("train")) == before - 1


def test_hold_to_record_trigger(tmp_path):
    s, face = _session(tmp_path)
    s.configure(trigger="hold")
    for _ in range(20):
        s.push_frame(face.read())
    s.set_hold(True)
    for _ in range(15):
        s.push_frame(face.read())
    s.set_hold(False)
    s.push_frame(face.read())
    rows = s.store.rows("train")
    assert len(rows) == 1 and rows[0]["text"] == PHRASES[0] and rows[0]["num_frames"] >= 15


def test_dashboard_server_serves_state_and_accepts_commands(tmp_path):
    pytest.importorskip("cv2")
    from punarvaani.app.lab.server import Lab, make_handler

    s, face = _session(tmp_path)
    lab = Lab(s, face, paced=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lab))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for _ in range(5):
            lab._publish(face.read(), None)
        page = urllib.request.urlopen(url + "/").read().decode()
        assert "Lip Lab" in page
        req = urllib.request.Request(
            url + "/api/config", data=json.dumps({"mode": "free"}).encode(), method="POST"
        )
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        state = json.loads(urllib.request.urlopen(url + "/api/state").read())
        assert state["mode"] == "free" and state["target"] is None
        assert [p["text"] for p in state["phrases"]] == PHRASES[:3]
    finally:
        lab.stop()
        server.shutdown()
