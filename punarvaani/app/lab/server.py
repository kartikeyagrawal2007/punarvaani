"""Lip Lab: a local web dashboard to record, train and test lip reading on your own face, live.

    punarvaani lab --user kartikey                 # webcam
    punarvaani lab --synthetic                     # cartoon face, no camera needed
    punarvaani lab --video clip.mp4                # a recorded video, looped

Then open http://127.0.0.1:8765. Everything runs on this computer; nothing is
uploaded. Takes are saved under data/users/<user>/lab/.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from punarvaani.app.lab.session import LabSession, TakeStore
from punarvaani.preprocessing.lip_crop import LIPS_INNER, LIPS_OUTER, STABLE_POINTS
from punarvaani.utils import get_logger

log = get_logger(__name__)

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_PHRASES = ["हाँ", "नहीं", "मुझे पानी चाहिए", "दर्द हो रहा है", "डॉक्टर को बुलाओ", "मुझे बाथरूम जाना है",
                   "धन्यवाद", "मैं ठीक हूँ"]  # fmt: skip
STATIC = Path(__file__).parent / "static"
TEAL, CORAL = (115, 122, 31), (74, 107, 255)  # BGR


def ensure_face_model(path: str | None) -> Path:
    if path:
        return Path(path)
    target = Path.home() / ".cache" / "punarvaani" / "face_landmarker.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading the MediaPipe face model (3.7 MB) to %s", target)
        urllib.request.urlretrieve(FACE_MODEL_URL, target)
    return target


class CameraSource:
    def __init__(self, index: int | str, loop: bool = False):
        import cv2

        self.cv2, self.loop, self.src = cv2, loop, index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Could not open {'camera ' + str(index) if isinstance(index, int) else index}. On a Mac, allow camera "
                "access for your terminal app in System Settings > Privacy & Security > Camera, then try again."
            )
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.paced = not isinstance(index, int)  # files play as fast as they decode unless we pace them

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB) if ok else None


class Lab:
    """Runs the frame loop in a thread and keeps the latest JPEGs for the MJPEG streams."""

    def __init__(self, session: LabSession, source, paced: bool, mirror: bool = True):
        import cv2

        self.cv2, self.session, self.source, self.paced, self.mirror = cv2, session, source, paced, mirror
        self.frame_jpeg = self.crop_jpeg = b""
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="lab-camera")

    def start(self) -> Lab:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False

    def _loop(self) -> None:
        period = 1 / 25
        next_t = next_proc = time.perf_counter()
        while self.running:
            frame = self.source.read()
            if frame is None:
                time.sleep(0.05)
                continue
            now = time.perf_counter()
            lm = None
            if now >= next_proc:  # webcams give ~30 fps; the models expect 25, so drop about 1 in 6
                next_proc = max(next_proc + period, now - period)
                try:
                    lm = self.session.push_frame(frame)
                except Exception:
                    log.exception("frame failed")
            self._publish(frame, lm)
            if self.paced:
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))

    def _publish(self, frame: np.ndarray, lm: np.ndarray | None) -> None:
        cv2 = self.cv2
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        speaking = bool(self.session.buffer)
        if lm is not None:
            for ring in (LIPS_OUTER, LIPS_INNER):
                pts = lm[list(ring)].astype(np.int32)
                cv2.polylines(img, [pts], True, CORAL if speaking else TEAL, 2, cv2.LINE_AA)
            for x, y in lm[list(STABLE_POINTS)]:
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        if self.mirror:
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        if w > 720:
            img = cv2.resize(img, (720, int(h * 720 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        crop = getattr(self.session, "last_crop", None)
        crop_buf = None
        if crop is not None:
            c = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
            _, crop_buf = cv2.imencode(
                ".jpg", cv2.flip(c, 1) if self.mirror else c, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
        with self.cond:
            self.frame_jpeg = buf.tobytes() if ok else self.frame_jpeg
            if crop_buf is not None:
                self.crop_jpeg = crop_buf.tobytes()
            self.cond.notify_all()


def make_handler(lab: Lab):
    session = lab.session

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet; errors still go to the logger
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(
                json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code
            )

        def _mjpeg(self, attr: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while lab.running:
                    with lab.cond:
                        lab.cond.wait(timeout=1.0)
                        jpg = getattr(lab, attr)
                    if not jpg:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                    self.wfile.write(f"{len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/video.mjpg":
                self._mjpeg("frame_jpeg")
            elif path == "/crop.mjpg":
                self._mjpeg("crop_jpeg")
            elif path == "/api/state":
                self._json(session.snapshot())
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            path = self.path.split("?")[0]
            ok = True
            if path == "/api/config":
                session.configure(
                    **{k: body.get(k) for k in ("mode", "speech_mode", "trigger", "sensitivity")}
                )
            elif path == "/api/hold":
                session.set_hold(body.get("down", False))
            elif path == "/api/phrases":
                session.set_phrases(body.get("phrases", []))
            elif path == "/api/train":
                ok = session.train()
            elif path == "/api/skip":
                session.skip()
            elif path == "/api/delete_last":
                session.delete_last_take()
            elif path == "/api/reset_results":
                session.reset_results()
            else:
                return self._json({"ok": False, "error": "unknown endpoint"}, 404)
            self._json({"ok": ok, "message": session.training.get("message", "")})

    return Handler


def build(args) -> tuple[Lab, LabSession]:
    store = TakeStore(Path(args.data_root) / args.user / "lab")
    phrases = DEFAULT_PHRASES
    if args.phrases:
        lines = Path(args.phrases).read_text(encoding="utf-8").splitlines()
        phrases = [p.strip() for p in lines if p.strip() and not p.startswith("#")]
    open_vocab = None
    if args.run:
        import torch

        from punarvaani.app.realtime import load_pipeline
        from punarvaani.models.encoder.avhubert import normalize_video

        pipe = load_pipeline(args.run)

        def open_vocab(crops: np.ndarray) -> str:
            return pipe.recognize(normalize_video(torch.from_numpy(crops)), speak=False).text

    holder: dict = {}
    if args.synthetic:
        from punarvaani.app.lab.synthetic import SyntheticFace

        source = SyntheticFace(lambda: holder["session"].target or (holder["session"].phrases or [None])[0])
        landmarker, paced = source.landmarks, True
    else:
        from punarvaani.preprocessing.lip_crop import MediaPipeLandmarker

        source = CameraSource(args.video if args.video else args.camera, loop=bool(args.video))
        landmarker, paced = MediaPipeLandmarker(ensure_face_model(args.face_model), fps=25.0), source.paced
    session = LabSession(store, landmarker, speaker=args.user, phrases=phrases, open_vocab=open_vocab)
    holder["session"] = session
    return Lab(session, source, paced=paced, mirror=not args.video).start(), session


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="punarvaani lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--user", default="me", help="whose lips: takes go to data/users/<user>/lab/")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="webcam index (0 = built-in)")
    src.add_argument("--video", help="use a video file instead of the webcam (looped)")
    src.add_argument("--synthetic", action="store_true", help="cartoon face, no camera needed")
    ap.add_argument(
        "--face-model", help="MediaPipe face_landmarker.task (downloaded automatically if omitted)"
    )
    ap.add_argument("--phrases", help="text file, one phrase per line (default: 8 everyday phrases)")
    ap.add_argument("--run", help="optional trained lip-reader run dir (config.yaml, vocab.json, best.pt)")
    ap.add_argument("--data-root", default="data/users")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    return ap


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    lab, _ = build(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(lab))
    server.daemon_threads = True
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"\n  Punarvaani Lip Lab running at {url}   (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lab.stop()
        server.server_close()
        close = getattr(lab.session.landmarker, "close", None)
        if close:
            close()  # MediaPipe needs an explicit close or it crashes on exit


if __name__ == "__main__":
    main()
