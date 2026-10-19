"""Webcam recorder for volunteers: shows each prompt, records spoken and silent takes.

    python -m punarvaani.recording.recorder --participant P07 --phrases configs/phrases_hi.txt \\
        --sentences my_sentences.txt --out recordings/P07

Controls: SPACE start/stop a take, R redo the last take, Q quit (progress is
saved after every take, so a session can be resumed). Output per take:
``P07_0012_silent.mp4`` (+ ``.wav`` from a separate audio recorder for spoken
takes) and a ``takes.jsonl`` manifest the data engine can crop directly.

Before recording: the participant has signed the consent form
(``punarvaani/recording/CONSENT_TEMPLATE.md``), the face is evenly lit from the
front, the phone/webcam is at mouth height ~40 cm away, and nothing covers
the mouth (no hand, no mic in front of the lips).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from punarvaani.recording.prompts import load_phrases, plan_session


def main(argv=None) -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--sentences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sentences = load_phrases(args.sentences) if args.sentences else []
    prompts = plan_session(load_phrases(args.phrases), sentences, seed=hash(args.participant) % 1000)
    done_file = out / "takes.jsonl"
    done = (
        {json.loads(line)["index"] for line in done_file.open(encoding="utf-8")}
        if done_file.exists()
        else set()
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FPS, 25)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    i = 0
    while i < len(prompts):
        p = prompts[i]
        if p.index in done:
            i += 1
            continue
        name = f"{args.participant}_{p.index:04d}_{p.mode}"
        writer, t0 = None, None
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if writer is not None:
                writer.write(frame)
            view = frame.copy()
            label = "बोलकर" if p.mode == "spoken" else "बिना आवाज़ के (silent)"
            # OpenCV cannot draw Devanagari; show the prompt index and show the text in the terminal.
            cv2.putText(view, f"#{p.index} {p.mode.upper()} {'REC' if writer else 'SPACE to start'}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if writer else (0, 200, 0), 2)  # fmt: skip
            cv2.imshow("recorder", view)
            if writer is None and t0 is None:
                print(f"\n[{p.index + 1}/{len(prompts)}] {label}:  {p.text}")
                t0 = 0
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and writer is None:
                writer, t0 = cv2.VideoWriter(str(out / f"{name}.mp4"), fourcc, 25, (w, h)), time.time()
            elif key == ord(" ") and writer is not None:
                writer.release()
                with done_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"index": p.index, "id": name, "text": p.text, "mode": p.mode, "kind": p.kind,
                                        "speaker": args.participant, "video_file": f"{name}.mp4",
                                        "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")  # fmt: skip
                i += 1
                break
            elif key == ord("r") and i > 0:
                i -= 1
                break
            elif key == ord("q"):
                cap.release()
                return
    cap.release()


if __name__ == "__main__":
    main()
