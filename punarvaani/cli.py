"""``punarvaani`` command line: one entry point for the common jobs.

punarvaani demo                       # synthetic end-to-end demo on CPU (~2 min)
punarvaani train-vsr --config ...     # Path A training (synthetic / Hindi / personalise)
punarvaani train-units --config ...   # Path B lips -> units
punarvaani train-vocoder --config ... # Path B unit vocoder
punarvaani evaluate --run ...         # CER/WER/VER, phrase mode, latency
punarvaani rebuild-voice --out profile/ notes/*.wav [--enroll me.wav]
punarvaani mine --urls urls.txt --out data/ --face-model face_landmarker.task
punarvaani lab --user kartikey        # live dashboard: record, train and test on your own lips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _demo(args) -> None:
    import torch

    from punarvaani.datasets.synthetic import SyntheticLipDataset
    from punarvaani.eval.evaluate import load_run
    from punarvaani.models.path_a.phrase_mode import PhraseBank
    from punarvaani.models.path_a.pipeline import PathAPipeline
    from punarvaani.models.path_a.tts import ToneTTS
    from punarvaani.training import train_vsr

    run = Path(args.out)
    if not (run / "best.pt").exists():
        print(f"training a tiny lip reader on the synthetic world -> {run} (CPU, ~1-2 min)")
        train_vsr.main(
            [
                "--config",
                "configs/synthetic_vsr.yaml",
                f"train.out_dir={run}",
                f"train.max_steps={args.steps}",
            ]
        )
    _, vocab, model = load_run(run)
    ds = SyntheticLipDataset(12, size=44, seed=4242)
    phrases = [ds.sentence(i) for i in range(6)]  # the "user's" phrase bank
    pipe = PathAPipeline(model, vocab, PhraseBank(phrases, vocab), tts=ToneTTS(), beam=6)
    print("\nphrase bank:", phrases)
    for i in range(12):
        item = ds[i]
        res = pipe.recognize(item["video"])
        print(f"said: {item['text']:<22} -> [{res.mode:>6}] {res.text:<22} conf={res.confidence:.2f} "
              f"{res.timings_ms['total']:.0f} ms" + (f"  (audio {res.audio.numel() / res.sample_rate:.1f}s)" if res.audio is not None else ""))  # fmt: skip
    torch.set_grad_enabled(True)


def _rebuild_voice(args) -> None:
    import torch

    from punarvaani.preprocessing.audio import load_audio
    from punarvaani.voice_rebuild.rebuild import VoiceRebuilder
    from punarvaani.voice_rebuild.speaker import ECAPATDNN

    ecapa = ECAPATDNN()
    if args.ecapa:
        ecapa.load_state_dict(torch.load(args.ecapa, map_location="cpu", weights_only=True))
    else:
        print(
            "warning: no --ecapa checkpoint given; speaker clustering uses untrained weights", file=sys.stderr
        )
    rebuilder = VoiceRebuilder(ecapa.embed, denoise_backend=args.denoise)
    wavs = [(Path(p).name, load_audio(p)) for p in args.inputs]
    profile = rebuilder.build(wavs, args.out, load_audio(args.enroll) if args.enroll else None)
    print(profile.report)


def _mine(args) -> None:
    from punarvaani.data_engine.pipeline import (
        FilterReport,
        HFTranscriber,
        MiningPipeline,
        download,
        extract_audio,
        write_splits,
    )
    from punarvaani.preprocessing.audio import load_audio
    from punarvaani.preprocessing.lip_crop import MediaPipeLandmarker, read_video_rgb

    out = Path(args.out)
    landmarker = MediaPipeLandmarker(args.face_model)
    pipe = MiningPipeline(out, landmarker.detect_all, HFTranscriber(args.asr_model))
    rows, total = [], FilterReport()
    for line in Path(args.urls).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        url, _, speaker = line.partition(" ")
        video_path = download(url, out / "raw")
        wav = load_audio(extract_audio(video_path, out / "raw" / f"{video_path.stem}.wav"))
        frames, _ = read_video_rgb(video_path)
        r, rep = pipe.process_video(
            video_path.stem, frames, wav, speaker=speaker.strip() or video_path.stem, source=url
        )
        rows += r
        total.merge(rep)
        print(video_path.stem, rep.summary())
    write_splits(rows, out, total)
    print(total.summary())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="punarvaani", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic end-to-end Path A demo")
    d.add_argument("--out", default="outputs/demo")
    d.add_argument("--steps", type=int, default=300)

    for name in ("train-vsr", "train-units", "train-vocoder", "evaluate"):
        sub.add_parser(name, add_help=False)
    sub.add_parser(
        "lab", add_help=False, help="live dashboard: record, train and test lip reading on your own face"
    )

    r = sub.add_parser("rebuild-voice", help="voice notes -> reference clip + speaker embedding")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--enroll", help="a clip known to be the patient")
    r.add_argument("--ecapa", help="trained ECAPA-TDNN state dict")
    r.add_argument("--denoise", default="auto", choices=["auto", "deepfilternet", "wiener", "none"])

    m = sub.add_parser("mine", help="mine Hindi AV clips from video URLs")
    m.add_argument("--urls", required=True, help="text file: '<url> <speaker_id>' per line")
    m.add_argument("--out", required=True)
    m.add_argument("--face-model", required=True)
    m.add_argument("--asr-model", default="openai/whisper-large-v3")

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "lab":
        from punarvaani.app.lab.server import main as lab_main

        lab_main(argv[1:])
        return
    if argv and argv[0] in ("train-vsr", "train-units", "train-vocoder", "evaluate"):
        from punarvaani.eval import evaluate
        from punarvaani.training import train_units, train_vocoder, train_vsr

        target = {"train-vsr": train_vsr, "train-units": train_units, "train-vocoder": train_vocoder,
                  "evaluate": evaluate}[argv[0]]  # fmt: skip
        target.main(argv[1:])
        return
    args = ap.parse_args(argv)
    {"demo": _demo, "rebuild-voice": _rebuild_voice, "mine": _mine}[args.cmd](args)


if __name__ == "__main__":
    main()
