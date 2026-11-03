"""Create a tiny synthetic speaker-folder dataset for plumbing tests."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.synthetic import create_synthetic_dataset  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a synthetic Hindi VSR dataset")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--speakers", type=int, default=3)
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    manifest = create_synthetic_dataset(
        args.out,
        speakers=args.speakers,
        clips=args.clips,
        seconds=args.seconds,
        fps=args.fps,
        size=(args.size, args.size),
        seed=args.seed,
    )
    manifest_path = os.path.join(args.out, "synthetic_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"Wrote {len(manifest['samples'])} clips across "
        f"{len(manifest['speakers'])} speakers to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
