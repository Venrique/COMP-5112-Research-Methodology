"""Apply the project's optional preprocessing steps to one image.

Example (Git Bash):
    ./.venv/Scripts/python.exe run_preprocessing.py input.png output.png \
        --steps clahe denoising deskew
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from src.preprocessing import CLAHE, Deskew, NonLocalMeans, PreprocessingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to the input image")
    parser.add_argument("output", type=Path, help="path for the processed image")
    parser.add_argument(
        "--steps", nargs="*", choices=("clahe", "denoising", "deskew"),
        default=("clahe", "denoising", "deskew"),
        help="enabled steps in execution order (default: all three); use --steps alone for baseline",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    step_types = {"clahe": CLAHE, "denoising": NonLocalMeans, "deskew": Deskew}
    pipeline = PreprocessingPipeline(tuple(step_types[name]() for name in args.steps))

    with Image.open(args.input) as source:
        image = source.copy()
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    result = pipeline(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved processed image to: {args.output}")
    print(json.dumps(pipeline.to_config(), indent=2))


if __name__ == "__main__":
    main()
