from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a cup dataset from human-reviewed YOLO labels")
    parser.add_argument("--reviewed", type=Path, default=Path("data/collection/reviewed_labels"))
    parser.add_argument("--images", type=Path, default=Path("data/collection/samples/images"))
    parser.add_argument("--output", type=Path, default=Path("data/datasets/cup"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    labels = [p for p in args.reviewed.glob("*.txt") if p.read_text(encoding="utf-8").strip()]
    random.Random(args.seed).shuffle(labels)
    cut = round(len(labels) * (1 - args.val_ratio))
    for split, items in (("train", labels[:cut]), ("val", labels[cut:])):
        for label in items:
            image = next((args.images / f"{label.stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (args.images / f"{label.stem}{ext}").exists()), None)
            if image is None:
                raise FileNotFoundError(f"image missing for {label.name}")
            (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
            (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, args.output / "labels" / split / label.name)
            shutil.copy2(image, args.output / "images" / split / image.name)
    (args.output / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [cup]\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
