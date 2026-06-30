"""
cache_images.py — Pre-resize all training and test images to the target size.

Resizing a 4096×2730 JPEG on every batch is the primary CPU bottleneck:
PIL must decompress the full-resolution image before resizing. Pre-caching
eliminates that cost — cached images load in ~1/10th the time.

Run once before training:
    python cache_images.py               # default 512×512
    python cache_images.py --size 640    # if you change IMG_SIZE in dataset.py

Outputs:
    cache/train/<original_relative_path>   — resized training images
    cache/test/<original_relative_path>    — resized test images
    cache/train_sizes.json                 — {rel_path: [W_orig, H_orig]}
    cache/test_sizes.json                  — same for test

train_sizes.json lets the dataset know original pixel dimensions for correct
coordinate normalisation without having to re-open the original files.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT      = Path(__file__).parent
TRAIN_DIR = ROOT / "train_dataset"
TEST_DIR  = ROOT / "test_dataset"
CACHE_DIR = ROOT / "cache"


def process_image(src: Path, dst: Path, size: int) -> tuple:
    """Resize src → dst and return (rel_path_str, orig_W, orig_H)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        W, H = img.size
        if not dst.exists():
            img.convert("RGB").resize((size, size), Image.BILINEAR).save(
                dst, format="JPEG", quality=92
            )
    return str(src), W, H


def cache_split(src_root: Path, dst_root: Path, sizes_path: Path, size: int, workers: int):
    jobs = [
        (Path(root) / fname, dst_root / Path(root).relative_to(src_root) / fname)
        for root, _, files in os.walk(src_root)
        for fname in files
        if fname.lower().endswith((".jpg", ".jpeg"))
    ]
    print(f"  {len(jobs)} images -> {dst_root}")

    sizes = {}
    done  = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, s, d, size): (s, d) for s, d in jobs}
        for fut in as_completed(futures):
            src_str, W, H = fut.result()
            rel = str(Path(src_str).relative_to(src_root)).replace("\\", "/")
            sizes[rel] = [W, H]
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] cached")

    with open(sizes_path, "w") as f:
        json.dump(sizes, f)
    print(f"  Sizes saved to {sizes_path}")


def main(args):
    print(f"Target size : {args.size}×{args.size}")
    print(f"Threads     : {args.workers}")
    print()

    print("=== Training images ===")
    cache_split(TRAIN_DIR, CACHE_DIR / "train",
                CACHE_DIR / "train_sizes.json", args.size, args.workers)

    print("\n=== Test images ===")
    cache_split(TEST_DIR, CACHE_DIR / "test",
                CACHE_DIR / "test_sizes.json", args.size, args.workers)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",    type=int, default=512)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel resize threads")
    args = parser.parse_args()
    main(args)
