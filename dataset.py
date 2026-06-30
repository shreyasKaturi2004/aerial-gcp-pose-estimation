"""
dataset.py — Data loading and train/val splitting for GCP pose estimation.

Key design choices:
- Filters the 1000-entry JSON down to the 607 usable samples (on-disk, valid label).
- Splits by GCP marker ID (3rd path component) to prevent leakage: the same
  physical marker appears in 7-8 images, so a random image-level split would
  put the same marker in both train and val.
- Coordinates are normalised by the original image dimensions (x/W, y/H) so
  they sit in [0,1] regardless of the resize used for the model input.
- Supports an optional image cache (produced by cache_images.py): pre-resized
  images at IMG_SIZE avoid repeated 4096px JPEG decompression — ~10x faster
  per batch on CPU.
- Augmentation applies colour jitter + flips. Flips require symmetric coord
  adjustment; rotation is omitted because it needs trig + clipping overhead.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
TRAIN_DIR = ROOT / "train_dataset"
JSON_PATH = TRAIN_DIR / "gcp_marks.json"
CACHE_DIR = ROOT / "cache"

CLASS_NAMES = ["Cross", "L-Shape", "Square"]   # alphabetical — fixed index mapping
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

IMG_SIZE = 512   # both width and height after resize


# ── Label loading ─────────────────────────────────────────────────────────────
def load_labels(
    json_path: Path = JSON_PATH,
    train_dir: Path = TRAIN_DIR,
) -> Dict[str, dict]:
    """
    Load gcp_marks.json and return only entries that are:
      1. Present on disk.
      2. Have a recognised verified_shape (drops the 4 None entries).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    valid = {}
    for rel_path, entry in raw.items():
        if entry.get("verified_shape") not in CLASS_TO_IDX:
            continue
        if not (train_dir / rel_path).exists():
            continue
        valid[rel_path] = entry
    return valid


def load_orig_sizes(sizes_json: Path) -> Optional[Dict[str, List[int]]]:
    """Load the pre-computed {rel_path: [W, H]} map produced by cache_images.py."""
    if sizes_json.exists():
        with open(sizes_json, "r") as f:
            return json.load(f)
    return None


# ── Train/val split ───────────────────────────────────────────────────────────
def _marker_key(rel_path: str) -> str:
    """Return 'project/survey/gcp_id' — the physical marker identity."""
    return "/".join(rel_path.replace("\\", "/").split("/")[:3])


def make_splits(
    labels: Dict[str, dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split by GCP marker ID, stratified by shape class.

    Stratification: for each shape class, randomly assign val_frac of that
    class's markers to val and the rest to train. This keeps class proportions
    similar across splits even with only ~104 unique markers.
    """
    rng = random.Random(seed)

    marker_paths: Dict[str, List[str]] = defaultdict(list)
    for p in labels:
        marker_paths[_marker_key(p)].append(p)

    # Infer each marker's shape (majority vote across its views)
    marker_shape: Dict[str, str] = {}
    for mk, paths in marker_paths.items():
        votes = [labels[p]["verified_shape"] for p in paths]
        marker_shape[mk] = max(set(votes), key=votes.count)

    shape_markers: Dict[str, List[str]] = defaultdict(list)
    for mk, shape in marker_shape.items():
        shape_markers[shape].append(mk)

    train_set, val_set = set(), set()
    for shape, markers in shape_markers.items():
        shuffled = markers[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_frac))
        val_set.update(shuffled[:n_val])
        train_set.update(shuffled[n_val:])

    train_paths = [p for p in labels if _marker_key(p) in train_set]
    val_paths   = [p for p in labels if _marker_key(p) in val_set]
    return train_paths, val_paths


def split_summary(labels: Dict, train_paths: List[str], val_paths: List[str]) -> None:
    """Print a quick sanity-check table after splitting."""
    for name, paths in [("train", train_paths), ("val", val_paths)]:
        counts  = Counter(labels[p]["verified_shape"] for p in paths)
        markers = len({_marker_key(p) for p in paths})
        print(f"  {name:5s}: {len(paths):3d} images | {markers:2d} markers | "
              + " | ".join(f"{c}={counts[c]}" for c in CLASS_NAMES))


# ── Dataset ───────────────────────────────────────────────────────────────────
class GCPDataset(Dataset):
    """
    Yields (image_tensor, coords_norm, class_idx, orig_wh) per sample.

      image_tensor : (3, IMG_SIZE, IMG_SIZE) float, ImageNet-normalised
      coords_norm  : (2,) float  — [x/W_orig, y/H_orig] in [0, 1]
      class_idx    : ()   long   — index into CLASS_NAMES
      orig_wh      : (2,) float  — [W_orig, H_orig] for PCK in pixel space

    If cache_dir / "train" exists (built by cache_images.py), images are loaded
    from there (already resized) which is ~10x faster than loading raw 4096px
    JPEGs. Original dimensions come from cache/train_sizes.json so coordinate
    normalisation is still exact.
    """

    _color_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    ])
    _to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        paths: List[str],
        labels: Dict[str, dict],
        train_dir: Path = TRAIN_DIR,
        img_size: int = IMG_SIZE,
        augment: bool = False,
        cache_dir: Optional[Path] = CACHE_DIR,
    ):
        self.paths     = paths
        self.labels    = labels
        self.train_dir = train_dir
        self.img_size  = img_size
        self.augment   = augment

        # Check for pre-resized image cache
        train_cache = cache_dir / "train" if cache_dir else None
        if train_cache and train_cache.exists():
            self._img_dir = train_cache
            sizes_json    = cache_dir / "train_sizes.json"
            self._orig_sizes = load_orig_sizes(sizes_json) or {}
            self._cached  = True
        else:
            self._img_dir    = train_dir
            self._orig_sizes = {}
            self._cached     = False

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        entry    = self.labels[rel_path]

        # Normalised path key for sizes dict (forward slashes, as written by cache_images.py)
        size_key = rel_path.replace("\\", "/")

        if self._cached and size_key in self._orig_sizes:
            # Fast path: load pre-resized image; get original dims from JSON
            W, H = self._orig_sizes[size_key]
            img  = Image.open(self._img_dir / rel_path).convert("RGB")
        else:
            # Slow path: open full-res original, resize manually
            img  = Image.open(self.train_dir / rel_path).convert("RGB")
            W, H = img.size
            img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Normalise coordinates using original resolution — independent of resize
        x_norm = entry["mark"]["x"] / W
        y_norm = entry["mark"]["y"] / H

        # Geometric augmentation: flips are bijective on coords, no clipping needed
        if self.augment:
            if random.random() < 0.5:
                img    = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_norm = 1.0 - x_norm
            if random.random() < 0.5:
                img    = img.transpose(Image.FLIP_TOP_BOTTOM)
                y_norm = 1.0 - y_norm

        if self.augment:
            img = self._color_aug(img)

        img_tensor  = self._to_tensor(img)
        coords_norm = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        class_idx   = torch.tensor(CLASS_TO_IDX[entry["verified_shape"]], dtype=torch.long)
        orig_wh     = torch.tensor([W, H], dtype=torch.float32)

        return img_tensor, coords_norm, class_idx, orig_wh
