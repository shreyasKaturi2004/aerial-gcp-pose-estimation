"""
inference.py — Run GCPNet on all test images and produce predictions.json.

Output format matches gcp_marks.json exactly:
  {
    "relative/path/image.JPG": {
      "mark": {"x": <float px>, "y": <float px>},
      "verified_shape": "<Cross|L-Shape|Square>"
    },
    ...
  }

Coordinates are in original image pixel space (before any resize).

Usage:
  python inference.py
  python inference.py --checkpoint checkpoints/best_model.pth
                      --test-dir test_dataset
                      --output predictions.json
                      --vis-dir visualizations
                      --n-vis 10
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from dataset import CLASS_NAMES, IMG_SIZE
from model import GCPNet

# ── Default paths ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
CKPT_PATH = ROOT / "checkpoints" / "best_model.pth"
TEST_DIR  = ROOT / "test_dataset"
OUT_JSON  = ROOT / "predictions.json"
VIS_DIR   = ROOT / "visualizations"

# ImageNet normalisation — must match training
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_images(root: Path) -> list[Path]:
    """Return all JPG/jpeg images under root, sorted."""
    imgs = sorted(root.rglob("*.JPG")) + sorted(root.rglob("*.jpg")) + \
           sorted(root.rglob("*.jpeg")) + sorted(root.rglob("*.JPEG"))
    # deduplicate (rglob can overlap on case-insensitive FS)
    seen, out = set(), []
    for p in imgs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_model(ckpt_path: Path) -> GCPNet:
    model = GCPNet(pretrained=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    m     = ckpt.get("val_metrics", {})
    print(f"Loaded checkpoint  epoch={epoch}  "
          f"val_loss={ckpt.get('val_loss', 0):.4f}  "
          f"F1={m.get('macro_f1', 0):.3f}")
    return model


def predict_batch(model: GCPNet, img_paths: list[Path], test_root: Path):
    """Yield (rel_path_str, x_px, y_px, shape_str) for each image."""
    model.eval()
    with torch.no_grad():
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size          # original pixel dimensions
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

            coords, logits = model(tensor)
            x_norm, y_norm = coords[0].tolist()
            cls_idx        = logits[0].argmax().item()

            x_px   = x_norm * W
            y_px   = y_norm * H
            shape  = CLASS_NAMES[cls_idx]

            # Key: path relative to test_root with forward slashes
            rel = img_path.relative_to(test_root).as_posix()
            yield rel, x_px, y_px, shape, W, H, img


def save_visualization(img: Image.Image, x_px: float, y_px: float,
                       shape: str, rel: str, out_path: Path) -> None:
    """Draw predicted mark and label on a 512-px thumbnail."""
    thumb = img.resize((512, 512), Image.LANCZOS)
    W_orig, H_orig = img.size
    sx = 512 / W_orig
    sy = 512 / H_orig
    tx = x_px * sx
    ty = y_px * sy

    draw = ImageDraw.Draw(thumb)
    r = 8
    draw.ellipse([tx - r, ty - r, tx + r, ty + r], outline="red", width=3)
    draw.line([tx - 20, ty, tx + 20, ty], fill="red", width=2)
    draw.line([tx, ty - 20, tx, ty + 20], fill="red", width=2)
    draw.text((tx + r + 3, ty - r), shape, fill="yellow")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(out_path, quality=90)


def main(args):
    test_root = Path(args.test_dir)
    ckpt_path = Path(args.checkpoint)
    out_json  = Path(args.output)
    vis_dir   = Path(args.vis_dir)

    print(f"Device         : {DEVICE}")
    print(f"Test directory : {test_root}")
    print(f"Checkpoint     : {ckpt_path}")

    model     = load_model(ckpt_path)
    img_paths = find_images(test_root)
    print(f"Found {len(img_paths)} test images\n")

    predictions = {}
    vis_interval = max(1, len(img_paths) // args.n_vis)
    vis_count = 0

    for i, (rel, x_px, y_px, shape, W, H, img) in enumerate(
            predict_batch(model, img_paths, test_root)):
        predictions[rel] = {
            "mark": {"x": x_px, "y": y_px},
            "verified_shape": shape,
        }

        # Save visualizations at regular intervals
        if vis_count < args.n_vis and i % vis_interval == 0:
            safe_name = rel.replace("/", "_").replace("\\", "_")
            vis_path  = vis_dir / f"{vis_count:02d}_{safe_name}"
            save_visualization(img, x_px, y_px, shape, rel, vis_path)
            vis_count += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(img_paths):
            print(f"  {i+1}/{len(img_paths)} done")

    # Write JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved {len(predictions)} predictions -> {out_json}")
    print(f"Saved {vis_count} visualizations   -> {vis_dir}/")

    # Shape distribution summary
    from collections import Counter
    counts = Counter(v["verified_shape"] for v in predictions.values())
    print("\nPredicted shape distribution:")
    for shape, cnt in sorted(counts.items()):
        print(f"  {shape:10s}: {cnt:4d}  ({100*cnt/len(predictions):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT_PATH))
    parser.add_argument("--test-dir",   default=str(TEST_DIR))
    parser.add_argument("--output",     default=str(OUT_JSON))
    parser.add_argument("--vis-dir",    default=str(VIS_DIR))
    parser.add_argument("--n-vis",      type=int, default=10)
    main(parser.parse_args())
