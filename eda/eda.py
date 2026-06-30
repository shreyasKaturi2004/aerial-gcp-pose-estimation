"""
Exploratory Data Analysis — Aerial GCP Pose Estimation
Run from project root: python eda/eda.py
Outputs: eda/plots/eda_samples.png, eda_scatter.png, eda_class_dist.png
"""

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "train_dataset"
TEST_DIR  = ROOT / "test_dataset"
JSON_PATH = TRAIN_DIR / "gcp_marks.json"
PLOT_DIR  = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

VALID_SHAPES = {"Cross", "Square", "L-Shape"}

# ── 1. Load labels ────────────────────────────────────────────────────────────
with open(JSON_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

found_paths  = [p for p in labels if (TRAIN_DIR / p).exists()]
missing_paths = [p for p in labels if not (TRAIN_DIR / p).exists()]

print(f"Total labelled entries in JSON : {len(labels)}")
print(f"Images found on disk           : {len(found_paths)}")
print(f"JSON entries missing from disk : {len(missing_paths)}")

# ── 2. All unique shape values ────────────────────────────────────────────────
print("\n=== verified_shape values (full JSON) ===")
all_shape_counts = Counter(v.get("verified_shape") for v in labels.values())
for val, cnt in sorted(all_shape_counts.items(), key=lambda x: (x[0] is None, x[0])):
    print(f"  {str(val):15s}: {cnt:4d}  ({100*cnt/len(labels):.1f}%)")

# ── 3. Class distribution (on-disk only) ─────────────────────────────────────
print("\n=== Shape distribution — usable (on-disk) samples ===")
shape_counts_disk = Counter(labels[p].get("verified_shape", None) for p in found_paths)
for val, cnt in sorted(shape_counts_disk.items(), key=lambda x: (x[0] is None, x[0])):
    print(f"  {str(val):15s}: {cnt:4d}  ({100*cnt/len(found_paths):.1f}%)")

# ── 4. Malformed entries ──────────────────────────────────────────────────────
missing_mark  = [p for p, e in labels.items() if "mark" not in e or e["mark"] is None]
missing_shape = [p for p, e in labels.items() if e.get("verified_shape") is None]
bad_shape     = [(p, e["verified_shape"]) for p, e in labels.items()
                 if e.get("verified_shape") is not None and e["verified_shape"] not in VALID_SHAPES]

print(f"\n=== Malformed entries ===")
print(f"  Missing 'mark'           : {len(missing_mark)}")
print(f"  Missing 'verified_shape' : {len(missing_shape)}")
print(f"  Unrecognised shape value : {len(bad_shape)}")
for p, s in bad_shape[:5]:
    print(f"    {p!r}  ->  {s!r}")

print(f"\n=== Entries missing verified_shape (detail) ===")
for path in missing_shape:
    on_disk = (TRAIN_DIR / path).exists()
    print(f"  {'[disk]' if on_disk else '[miss]'}  {path}")
    print(f"           mark: {labels[path].get('mark')}")

# ── 5. Case-insensitive duplicate paths ───────────────────────────────────────
lower_counts = Counter(p.lower() for p in labels)
case_dupes = {k: v for k, v in lower_counts.items() if v > 1}
print(f"\nCase-insensitive duplicate paths: {len(case_dupes)}")

# ── 6. Coordinate statistics ──────────────────────────────────────────────────
xs = np.array([labels[p]["mark"]["x"] for p in found_paths])
ys = np.array([labels[p]["mark"]["y"] for p in found_paths])
print(f"\n=== Coordinate stats (on-disk, n={len(xs)}) ===")
print(f"  x: min={xs.min():.1f}  max={xs.max():.1f}  mean={xs.mean():.1f}  std={xs.std():.1f}")
print(f"  y: min={ys.min():.1f}  max={ys.max():.1f}  mean={ys.mean():.1f}  std={ys.std():.1f}")

# ── 7. Missing file breakdown by project ─────────────────────────────────────
print(f"\n=== Missing files by project ({len(missing_paths)} total) ===")
for proj, cnt in Counter(p.split("/")[0] for p in missing_paths).most_common():
    print(f"  {proj:45s}: {cnt}")

# ── 8. Project sample counts (JSON) ──────────────────────────────────────────
print("\n=== Samples per project (train JSON) ===")
for proj, cnt in Counter(p.split("/")[0] for p in labels).most_common():
    print(f"  {proj:45s}: {cnt}")

# ── 9. Views per physical GCP marker ─────────────────────────────────────────
gcp_counter = Counter(
    "/".join(p.replace("\\", "/").split("/")[:3])
    for p in found_paths
    if len(p.replace("\\", "/").split("/")) >= 3
)
view_dist = Counter(gcp_counter.values())
print(f"\n=== Views per GCP marker (on-disk, {len(gcp_counter)} unique markers) ===")
for views, cnt in sorted(view_dist.items()):
    print(f"  {views:2d} view(s): {cnt} markers")

# ── 10. Image size + out-of-bounds scan (full) ───────────────────────────────
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("\nPillow not installed — skipping image checks")

if PIL_AVAILABLE:
    print(f"\n=== Image size + out-of-bounds scan (all {len(found_paths)} on-disk files) ===")
    size_all = Counter()
    oob, corrupted = [], []
    for rel_path in found_paths:
        try:
            with PILImage.open(TRAIN_DIR / rel_path) as img:
                w, h = img.size
                size_all[(w, h)] += 1
                mx = labels[rel_path]["mark"]["x"]
                my = labels[rel_path]["mark"]["y"]
                if not (0 <= mx < w and 0 <= my < h):
                    oob.append((rel_path, mx, my, w, h))
        except Exception as e:
            corrupted.append((rel_path, str(e)))

    for (w, h), cnt in sorted(size_all.items(), key=lambda x: -x[1]):
        print(f"  {w}x{h}: {cnt} images")
    print(f"  Corrupted / unreadable  : {len(corrupted)}")
    print(f"  Out-of-bounds coords    : {len(oob)}")
    for p, mx, my, w, h in oob[:5]:
        print(f"    {p}  mark=({mx:.1f},{my:.1f})  img=({w},{h})")

# ── 11. Test dataset summary ──────────────────────────────────────────────────
test_imgs = [
    Path(root) / fname
    for root, _, files in os.walk(TEST_DIR)
    for fname in files
    if fname.lower().endswith((".jpg", ".jpeg"))
]
print(f"\n=== Test dataset ===")
print(f"  Total images: {len(test_imgs)}")
for proj, cnt in Counter(p.relative_to(TEST_DIR).parts[0] for p in test_imgs).most_common(10):
    print(f"  {proj:45s}: {cnt}")
print(f"  ... ({len(Counter(p.relative_to(TEST_DIR).parts[0] for p in test_imgs))} project folders total)")

# ── 12. Visualisations ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("\nmatplotlib not installed — skipping plots")

if PIL_AVAILABLE and MPL_AVAILABLE:
    random.seed(42)
    by_shape = defaultdict(list)
    for p in found_paths:
        by_shape[labels[p].get("verified_shape", "None")].append(p)

    SHAPES   = ["Cross", "Square", "L-Shape"]
    COLORS   = {"Cross": "tab:blue", "Square": "tab:orange", "L-Shape": "tab:green", "None": "red"}
    CROP_HALF = 200

    # ── Sample crops ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(SHAPES), 3, figsize=(18, len(SHAPES) * 6))
    fig.suptitle("Sample crops (400×400 px) centred on GCP mark", fontsize=14, y=1.01)
    for row, shape in enumerate(SHAPES):
        samples = random.sample(by_shape[shape], min(3, len(by_shape[shape])))
        for col, rel_path in enumerate(samples):
            ax = axes[row][col]
            entry = labels[rel_path]
            mx, my = entry["mark"]["x"], entry["mark"]["y"]
            with PILImage.open(TRAIN_DIR / rel_path) as img:
                W, H = img.size
                x0, y0 = max(0, int(mx) - CROP_HALF), max(0, int(my) - CROP_HALF)
                x1, y1 = min(W, int(mx) + CROP_HALF), min(H, int(my) + CROP_HALF)
                crop = img.crop((x0, y0, x1, y1))
            ax.imshow(np.array(crop))
            ax.plot(mx - x0, my - y0, "r+", markersize=18, markeredgewidth=2)
            ax.set_title(f"{shape}\n{Path(rel_path).name}\nmark=({mx:.0f},{my:.0f})", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "eda_samples.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {PLOT_DIR / 'eda_samples.png'}")

    # ── Coordinate scatter ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for shape in SHAPES:
        pts = [(labels[p]["mark"]["x"], labels[p]["mark"]["y"])
               for p in found_paths if labels[p].get("verified_shape") == shape]
        if pts:
            sx, sy = zip(*pts)
            ax2.scatter(sx, sy, c=COLORS[shape], label=f"{shape} (n={len(pts)})", alpha=0.5, s=15)
    ax2.axvline(4096, color="k",      ls="--",  lw=0.8, label="img width=4096")
    ax2.axhline(2730, color="purple", ls=":",   lw=0.8, label="img height=2730")
    ax2.axhline(3068, color="purple", ls="-.",  lw=0.8, label="img height=3068")
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")
    ax2.set_title("GCP mark coordinates (train, on-disk)")
    ax2.legend()
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "eda_scatter.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_DIR / 'eda_scatter.png'}")

    # ── Class distribution bar ────────────────────────────────────────────────
    counts = Counter(labels[p].get("verified_shape", "None") for p in found_paths)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bars = ax3.bar(list(counts.keys()), list(counts.values()),
                   color=[COLORS.get(k, "gray") for k in counts.keys()])
    for bar, val in zip(bars, counts.values()):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(val), ha="center", va="bottom", fontsize=10)
    ax3.set_title(f"Class distribution — on-disk train set (n={len(found_paths)})")
    ax3.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "eda_class_dist.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_DIR / 'eda_class_dist.png'}")
