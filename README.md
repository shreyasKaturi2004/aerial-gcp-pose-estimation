# Aerial GCP Pose Estimation

Automated Ground Control Point (GCP) detection for aerial surveying. Given a cropped aerial drone image containing a GCP marker, the model simultaneously predicts:

1. The exact **(x, y) pixel coordinates** of the marker's center
2. The **shape class** of the marker — Cross, Square, or L-Shape

---

## Architecture — GCPNet

A pretrained **ResNet18** backbone with two parallel output heads:

| Head | Output | Purpose |
|------|--------|---------|
| Regression | (x, y) normalized coordinates | Keypoint localization |
| Classification | 3-class logits | Shape identification |

**Why ResNet18:** The usable dataset was only 607 images. Larger backbones overfit badly at this scale. ResNet18's ImageNet pretrained features (edges, textures, spatial patterns) transfer well to aerial imagery without requiring large data volumes.

---

## Dataset & EDA Findings

Several discrepancies between the problem statement and actual data were found and corrected during EDA:

| Issue | What the spec said | What the data actually had | How it was handled |
|-------|-------------------|---------------------------|-------------------|
| Label file name | `curated_gcp_marks.json` | `gcp_marks.json` | Used real filename |
| Available images | 1,000 entries | 609 files exist on disk | Filtered to on-disk files only |
| Shape string | `"L-Shaped"` | `"L-Shape"` | Used exact string from JSON |
| Image resolution | 2048×1365 | 4096×2730 or 4096×3068 | Coordinates treated in this larger pixel space |
| Samples with no shape label | Not mentioned | 2 entries with `verified_shape: null` | Dropped — 607 usable samples remain |

**Class distribution (607 samples):**

| Shape | Count | % |
|-------|-------|---|
| Square | 303 | 49.9% |
| L-Shape | 191 | 31.5% |
| Cross | 113 | 18.6% |

Addressed with **class-weighted cross-entropy loss** (inverse frequency, normalized to 3 classes).

**Data leakage prevention:** Each of the 104 unique physical GCP markers was photographed 7–8 times from different angles. A random image-level split would put the same physical marker into both train and val sets. Instead, the split was done **by marker ID**, stratified by shape class — 80% of markers to train (83 markers, 487 images), 20% to val (21 markers, 120 images).

---

## Training Details

**Coordinate representation:** Normalized to `[0, 1]` relative to original image width/height. Scale-independent of any resizing, and trivially converted back to pixel space as `x_px = x_norm * W_orig`.

**Loss function:**
```
total_loss = lambda_reg * SmoothL1(pred_coords, gt_coords) + CrossEntropy(logits, gt_class)
```
- SmoothL1 (Huber loss) for regression — robust to occasional label noise
- Class-weighted CrossEntropy for classification
- `lambda_reg = 50` — needed to give the regression head enough gradient signal relative to the classification head

**Optimizer:** Adam with two learning rate groups — backbone at `lr/10` (1e-5), heads at `lr` (1e-4). This avoids overwriting pretrained features early in training.

**Augmentation:** Horizontal/vertical flips (with coordinate adjustment), color jitter, Gaussian blur. The test set spans 72 project domains vs. only 11 in training — augmentation was deliberately chosen to simulate domain shift.

**Hardware:** RTX 3050 (4.3 GB VRAM), batch size 16, ~27 sec/epoch. Early stopping with patience=8 on combined PCK@50 + macro F1.

---

## Results

### Classification — **Macro F1: 0.852**

The model reliably identifies GCP shape type. Predicted distribution on the 600-image test set:

| Shape | Predicted count | % |
|-------|----------------|---|
| Cross | 252 | 42.0% |
| L-Shape | 154 | 25.7% |
| Square | 194 | 32.3% |

### Localization — **PCK@10/25/50: 0.000**

The regression head did not converge. It predicts near the image center (~x=2050, y=1400 in a 4096×2730 image) for all inputs, which is confirmed visually in the saved overlays in `visualizations/`.

**Root cause:** ResNet18's global average pool collapses all spatial feature maps into a single 512-dimensional vector before the regression head sees them. That vector carries no positional information — it encodes *what* is in the image, not *where*. A GCP marker that shrinks to roughly 10 pixels in the 512×512 resized input leaves essentially no positional signal in a globally-pooled feature. The regression head had nothing to localize from.

**What would fix this:** Replace direct coordinate regression with a **heatmap + soft-argmax** head. Instead of branching off the global pool, branch off `layer2` of ResNet18 (which preserves 64×64 spatial resolution for a 512×512 input). A small conv layer predicts a spatial heatmap; soft-argmax converts the heatmap peak into (x, y) coordinates — differentiable, no spatial information discarded, and each 64×64 cell covers 64 original pixels, giving ~32px precision in the 4096px original space which clears the PCK@50 bar.

### Validation metrics (best checkpoint, epoch 33)

| Metric | Value |
|--------|-------|
| Macro F1 | **0.852** |
| PCK@10 | 0.000 |
| PCK@25 | 0.000 |
| PCK@50 | 0.000 |
| Val loss | 1.8266 |

---

## Engineering Challenges

**1. SSL certificate failure on pretrained weight download**
PyTorch's default weight download failed with `CERTIFICATE_VERIFY_FAILED` on this machine. Resolved by writing a custom download with `ssl.CERT_NONE` to fetch the ResNet18 weights manually.

**2. Silent CPU fallback despite CUDA GPU being present**
Initial PyTorch install was the CPU-only build. Training appeared to run but took ~4 minutes per epoch instead of ~27 seconds, with no error. Diagnosed by checking `torch.cuda.is_available()` inside the training loop (not just at script start), then fixed by reinstalling the CUDA 12.6-matched build. GPU memory usage is now logged at the start of every epoch to make future regressions immediately visible.

**3. Localization failure — diagnosed, not just retried**
After the first full training run showed PCK@50 = 0 across 23 epochs, the response was to inspect *where* the model was predicting (consistently near image center) and trace that to the architectural root cause (global pool destroying spatial information), rather than continuing to tune lambda or extend training. A second run with `lambda_reg=50` was tried to confirm — same result. The fix is architectural.

---

## Repository Structure

```
aerial-gcp-pose-estimation/
├── dataset.py          # Data loading, marker-ID-based splits, augmentation
├── model.py            # GCPNet — ResNet18 + regression + classification heads
├── train.py            # Training loop, loss, PCK metric, checkpointing
├── inference.py        # Batch inference on test set → predictions.json
├── cache_images.py     # One-time pre-resize of training images to 512×512
├── eda/
│   └── eda.py          # EDA script — class distribution, size checks, plots
├── checkpoints/
│   └── best_model.pth  # Best validation checkpoint (see link below)
├── predictions.json    # Test set predictions (600 images)
├── visualizations/     # 10 sample overlays (predicted mark + shape label)
└── requirements.txt
```

---

## Reproducing

```bash
pip install -r requirements.txt

# Optional: rebuild image cache for faster training (512×512 pre-resized)
python cache_images.py --size 512 --workers 8

# Optional: regenerate EDA plots
python eda/eda.py

# Train
python train.py --epochs 40 --batch-size 16 --workers 2

# Run inference on test set
python inference.py --checkpoint checkpoints/best_model.pth --output predictions.json
```

---

## Model Weights

`checkpoints/best_model.pth` is too large for this repository.

**Download:** <!-- ADD GOOGLE DRIVE LINK HERE -->

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
```
