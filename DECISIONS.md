# Design Decision Log

Key technical decisions made during development, with the reasoning behind each.

---

## 1. Architecture — ResNet18 backbone

**Decision:** Use a pretrained ResNet18 as the shared backbone, rather than a larger model (ResNet50, EfficientNet, ViT) or training from scratch.

**Reasoning:** The usable dataset after filtering was 607 images — far too small to train a large model without heavy overfitting. ResNet18's ImageNet pretrained weights give strong low/mid-level features (edges, textures, spatial gradients) that transfer well to aerial imagery, without requiring the volume of data a deeper model would need. A smaller backbone also means faster iteration during experiments, which mattered given the limited compute budget.

---

## 2. Two-head output structure

**Decision:** Branch two separate heads off the shared backbone — one regression head for (x, y) coordinates, one classification head for shape.

**Reasoning:** Localization and shape recognition are related but not identical tasks. Letting each head specialize avoids conflicting gradient directions that would arise if both tasks competed through a single shared linear layer. The backbone learns features useful to both; the heads learn task-specific mappings independently.

---

## 3. Loss function — SmoothL1 + class-weighted CrossEntropy

**Decision:** Use Smooth L1 (Huber) loss for coordinate regression, and class-weighted cross-entropy for shape classification.

**Reasoning:**
- **Smooth L1 over MSE:** More robust to occasional label noise in the coordinate annotations. MSE penalizes large errors quadratically, which amplifies the effect of mislabeled or ambiguous marks. Smooth L1 is linear for large errors, MSE-like for small ones.
- **Class-weighted CE:** The dataset is imbalanced — Square ~50%, L-Shape ~31%, Cross ~18.5%. Without weighting, the model would learn to predict Square for everything and achieve ~50% accuracy while ignoring minority classes. Inverse-frequency weights (normalized to 3 classes) give Cross 1.82×, L-Shape 1.08×, Square 0.66× weight, forcing the model to treat each class as equally important.

---

## 4. Lambda_reg — balancing regression and classification gradient scale

**Decision:** Weight the regression loss by `lambda_reg` before adding to classification loss. Tried `lambda_reg=10` first, then `lambda_reg=50`.

**Reasoning:** At initialization, SmoothL1 on normalized [0, 1] coordinates produces losses in the range 0.05–0.1, while cross-entropy starts around 1.1 (log of 3 classes). Without scaling, the classification gradient dominates and the regression head receives almost no useful signal. `lambda_reg=10` was the first attempt to rebalance; it was increased to 50 after the first run showed the regression head still failed to converge (PCK@50 = 0 across 23 epochs). Neither value resolved the localization failure, which was ultimately diagnosed as architectural rather than a hyperparameter issue (see Decision 8).

---

## 5. Optimizer and learning rate strategy — two-group Adam

**Decision:** Use Adam with two learning rate groups: backbone at `lr / 10` (1e-5), both heads at `lr` (1e-4).

**Reasoning:** The backbone starts with pretrained ImageNet weights — these features are already useful and should be fine-tuned gently, not overwritten. Setting the backbone LR to 1/10th of the head LR lets the heads learn quickly from scratch while the backbone adapts slowly, preserving the pretrained representations early in training. A single global LR would either train the heads too slowly or corrupt the backbone too aggressively.

---

## 6. Train/val split — by physical marker ID, stratified by shape

**Decision:** Split by unique physical GCP marker (104 total), not by individual image. 80% of markers → train (83 markers, 487 images), 20% → val (21 markers, 120 images). Stratified within each shape class.

**Reasoning:** Each physical marker was photographed 7–8 times from slightly different angles. A random image-level split would put different views of the same physical marker into both train and val, creating severe data leakage — the model would be validated on near-duplicates of its training data and appear to generalize when it isn't. Splitting by marker ID ensures no physical GCP appears in both sets. Stratification ensures the class distribution is preserved in both splits despite the small number of markers.

---

## 7. Augmentation choices

**Decision:** Apply horizontal flips, vertical flips (with coordinate adjustment), color jitter, and Gaussian blur during training.

**Reasoning:** The test set spans 72 different project/survey domains versus only 11 in the training set — significant domain shift is expected. The augmentations were chosen specifically to simulate this:
- **Flips:** GCP markers have no canonical orientation from an aerial view; the model should be invariant to which direction the drone was flying. Coordinates are adjusted accordingly (x_new = 1 - x for H-flip, y_new = 1 - y for V-flip).
- **Color jitter:** Aerial imagery varies widely in color temperature, saturation, and brightness depending on time of day, season, and sensor settings across different surveys.
- **Gaussian blur:** Simulates varying image sharpness and altitude-induced defocus across different drone platforms and flight altitudes.

---

## 8. Localization failure — diagnosis and proposed fix

**Decision (diagnosis):** The regression head failed to converge (PCK@10/25/50 = 0 across 43 epochs, two separate runs) because the architecture is structurally unable to perform spatial localization.

**Reasoning:** ResNet18's global average pool collapses all spatial feature maps into a single 512-dimensional vector before the regression head sees any data. That vector encodes *what* is present globally in the image, but contains no information about *where* it is spatially. A GCP marker shrinks to roughly 10 pixels in the 512×512 resized input, leaving essentially no positional signal in a globally-pooled feature. The regression head had nothing to localize from, regardless of lambda_reg or number of epochs.

**Proposed fix:** Replace direct coordinate regression with a **heatmap + soft-argmax** head:

1. Branch the regression head off `layer2` of ResNet18 (before global pooling), which preserves 64×64 spatial resolution for a 512×512 input.
2. A small convolutional layer predicts a spatial heatmap over those 64×64 positions.
3. Soft-argmax converts the heatmap to (x, y) coordinates — differentiable end-to-end, no spatial information discarded.
4. Each 64×64 cell covers 64 original pixels; soft-argmax interpolation between cells gives ~32px precision in the 4096px original space, which clears the PCK@50 threshold.

The classification head would remain unchanged, still using the globally-pooled features from `layer4`.
