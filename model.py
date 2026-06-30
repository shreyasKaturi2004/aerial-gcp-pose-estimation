"""
model.py — GCPNet architecture for joint keypoint localisation + shape classification.

Architecture: ResNet18 backbone → shared feature vector → two parallel heads.

Why ResNet18:
  - Only 607 training images. Larger backbones (ResNet50, ViT) overfit badly at
    this scale without heavy regularisation.
  - ImageNet pretraining gives strong low/mid-level features (edges, textures)
    that transfer well to aerial imagery.
  - 512-dim feature vector is large enough for two simple MLP heads.

Why two separate heads (not one shared MLP):
  - Localisation and shape recognition are related but not identical tasks.
    Letting each head specialise avoids conflicting gradient directions on the
    shared linear layer.

Regression head output:
  - Sigmoid activation → guaranteed [0, 1] output, matching the normalised
    coordinate targets. Avoids having to clamp or clip predictions at inference.

Classification head output:
  - Raw logits (no softmax). CrossEntropyLoss applies log-softmax internally.
"""

import torch
import torch.nn as nn
from torchvision import models

from dataset import CLASS_NAMES


class GCPNet(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES), pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace the original 1000-class head with an identity so we get the
        # raw 512-d feature vector from the global average pool.
        in_features: int = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ── Regression head ────────────────────────────────────────────────
        # Two-layer MLP ending in Sigmoid so predictions live in [0, 1].
        # Dropout(0.3) acts as a lightweight regulariser for the small dataset.
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

        # ── Classification head ────────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            # no activation — CrossEntropyLoss expects raw logits
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) image batch, ImageNet-normalised.
        Returns:
            coords: (B, 2) normalised coordinates [x_norm, y_norm] in [0, 1].
            logits: (B, num_classes) raw class scores.
        """
        features = self.backbone(x)   # (B, 512)
        coords   = self.reg_head(features)
        logits   = self.cls_head(features)
        return coords, logits
