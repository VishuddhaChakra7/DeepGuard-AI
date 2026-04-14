import torch
import torch.nn as nn
import timm


class DeepGuardNet(nn.Module):
    """
    Deepfake detection model built on a timm backbone.

    Default backbone: efficientnet_b0 (fast, ~5.3 M params).
    Switch to efficientnet_b4 for higher accuracy at the cost of speed/memory.

    Args:
        num_classes:   Number of output classes (2 = Real / Fake).
        dropout:       Dropout rate for the classification head.
        pretrained:    Load ImageNet pre-trained weights.
        backbone_name: Any timm model name with `num_classes=0` support.
        freeze_ratio:  Fraction of backbone params to freeze initially (0–1).
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.4,
        pretrained: bool = True,
        backbone_name: str = "efficientnet_b0",
        freeze_ratio: float = 0.7,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat = self.backbone.num_features

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.75),
            nn.Linear(512, num_classes),
        )

        self._freeze(freeze_ratio)

    # ── Freeze helpers ────────────────────────────────────────────────────
    def _freeze(self, ratio: float) -> None:
        """Freeze the first `ratio` fraction of backbone parameters."""
        params = list(self.backbone.parameters())
        cutoff = int(len(params) * ratio)
        for i, p in enumerate(params):
            p.requires_grad = i >= cutoff

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    # ── Param summary ─────────────────────────────────────────────────────
    def param_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct       = 100.0 * trainable / total
        return (
            f"Backbone : {self.backbone_name}\n"
            f"Total    : {total / 1e6:.2f} M\n"
            f"Trainable: {trainable / 1e6:.2f} M  ({pct:.1f}%)"
        )

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
