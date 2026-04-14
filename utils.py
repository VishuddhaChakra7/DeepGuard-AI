import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_logging(log_file: str = "deepguard.log") -> logging.Logger:
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Seeds all random number generators for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # NOTE: benchmark=True speeds up training when input sizes are fixed.
        # Set to False only if input sizes vary across batches.
        torch.backends.cudnn.benchmark = True


class LabelSmoothCE(nn.Module):
    """Cross-Entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, targets)
        smooth_loss = -log_probs.sum(dim=-1).mean() / n_classes
        return (1 - self.smoothing) * nll + self.smoothing * smooth_loss
