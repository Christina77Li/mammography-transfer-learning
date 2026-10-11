"""Shared epoch execution with the same criterion for training and validation."""
from __future__ import annotations

from src.training.evaluate import evaluate
from src.training.gradual_unfreeze import batchnorm_status
from src.training.losses import build_criterion
from src.training.metrics import checked_metrics
from src.training.train_epoch import train_one_epoch
from src.utils.config import bn_policy


class Trainer:
    """Own loaders, criterion and training policy for one effective phase."""

    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.freeze_all_bn, _ = bn_policy(config)
        self.criterion, self.loss_meta = build_criterion(
            config.get("loss", {"type": "bce"}), device, train_loader=train_loader
        )
        clipping = config.get("training", {}).get("gradient_clipping", {})
        self.grad_clip_max_norm = (
            float(clipping.get("max_norm", 1.0)) if clipping.get("enabled", False) else None
        )

    def run_epoch(self, optimizer, frozen_bn_blocks=None, label=""):
        train_metrics = train_one_epoch(
            self.model, self.train_loader, self.criterion, optimizer, self.device,
            frozen_bn_blocks=frozen_bn_blocks or [],
            freeze_all_bn=self.freeze_all_bn,
            grad_clip_max_norm=self.grad_clip_max_norm,
        )
        if self.freeze_all_bn:
            print(f"[batchnorm] {label} {batchnorm_status(self.model)}")
        val_metrics = checked_metrics(
            evaluate(self.model, self.val_loader, self.criterion, self.device)
        )
        return train_metrics, val_metrics
