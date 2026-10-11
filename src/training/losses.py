from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from src.data.dataset import labels_from_dataset


def labels_from_loader(loader) -> np.ndarray:
    return labels_from_dataset(loader.dataset)


def compute_prevalence_and_pos_weight(labels, device, sqrt_weight: bool = True, eps: float = 1e-6):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        raise ValueError("Cannot compute pos_weight from empty labels.")
    positives = max(int(labels.sum()), 0)
    negatives = max(int(labels.size - positives), 0)
    if positives == 0 or negatives == 0:
        raise ValueError("pos_weight requires both training classes.")
    prevalence = float(labels.mean())
    ratio = negatives / max(positives, eps)
    value = math.sqrt(ratio) if sqrt_weight else ratio
    pos_weight = torch.tensor([value], dtype=torch.float32, device=device)
    return prevalence, pos_weight


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        alpha = self.alpha
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
        return sigmoid_focal_loss(logits, targets, alpha=alpha, gamma=self.gamma, reduction=self.reduction)


def build_criterion(loss_cfg: dict, device, train_loader=None, explicit_labels=None):
    loss_type = str(loss_cfg.get("type", "bce")).replace("-", "_")
    gamma = float(loss_cfg.get("gamma", 2.0))
    focal_alpha = loss_cfg.get("focal_alpha")

    prevalence = None
    pos_weight = None
    labels = None
    if explicit_labels is not None:
        labels = explicit_labels
    elif train_loader is not None:
        labels = labels_from_loader(train_loader)
    if labels is not None:
        prevalence = float(np.asarray(labels, dtype=int).mean())
    pos_weight_source = "none"
    if loss_type in {"bce_pos", "focal", "focal_pos"}:
        if labels is None:
            raise ValueError(f"loss type {loss_type} requires train_loader or explicit_labels.")
        if "pos_weight" in loss_cfg and loss_cfg.get("pos_weight") is not None:
            pos_weight = torch.tensor([float(loss_cfg["pos_weight"])], dtype=torch.float32, device=device)
            pos_weight_source = "manual"
        else:
            prevalence, pos_weight = compute_prevalence_and_pos_weight(
                labels,
                device=device,
                sqrt_weight=bool(loss_cfg.get("sqrt_pos_weight", True)),
            )
            pos_weight_source = "auto"
    print(
        "[loss] "
        f"type={loss_type} "
        f"train_prevalence={prevalence if prevalence is not None else 'unknown'} "
        f"pos_weight_used={pos_weight is not None} "
        f"pos_weight={None if pos_weight is None else round(float(pos_weight.item()), 6)} "
        f"pos_weight_source={pos_weight_source}"
    )

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "bce_pos":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        alpha = float(focal_alpha) if focal_alpha is not None else (1.0 - prevalence)
        criterion = FocalLossWithLogits(alpha=alpha, gamma=gamma)
    elif loss_type == "focal_pos":
        alpha = float(pos_weight.item() / (1.0 + pos_weight.item()))
        criterion = FocalLossWithLogits(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return criterion, {
        "loss_type": loss_type,
        "prevalence": prevalence,
        "pos_weight": None if pos_weight is None else float(pos_weight.item()),
        "pos_weight_source": pos_weight_source,
    }
