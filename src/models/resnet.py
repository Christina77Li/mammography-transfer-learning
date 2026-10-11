from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 1, imagenet_weights: bool = True, name: str = "resnet101"):
    if name.lower() != "resnet101":
        raise ValueError(f"Unknown model name: {name}")
    weights = models.ResNet101_Weights.IMAGENET1K_V2 if imagenet_weights else None
    model = models.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
