from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 1, imagenet_weights: bool = True, name: str = "resnet101"):
    if name.lower() != "resnet101":
        raise ValueError(f"Unknown model name: {name}")
    weights = None
    if imagenet_weights:
        try:
            weights = models.ResNet101_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = "IMAGENET1K_V2"
    try:
        model = models.resnet101(weights=weights)
    except TypeError:
        model = models.resnet101(pretrained=imagenet_weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
