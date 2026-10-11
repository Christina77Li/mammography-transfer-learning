from __future__ import annotations

import torch

from src.training.gradual_unfreeze import batchnorm_status, set_all_bn_eval, set_frozen_bn_eval


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    frozen_bn_blocks=None,
    freeze_all_bn=False,
    grad_clip_max_norm=None,
):
    model.train()
    if freeze_all_bn:
        set_all_bn_eval(model)
    if frozen_bn_blocks:
        set_frozen_bn_eval(model, frozen_bn_blocks)
    bn_status = batchnorm_status(model)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).view(-1)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite loss/logits; check data and optimization settings.")
        loss.backward()
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_max_norm))
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += batch_size

    if not total:
        raise ValueError("Cannot train on an empty dataloader.")
    return {"loss": running_loss / max(total, 1), "acc": correct / max(total, 1), **bn_status}
