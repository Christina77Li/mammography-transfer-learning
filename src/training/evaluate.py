from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, sweep_steps: int = 1001) -> dict:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_targets = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1)
        logits = model(inputs).view(-1)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite loss/logits; check data and optimization settings.")
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_n += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    if total_n == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits).numpy()
    y = targets.long().numpy()
    pred05 = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, pred05)
    recall = recall_score(y, pred05, zero_division=0)
    f1 = f1_score(y, pred05, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y, probs)
    except ValueError:
        pr_auc = float("nan")

    best_f1 = 0.0
    best_th = 0.5
    for th in np.linspace(0.0, 1.0, sweep_steps):
        f1_th = f1_score(y, (probs >= th).astype(int), zero_division=0)
        if f1_th > best_f1:
            best_f1 = float(f1_th)
            best_th = float(th)

    tn, fp, fn, tp = confusion_matrix(y, pred05, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    pos_mask = y == 1
    neg_mask = y == 0

    return {
        "loss": total_loss / max(total_n, 1),
        "acc@0.5": float(acc),
        "recall@0.5": float(recall),
        "f1@0.5": float(f1),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_th": float(best_th),
        "sensitivity@0.5": float(sensitivity),
        "specificity@0.5": float(specificity),
        "num_pred_pos@0.5": int(pred05.sum()),
        "prob_pos_mean": float(probs[pos_mask].mean()) if pos_mask.any() else float("nan"),
        "prob_neg_mean": float(probs[neg_mask].mean()) if neg_mask.any() else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(total_n),
        "prevalence": float(targets.float().mean().item()),
    }
