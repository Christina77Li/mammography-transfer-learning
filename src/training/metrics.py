"""Validation checks and epoch history fields."""
from __future__ import annotations

from src.data.dataset import labels_from_dataset
from src.training.gradual_unfreeze import frozen_blocks, count_trainable_params

def monitor_value(metrics: dict, monitor: str):
    if monitor.startswith("val_"):
        monitor = monitor[4:]
    return metrics.get("f1@0.5" if monitor == "f1" else monitor)


def dataset_prevalence(loader):
    labels = labels_from_dataset(loader.dataset)
    return int(len(labels)), float(labels.mean()) if len(labels) else 0.0


def checked_metrics(metrics):
    import math
    for key in ("loss", "auc", "pr_auc", "f1@0.5"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise ValueError(f"Validation metric {key} is missing or nonfinite.")
    return metrics


def phase_rows_common(config, phase_name, epoch, global_epoch, train_m, val_m, optimizer, model, unfreeze_event, elapsed):
    frozen, unfrozen = frozen_blocks(model)
    lr_summary = ",".join(f"{group.get('name', f'group_{i}')}:{group['lr']}" for i, group in enumerate(optimizer.param_groups))
    stage_name = "head_only" if not unfrozen else "+".join([*unfrozen, "head"])
    row = {
        "experiment_name": config["experiment_name"],
        "phase": phase_name,
        "stage_name": stage_name,
        "epoch": epoch,
        "global_epoch": global_epoch,
        "current_trainable_blocks": ",".join(unfrozen),
        "train_loss": train_m["loss"],
        "train_acc": train_m["acc"],
        "val_loss": val_m["loss"],
        "val_acc": val_m["acc@0.5"],
        "val_auc": val_m["auc"],
        "val_pr_auc": val_m["pr_auc"],
        "val_f1": val_m["f1@0.5"],
        "val_f1_at_0.5": val_m["f1@0.5"],
        "val_recall": val_m["recall@0.5"],
        "recall_at_0.5": val_m["recall@0.5"],
        "val_sensitivity": val_m["sensitivity@0.5"],
        "val_specificity": val_m["specificity@0.5"],
        "specificity_at_0.5": val_m["specificity@0.5"],
        "val_best_f1": val_m["best_f1"],
        "best_f1": val_m["best_f1"],
        "val_best_th": val_m["best_th"],
        "best_threshold": val_m["best_th"],
        "num_pred_pos_at_0.5": val_m["num_pred_pos@0.5"],
        "val_prob_pos_mean": val_m["prob_pos_mean"],
        "val_prob_neg_mean": val_m["prob_neg_mean"],
        "current_lr": lr_summary,
        "bn_total": train_m.get("bn_total"),
        "bn_eval": train_m.get("bn_eval"),
        "bn_train": train_m.get("bn_train"),
        "bn_all_eval": train_m.get("bn_all_eval"),
        "bn_trainable_param_tensors": train_m.get("bn_trainable_param_tensors"),
        "trainable_params": count_trainable_params(model),
        "frozen_blocks": ",".join(frozen),
        "unfrozen_blocks": ",".join(unfrozen),
        "unfreeze_event": unfreeze_event,
        "epoch_time_sec": elapsed,
    }
    for i, group in enumerate(optimizer.param_groups):
        row[f"lr_group_{i}"] = group["lr"]
        row[f"lr_group_{i}_name"] = group.get("name", f"group_{i}")
    return row


