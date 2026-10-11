"""Loss guards and final reporting scope, independent of training execution."""
from __future__ import annotations

import json
import csv
from pathlib import Path

def select_guarded(records, best_loss, tolerance, absolute_limit=None):
    eligible = [row for row in records if row["val_loss"] <= best_loss + tolerance and (absolute_limit is None or row["val_loss"] <= absolute_limit)]
    if not eligible:
        raise ValueError("No checkpoint satisfies the configured loss guard. Training stopped without selecting an ineligible checkpoint.")
    return max(eligible, key=lambda row: (row["val_auc"], -row["val_loss"]))


def reporting_checkpoint(output_dir, config, phases):
    requested = config.get("report", {}).get("phase")
    full = [p for p in phases if p.get("loader", "train") == "train"]
    phase = requested or (full[-1]["name"] if full else phases[-1]["name"])
    monitor = config.get("report", {}).get("monitor", config.get("save_best", {}).get("monitor", config.get("early_stopping", {}).get("monitor", "val_loss")))
    name = {"val_loss": "best_loss.pth", "val_auc": "best_auc.pth", "val_f1": "best_f1.pth", "val_f1@0.5": "best_f1.pth"}[monitor]
    path = output_dir / "phases" / phase / "checkpoints" / name
    if not path.exists():
        raise ValueError(f"Reporting checkpoint was not saved: {path}")
    payload = json.loads(path.with_suffix(path.suffix + ".metrics.json").read_text())
    return phase, monitor, path, payload


def best_metrics_from_checkpoint(output_dir: Path, monitor: str, fallback: dict):
    key = monitor.replace("val_", "")
    checkpoint_name = {
        "loss": "best_loss.pth",
        "auc": "best_auc.pth",
        "f1": "best_f1.pth",
        "f1@0.5": "best_f1.pth",
    }.get(key)
    if not checkpoint_name:
        return fallback, None
    metrics_path = output_dir / "checkpoints" / f"{checkpoint_name}.metrics.json"
    if not metrics_path.exists():
        return fallback, None
    payload = json.loads(metrics_path.read_text())
    epoch = payload.pop("epoch", None)
    return payload, epoch


def write_stage_summary_csv(output_dir: Path, phase_summaries: list[dict]) -> None:
    fields = [
        "stage",
        "selected_epoch",
        "selected_val_loss",
        "selected_val_auc",
        "best_val_loss",
        "best_val_loss_epoch",
        "best_val_loss_auc",
        "best_val_auc",
        "best_val_auc_epoch",
        "best_val_auc_loss",
        "selection_rule",
        "loss_tolerance",
        "selected_checkpoint_path",
    ]
    rows = []
    for summary in phase_summaries:
        phase = summary.get("phase")
        for stage, selected in (summary.get("selected_transition_checkpoints") or {}).items():
            rows.append(
                {
                    "stage": f"{phase}/{stage}",
                    "selected_epoch": selected.get("epoch"),
                    "selected_val_loss": selected.get("val_loss"),
                    "selected_val_auc": selected.get("val_auc"),
                    "best_val_loss": selected.get("stage_best_val_loss"),
                    "best_val_loss_epoch": selected.get("stage_best_val_loss_epoch"),
                    "best_val_loss_auc": selected.get("stage_best_val_loss_auc"),
                    "best_val_auc": selected.get("stage_best_val_auc"),
                    "best_val_auc_epoch": selected.get("stage_best_val_auc_epoch"),
                    "best_val_auc_loss": selected.get("stage_best_val_auc_loss"),
                    "selection_rule": selected.get("selection_rule"),
                    "loss_tolerance": selected.get("loss_tolerance"),
                    "selected_checkpoint_path": selected.get("path"),
                }
            )
    if not rows:
        return
    path = output_dir / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


