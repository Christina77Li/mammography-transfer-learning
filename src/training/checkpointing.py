from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "config": config,
            "context": getattr(model, "_checkpoint_context", {}),
            "resume_supported": False,
        },
        path,
    )
    metrics_path = path.with_suffix(path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps({"epoch": int(epoch), **metrics, "context": getattr(model, "_checkpoint_context", {})}, indent=2))


class CheckpointManager:
    def __init__(self, output_dir, config=None):
        self.checkpoint_dir = Path(output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.best = {"loss": float("inf"), "auc": -float("inf"), "f1": -float("inf")}
        self.best_epoch = {}

    def update(self, model, optimizer, epoch, val_metrics):
        val_loss = float(val_metrics.get("loss", float("inf")))
        val_auc = float(val_metrics.get("auc", float("nan")))
        val_f1 = float(val_metrics.get("f1@0.5", -float("inf")))

        if math.isfinite(val_loss) and val_loss < self.best["loss"]:
            self.best["loss"] = val_loss
            self.best_epoch["loss"] = epoch
            save_checkpoint(self.checkpoint_dir / "best_loss.pth", model, optimizer, epoch, val_metrics, self.config)
        if math.isfinite(val_auc) and val_auc > self.best["auc"]:
            self.best["auc"] = val_auc
            self.best_epoch["auc"] = epoch
            save_checkpoint(self.checkpoint_dir / "best_auc.pth", model, optimizer, epoch, val_metrics, self.config)
        if math.isfinite(val_f1) and val_f1 > self.best["f1"]:
            self.best["f1"] = val_f1
            self.best_epoch["f1"] = epoch
            save_checkpoint(self.checkpoint_dir / "best_f1.pth", model, optimizer, epoch, val_metrics, self.config)

        save_checkpoint(self.checkpoint_dir / "last.pth", model, optimizer, epoch, val_metrics, self.config)

    def best_for_monitor(self, monitor: str):
        key = monitor.replace("val_", "")
        if key == "f1@0.5":
            key = "f1"
        if key not in self.best:
            key = "loss"
        return self.best.get(key), self.best_epoch.get(key)
