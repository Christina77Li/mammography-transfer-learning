from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import torch

from src.utils.paths import project_path


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


def load_initial_checkpoint(model, checkpoint_path, device):
    checkpoint_path = project_path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    metadata = {}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = {"epoch": payload.get("epoch"), "metrics": payload.get("metrics", {}), "context": payload.get("context", {})}
    else:
        sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".metrics.json")
        if sidecar.exists():
            values = json.loads(sidecar.read_text())
            metadata = {"epoch": values.get("epoch"), "metrics": {k: v for k, v in values.items() if k not in {"epoch", "context"}}, "context": values.get("context", {})}
    metadata["path"] = str(checkpoint_path)
    print(f"[init_checkpoint] loaded={checkpoint_path} actual_epoch={metadata.get('epoch')}")
    return metadata


def copy_selected_checkpoint(source, destination):
    """Copy weights and their adjacent metric record as one selected checkpoint."""
    source, destination = Path(source), Path(destination)
    shutil.copy2(source, destination)
    sidecar = source.with_suffix(source.suffix + ".metrics.json")
    if sidecar.exists():
        shutil.copy2(sidecar, destination.with_suffix(destination.suffix + ".metrics.json"))


def restore_selected_checkpoint(path, model, optimizer, device):
    """Restore matching weights, optimizer and context after stage selection."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    model._checkpoint_context = state.get("context", {})
    return state["epoch"], state["metrics"]


def prune_epoch_checkpoints(records, keep_paths):
    """Remove temporary epoch files after a successful stage rollback."""
    for record in records:
        path = Path(record["path"])
        if path not in keep_paths and path.exists():
            path.unlink()
            sidecar = path.with_suffix(path.suffix + ".metrics.json")
            if sidecar.exists():
                sidecar.unlink()
