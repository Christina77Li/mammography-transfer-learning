"""Experiment setup, phase dispatch and final reporting."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
import torch
import torchvision
import yaml
from src.analysis.plot_curves import plot_experiment_curves
from src.data.dataset import build_dataloaders
from src.logging.history_logger import HistoryLogger
from src.logging.wandb_logger import WandbLogger
from src.models.resnet import get_model
from src.training.checkpointing import CheckpointManager, load_initial_checkpoint
from src.training.checkpoint_policy import reporting_checkpoint, best_metrics_from_checkpoint, write_stage_summary_csv
from src.training.phase_trainer import run_phase
from src.training.stage_trainer import run_stagewise_rollback_phase
from src.utils.config import preflight, effective_phase
from src.utils.paths import ensure_dir
from src.utils.provenance import git_provenance
from src.utils.seed import set_seed

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(path, config):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def versions():
    git_info = git_provenance()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "gpu_devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
            }
            for i in range(torch.cuda.device_count())
        ],
        **git_info,
        "git": git_info,
    }


def check_runtime(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f"Requested device {requested} is unavailable. Use --device cpu or auto.")
    return device


def run_experiment(config_path, config=None):
    config = preflight(config if config is not None else load_config(config_path))
    device = check_runtime(config)
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    # Validate every phase's split and labels without writing; failures leave output untouched.
    loader_cache = {}
    phases = config["phases"]
    for phase in phases:
        cfg = effective_phase(config, phase)
        key = json.dumps(cfg["data"], sort_keys=True)
        if key not in loader_cache:
            loader_cache[key] = build_dataloaders(cfg, create_splits=False)
    model_cfg = config.get("model", {})
    model = get_model(num_classes=1, imagenet_weights=bool(model_cfg.get("imagenet_weights", True)), name=model_cfg.get("name", "resnet101")).to(device)
    init_checkpoint = model_cfg.get("init_checkpoint") or config.get("init_checkpoint")
    loaded = load_initial_checkpoint(model, init_checkpoint, device) if init_checkpoint else {}
    config["loaded_checkpoint"] = loaded
    output_dir = Path(config["output_dir"])
    # Exclusive directory reservation prevents concurrent runs from mixing results.
    if output_dir.exists():
        output_dir.rmdir()  # only the previously validated empty directory
    output_dir.mkdir(parents=True, exist_ok=False)
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "figures")
    save_yaml(output_dir / "config_used.yaml", config)
    (output_dir / "environment.json").write_text(json.dumps(versions(), indent=2))
    historical = {p["name"]: {k: v for k, v in p.items() if k.startswith("source_")} for p in phases}
    (output_dir / "initialization.json").write_text(json.dumps({"loaded_checkpoint": loaded, "historical_reference": historical, "transfer_mode": "weights_only; new optimizer/scheduler"}, indent=2))
    history = HistoryLogger(output_dir / "history.csv")
    wandb_logger = WandbLogger(config, config["experiment_name"])
    global_epoch = 0
    summaries = []
    start = time.time()
    try:
        for phase in phases:
            cfg = effective_phase(config, phase)
            # Persist new deterministic split/subset caches only after all preflight checks.
            loaders = build_dataloaders(cfg)
            (output_dir / f"data_metadata_{phase['name']}.json").write_text(json.dumps(loaders["metadata"], indent=2))
            train_loader = loaders[phase.get("loader", "train")]
            manager = CheckpointManager(output_dir / "phases" / phase["name"], config=cfg)
            runner = run_stagewise_rollback_phase if phase.get("stagewise_rollback") else run_phase
            global_epoch, summary = runner(cfg, phase, model, train_loader, loaders["val"], output_dir, history, wandb_logger, device, global_epoch, checkpoint_manager=manager)
            summaries.append(summary)
            if phase.get("resume_best"):
                path = manager.checkpoint_dir / phase["resume_best"]
                load_initial_checkpoint(model, path, device)
        phase, monitor, selected_path, selected = reporting_checkpoint(output_dir, config, phases)
        scope = output_dir / "phases" / phase
        loss_metrics, loss_epoch = best_metrics_from_checkpoint(scope, "val_loss", {})
        auc_metrics, auc_epoch = best_metrics_from_checkpoint(scope, "val_auc", {})
        metrics = {k: v for k, v in selected.items() if k not in {"epoch", "context"}}
        loss_delta = float(config.get("early_stopping", {}).get("min_delta", .001))
        payload = {
            "experiment": config["experiment_name"],
            "best_epoch": selected["epoch"],
            "best_monitor": monitor,
            "best_phase": phase,
            "best_stage": selected.get("context", {}).get("stage"),
            "selected_checkpoint_path": str(selected_path),
            "training_time_total_min": (time.time() - start) / 60,
            "best_val_loss": loss_metrics.get("loss"),
            "best_val_loss_epoch": loss_epoch,
            "auc_at_best_val_loss": loss_metrics.get("auc"),
            "best_val_auc": auc_metrics.get("auc"),
            "best_val_auc_epoch": auc_epoch,
            "val_loss_at_best_val_auc": auc_metrics.get("loss"),
            "best_val_auc_checkpoint_path": str(scope / "checkpoints" / "best_auc.pth"),
            "val_loss_increased_after_best_loss": any(row.get("phase") == phase and int(row.get("global_epoch", 0)) > loss_epoch and row.get("val_loss") is not None and float(row["val_loss"]) > loss_metrics["loss"] + loss_delta for row in history.rows),
            "auc_and_val_loss_diverged": loss_epoch != auc_epoch and auc_metrics["loss"] > loss_metrics["loss"] + loss_delta,
            **metrics,
            "report_scope": "one phase; default last full-data phase",
            "loaded_checkpoint": loaded,
            "phase_summaries": summaries,
            "git": git_provenance(),
        }
        (output_dir / "final_metrics.json").write_text(json.dumps(payload, indent=2))
        write_stage_summary_csv(output_dir, summaries)
        plot_experiment_curves(output_dir / "history.csv", output_dir / "figures")
        print(f"[final] phase={phase} stage={payload['best_stage']} epoch={selected['epoch']} monitor={monitor} checkpoint={selected_path}")
        return payload
    except Exception as exc:
        (output_dir / "run_failed.json").write_text(json.dumps({"error": str(exc), "completed_phases": [s["phase"] for s in summaries]}, indent=2))
        raise
    finally:
        wandb_logger.finish()


