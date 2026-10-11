from __future__ import annotations

import json
import platform
import shutil
import time
import copy
import csv
from pathlib import Path

import torch
import yaml
import torchvision

from src.analysis.plot_curves import plot_experiment_curves
from src.data.dataset import build_dataloaders, labels_from_dataset
from src.logging.history_logger import HistoryLogger
from src.logging.wandb_logger import WandbLogger
from src.models.resnet import get_model
from src.training.checkpointing import CheckpointManager, save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.evaluate import evaluate
from src.training.gradual_unfreeze import (
    BACKBONE_BLOCKS,
    GradualUnfreezeController,
    batchnorm_status,
    count_trainable_params,
    freeze_backbone_train_head,
    frozen_blocks,
    modules_for_block,
    set_all_bn_eval,
    unfreeze_all,
)
from src.training.losses import build_criterion
from src.training.scheduler import build_scheduler, step_scheduler
from src.training.train_epoch import train_one_epoch
from src.utils.paths import ensure_dir, project_path
from src.utils.provenance import git_provenance
from src.utils.seed import set_seed
from src.utils.config import effective_phase, preflight, bn_policy


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(path, config):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def phase_config(config: dict, phase_cfg: dict) -> dict:
    """Return a config copy with optional phase-level data overrides applied."""
    return effective_phase(config, phase_cfg)


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


def monitor_value(metrics: dict, monitor: str):
    if monitor.startswith("val_"):
        monitor = monitor[4:]
    return metrics.get("f1@0.5" if monitor == "f1" else monitor)


def dataset_prevalence(loader):
    labels = labels_from_dataset(loader.dataset)
    return int(len(labels)), float(labels.mean()) if len(labels) else 0.0


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


def make_optimizer(model, config, gradual=False):
    opt_cfg = config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 5e-4))
    if gradual:
        params = [{"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay, "name": "head"}]
    elif "backbone_lr" in opt_cfg and "head_lr" in opt_cfg:
        head_param_ids = {id(p) for p in model.fc.parameters()}
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
        head_params = [p for p in model.fc.parameters() if p.requires_grad]
        params = [
            {
                "params": backbone_params,
                "lr": float(opt_cfg["backbone_lr"]),
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": float(opt_cfg["head_lr"]),
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]
    else:
        params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay, "name": "all_trainable"}]
    opt_type = opt_cfg.get("type", "adam").lower()
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(opt_cfg.get("momentum", 0.9)), weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def configure_stage_trainability(model, trainable_blocks, freeze_all_bn=False):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    for block in trainable_blocks:
        for module in modules_for_block(model, block):
            for p in module.parameters():
                p.requires_grad = True
            module.train()
    if freeze_all_bn:
        set_all_bn_eval(model)


def make_stage_optimizer(model, config, stage_cfg):
    opt_cfg = config.get("optimizer", {})
    weight_decay = float(opt_cfg.get("weight_decay", 5e-4))
    params = []
    head_lr = float(stage_cfg.get("head_lr", opt_cfg.get("lr", 1e-3)))
    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    if head_params:
        params.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "head"})
    for block in stage_cfg.get("trainable_blocks", []):
        block_params = []
        for module in modules_for_block(model, block):
            block_params.extend([p for p in module.parameters() if p.requires_grad])
        if block_params:
            params.append(
                {
                    "params": block_params,
                    "lr": float(stage_cfg[f"{block}_lr"]),
                    "weight_decay": weight_decay,
                    "name": block,
                }
            )
    opt_type = opt_cfg.get("type", "adam").lower()
    if opt_type == "adamw":
        return torch.optim.AdamW(params, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(params, momentum=float(opt_cfg.get("momentum", 0.9)), weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(params, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer type: {opt_type}")




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


def check_runtime(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f"Requested device {requested} is unavailable. Use --device cpu or auto.")
    return device


def apply_bn_affine_policy(model, config):
    _, train_affine = bn_policy(config)
    if not train_affine:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                for parameter in module.parameters():
                    parameter.requires_grad = False


def checked_metrics(metrics):
    import math
    for key in ("loss", "auc", "pr_auc", "f1@0.5"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise ValueError(f"Validation metric {key} is missing or nonfinite.")
    return metrics


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


def run_phase(
    config,
    phase_cfg,
    model,
    train_loader,
    val_loader,
    output_dir,
    history,
    wandb_logger,
    device,
    global_epoch_start,
    checkpoint_manager=None,
):
    phase_name = phase_cfg["name"]
    init_checkpoint = config.get("model", {}).get("init_checkpoint") or config.get("init_checkpoint")
    gradual = bool(phase_cfg.get("gradual_unfreeze", False))
    freeze_head_only = bool(phase_cfg.get("freeze_backbone", False) or gradual)
    if freeze_head_only:
        freeze_backbone_train_head(model)
    else:
        unfreeze_all(model)

    freeze_all_bn, _ = bn_policy(config)
    apply_bn_affine_policy(model, config)
    optimizer = make_optimizer(model, config, gradual=gradual)
    grad_clip_cfg = config.get("training", {}).get("gradient_clipping", {})
    grad_clip_max_norm = None
    if bool(grad_clip_cfg.get("enabled", False)):
        grad_clip_max_norm = float(grad_clip_cfg.get("max_norm", 1.0))
    criterion, loss_meta = build_criterion(config.get("loss", {"type": "bce"}), device, train_loader=train_loader)
    scheduler = build_scheduler(config, optimizer)
    early_cfg = config.get("early_stopping", {})
    early = EarlyStopping(
        monitor=early_cfg.get("monitor", "val_loss"),
        mode=early_cfg.get("mode", "min"),
        patience=early_cfg.get("patience", 10),
        min_delta=early_cfg.get("min_delta", 0.001),
        start_epoch=early_cfg.get("start_epoch", 10),
    )
    checkpoint_manager = checkpoint_manager or CheckpointManager(output_dir / "phases" / phase_name, config=config)
    controller = None
    if gradual:
        gu_cfg = config.get("gradual_unfreeze", {})
        controller = GradualUnfreezeController(
            model,
            optimizer,
            plan=gu_cfg.get("plan"),
            monitor=gu_cfg.get("monitor", config.get("scheduler", {}).get("monitor", "val_loss")),
            mode=gu_cfg.get("mode", config.get("scheduler", {}).get("mode", "min")),
            patience=gu_cfg.get("patience", 3),
            min_delta=gu_cfg.get("min_delta", 0.001),
            recovery_tolerance=gu_cfg.get("recovery_tolerance"),
        )
        fixed_stage_epochs = [int(x) for x in gu_cfg.get("fixed_stage_epochs", [])]
        fixed_unfreeze_epochs = []
        if fixed_stage_epochs:
            cumulative = 0
            for stage_epochs in fixed_stage_epochs[:-1]:
                cumulative += stage_epochs
                fixed_unfreeze_epochs.append(cumulative + 1)
    else:
        fixed_unfreeze_epochs = []

    best_monitor_metric = None
    best_auc_value = -float("inf")
    best_auc_epoch = None
    best_auc_stage = None
    phase_best_loss = float("inf")
    phase_best_loss_epoch = None
    phase_best_loss_auc = None
    phase_best_auc_loss = None
    phase_epoch_records = []
    checkpoint_selection_cfg = phase_cfg.get("checkpoint_selection", config.get("checkpoint_selection", {}))
    save_epoch_checkpoints = bool(checkpoint_selection_cfg.get("enabled", False) or checkpoint_selection_cfg.get("selection"))
    phase_best_loss_path = output_dir / "checkpoints" / f"{phase_name}_best_val_loss.pth"
    phase_best_auc_path = output_dir / "checkpoints" / f"{phase_name}_best_val_auc.pth"
    subset_ratio = config.get("data", {}).get("small_subset", {}).get("pos_neg_ratio")
    train_subset_size, train_subset_prev = dataset_prevalence(train_loader)
    full_val_size, full_val_prev = dataset_prevalence(val_loader)
    last_metrics = None
    global_epoch = global_epoch_start
    for epoch in range(1, int(phase_cfg.get("epochs", config.get("training", {}).get("epochs", 1))) + 1):
        start = time.time()
        unfreeze_event = ""
        if controller is not None:
            ramp_event = controller.maybe_apply_ramp()
            if ramp_event:
                unfreeze_event = ramp_event
        if controller is not None and epoch in fixed_unfreeze_epochs:
            event = controller.unfreeze_next()
            unfreeze_event = ",".join([item for item in [unfreeze_event, event] if item])
            scheduler = build_scheduler(config, optimizer)
        frozen_bn_blocks, _ = frozen_blocks(model) if freeze_head_only else ([], [])
        train_m = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            frozen_bn_blocks=frozen_bn_blocks,
            freeze_all_bn=freeze_all_bn,
            grad_clip_max_norm=grad_clip_max_norm,
        )
        if freeze_all_bn:
            print(f"[batchnorm] epoch={epoch} {batchnorm_status(model)}")
        val_m = checked_metrics(evaluate(model, val_loader, criterion, device))
        last_metrics = val_m

        lrs_before_scheduler = [group["lr"] for group in optimizer.param_groups]
        sched_monitor = config.get("scheduler", {}).get("monitor", "val_loss")
        step_scheduler(scheduler, monitor_value(val_m, sched_monitor))
        lr_reduced = any(group["lr"] < before for group, before in zip(optimizer.param_groups, lrs_before_scheduler))

        if controller is not None and not fixed_unfreeze_epochs:
            def save_pre_unfreeze(block):
                save_checkpoint(
                    output_dir / "checkpoints" / f"pre_unfreeze_{block}_epoch{global_epoch + 1}.pth",
                    model,
                    optimizer,
                    global_epoch + 1,
                    val_m,
                    config,
                )

            event = controller.step(monitor_value(val_m, controller.monitor), pre_unfreeze_callback=save_pre_unfreeze)
            if event:
                unfreeze_event = ",".join([item for item in [unfreeze_event, event] if item])
                if event.endswith("_unfrozen"):
                    block_name = event.removesuffix("_unfrozen")
                    save_checkpoint(
                        output_dir / "checkpoints" / f"post_unfreeze_{block_name}_epoch{global_epoch + 1}.pth",
                        model,
                        optimizer,
                        global_epoch + 1,
                        val_m,
                        config,
                    )
                    scheduler = build_scheduler(config, optimizer)
                elif event.endswith("_cooldown_complete"):
                    block_name = event.removesuffix("_cooldown_complete")
                    save_checkpoint(
                        output_dir / "checkpoints" / f"post_cooldown_{block_name}_epoch{global_epoch + 1}.pth",
                        model,
                        optimizer,
                        global_epoch + 1,
                        val_m,
                        config,
                    )
        if lr_reduced:
            unfreeze_event = ",".join([event for event in [unfreeze_event, "lr_reduced"] if event])

        global_epoch += 1
        _, active_blocks = frozen_blocks(model)
        model._checkpoint_context = {"phase": phase_name, "stage": "head_only" if not active_blocks else "+".join([*active_blocks, "head"]), "loader": phase_cfg.get("loader", "train")}
        elapsed = time.time() - start
        row = phase_rows_common(config, phase_name, epoch, global_epoch, train_m, val_m, optimizer, model, unfreeze_event, elapsed)
        row["subset_ratio"] = None if subset_ratio is None else f"1:{subset_ratio}"
        row["train_subset_size"] = train_subset_size
        row["train_subset_prevalence"] = train_subset_prev
        row["full_validation_prevalence"] = full_val_prev
        row["initialized_from_checkpoint"] = init_checkpoint or None
        source = config.get("loaded_checkpoint", {})
        row["source_checkpoint_type"] = "loaded_weights" if source else None
        row["source_epoch"] = source.get("epoch")
        row["source_val_loss"] = source.get("metrics", {}).get("loss")
        row["source_val_auc"] = source.get("metrics", {}).get("auc")
        row["early_stopping_counter"] = early.num_bad_epochs
        if val_m["auc"] > best_auc_value:
            best_auc_value = val_m["auc"]
            best_auc_epoch = global_epoch
            best_auc_stage = row["stage_name"]
            phase_best_auc_loss = val_m["loss"]
            save_checkpoint(phase_best_auc_path, model, optimizer, global_epoch, val_m, config)
        if val_m["loss"] < phase_best_loss:
            phase_best_loss = val_m["loss"]
            phase_best_loss_epoch = global_epoch
            phase_best_loss_auc = val_m["auc"]
            save_checkpoint(phase_best_loss_path, model, optimizer, global_epoch, val_m, config)
        epoch_checkpoint_path = None
        if save_epoch_checkpoints:
            epoch_checkpoint_path = output_dir / "checkpoints" / f"{phase_name}_epoch{global_epoch}.pth"
            save_checkpoint(epoch_checkpoint_path, model, optimizer, global_epoch, val_m, config)
            phase_epoch_records.append(
                {
                    "epoch": global_epoch,
                    "phase_epoch": epoch,
                    "val_loss": float(val_m["loss"]),
                    "val_auc": float(val_m["auc"]),
                    "val_pr_auc": float(val_m["pr_auc"]),
                    "val_f1": float(val_m["f1@0.5"]),
                    "path": epoch_checkpoint_path,
                }
            )
        wandb_payload = {k.replace("val_", "val/").replace("train_", "train/"): v for k, v in row.items()}
        wandb_payload.update({"epoch": global_epoch, "phase": phase_name})
        wandb_logger.log(wandb_payload)

        checkpoint_manager.update(model, optimizer, global_epoch, val_m)
        early_metric = monitor_value(val_m, early.monitor)
        should_stop = early.step(early_metric, epoch)
        row["early_stopping_counter"] = early.num_bad_epochs
        history.append(row)
        if early.save_best:
            best_monitor_metric = early_metric
        print(
            f"[{config['experiment_name']}:{phase_name}] epoch {epoch} "
            f"init={init_checkpoint or ('imagenet' if config.get('model', {}).get('imagenet_weights', True) else 'random')} "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} "
            f"val_auc={val_m['auc']:.4f} val_f1={val_m['f1@0.5']:.4f} event={unfreeze_event or '-'}"
        )
        if should_stop:
            print(f"[early stopping] phase={phase_name} epoch={epoch} monitor={early.monitor}")
            break

    selected_transition = None
    if checkpoint_selection_cfg.get("selection") == "best_auc_with_loss_tolerance" and phase_epoch_records:
        loss_tolerance = float(
            checkpoint_selection_cfg.get(
                "loss_tolerance_from_best_loss",
                checkpoint_selection_cfg.get("loss_tolerance", 0.0),
            )
        )
        max_allowed_val_loss = checkpoint_selection_cfg.get("max_allowed_val_loss")
        if max_allowed_val_loss is not None:
            max_allowed_val_loss = float(max_allowed_val_loss)
        selected = select_guarded(phase_epoch_records, phase_best_loss, loss_tolerance, max_allowed_val_loss)
        tolerance_satisfied = True
        if selected is not None:
            reason = "best_auc_within_loss_tolerance"
            if selected["epoch"] == phase_best_loss_epoch:
                reason = "best_val_loss"
            elif selected["epoch"] == best_auc_epoch:
                reason = "best_val_auc"
            selected_path = Path(selected["path"])
            transition_path = output_dir / "checkpoints" / "selected_transition_checkpoint.pth"
            shutil.copy2(selected_path, transition_path)
            metrics_src = selected_path.with_suffix(selected_path.suffix + ".metrics.json")
            if metrics_src.exists():
                shutil.copy2(metrics_src, transition_path.with_suffix(transition_path.suffix + ".metrics.json"))
            selected_transition = {
                "path": str(transition_path),
                "epoch": selected["epoch"],
                "val_loss": selected["val_loss"],
                "val_auc": selected["val_auc"],
                "val_pr_auc": selected.get("val_pr_auc"),
                "val_f1": selected.get("val_f1"),
                "reason": reason,
                "loss_tolerance": loss_tolerance,
                "max_allowed_val_loss": max_allowed_val_loss,
                "tolerance_satisfied": tolerance_satisfied,
                "best_val_loss": phase_best_loss,
                "best_val_loss_epoch": phase_best_loss_epoch,
                "best_val_loss_auc": phase_best_loss_auc,
                "best_val_auc": best_auc_value,
                "best_val_auc_epoch": best_auc_epoch,
                "best_val_auc_loss": phase_best_auc_loss,
            }
            print(
                "[checkpoint selection] "
                f"best_val_loss={phase_best_loss:.6f} best_val_loss_epoch={phase_best_loss_epoch} "
                f"best_val_loss_auc={phase_best_loss_auc:.6f} "
                f"best_val_auc={best_auc_value:.6f} best_val_auc_epoch={best_auc_epoch} "
                f"best_val_auc_loss={phase_best_auc_loss:.6f} "
                f"selected_transition_epoch={selected['epoch']} "
                f"selected_transition_val_loss={selected['val_loss']:.6f} "
                f"selected_transition_val_auc={selected['val_auc']:.6f} "
                f"selected_transition_reason={reason} selected_checkpoint_path={transition_path} "
                f"loss_tolerance={loss_tolerance} max_allowed_val_loss={max_allowed_val_loss} "
                f"tolerance_satisfied={tolerance_satisfied}"
            )
            history.append(
                {
                    "experiment_name": config["experiment_name"],
                    "phase": phase_name,
                    "epoch": selected["phase_epoch"],
                    "global_epoch": global_epoch,
                    "unfreeze_event": "selected_transition_checkpoint",
                    "selected_transition_epoch": selected["epoch"],
                    "selected_transition_val_loss": selected["val_loss"],
                    "selected_transition_val_auc": selected["val_auc"],
                    "selected_transition_reason": reason,
                    "selected_checkpoint_path": str(transition_path),
                    "loss_tolerance_used": loss_tolerance,
                    "max_allowed_val_loss": max_allowed_val_loss,
                    "transition_tolerance_satisfied": tolerance_satisfied,
                    "stage_best_val_loss": phase_best_loss,
                    "stage_best_val_loss_epoch": phase_best_loss_epoch,
                    "stage_best_val_loss_auc": phase_best_loss_auc,
                    "stage_best_val_auc": best_auc_value,
                    "stage_best_val_auc_epoch": best_auc_epoch,
                    "stage_best_val_auc_loss": phase_best_auc_loss,
                    "subset_ratio": None if subset_ratio is None else f"1:{subset_ratio}",
                    "train_subset_size": train_subset_size,
                    "train_subset_prevalence": train_subset_prev,
                    "full_validation_prevalence": full_val_prev,
                }
            )

    if last_metrics is not None:
        save_checkpoint(output_dir / "checkpoints" / "final.pth", model, optimizer, global_epoch, last_metrics, config)

    _, final_unfrozen = frozen_blocks(model)
    conv1_reached = "conv1" in final_unfrozen
    if conv1_reached and bool(phase_cfg.get("save_final_conv1_completed", False)):
        save_checkpoint(
            output_dir / "checkpoints" / "final_conv1_completed.pth",
            model,
            optimizer,
            global_epoch,
            last_metrics or {},
            config,
        )

    phase_summary = {
        "phase": phase_name,
        "subset_size": train_subset_size,
        "subset_prevalence": train_subset_prev,
        "val_size": full_val_size,
        "val_prevalence": full_val_prev,
        "last_metrics": last_metrics,
        "loss_meta": loss_meta,
        "best_monitor": early.monitor,
        "best_monitor_value": best_monitor_metric,
        "best_auc_value": best_auc_value,
        "best_auc_epoch": best_auc_epoch,
        "best_auc_stage": best_auc_stage,
        "best_val_loss_checkpoint": str(phase_best_loss_path) if phase_best_loss_path.exists() else None,
        "best_val_auc_checkpoint": str(phase_best_auc_path) if phase_best_auc_path.exists() else None,
        "selected_transition_checkpoint": selected_transition,
        "final_stage": "head_only" if not final_unfrozen else "+".join([*final_unfrozen, "head"]),
        "conv1_reached": conv1_reached,
        "conv1_completed": conv1_reached,
        "global_epoch": global_epoch,
    }
    return global_epoch, phase_summary


def run_stagewise_rollback_phase(
    config,
    phase_cfg,
    model,
    train_loader,
    val_loader,
    output_dir,
    history,
    wandb_logger,
    device,
    global_epoch_start,
    checkpoint_manager=None,
):
    phase_name = phase_cfg["name"]
    effective_config = effective_phase(config, phase_cfg)
    freeze_all_bn, _ = bn_policy(effective_config)
    criterion, loss_meta = build_criterion(effective_config.get("loss", {"type": "bce"}), device, train_loader=train_loader)
    checkpoint_manager = checkpoint_manager or CheckpointManager(output_dir / "phases" / phase_name, config=effective_config)
    transition_cfg = effective_config.get("stage_transition", {})
    transition_patience = int(transition_cfg.get("patience", 3))
    transition_min_delta = float(transition_cfg.get("min_delta", 0.001))
    transition_monitor = transition_cfg.get("monitor", "val_loss")
    transition_mode = transition_cfg.get("mode", "min")
    default_rollback_to = transition_cfg.get("selection", transition_cfg.get("rollback_to", transition_cfg.get("default_selection", "best_val_loss")))
    default_loss_tolerance = float(
        transition_cfg.get(
            "loss_tolerance",
            transition_cfg.get("loss_tolerance_from_best_loss", 0.0),
        )
    )
    if (
        default_rollback_to == "best_auc_with_loss_tolerance"
        and float(transition_cfg.get("loss_tolerance_from_best_loss", 0.0)) > 0.0
        and default_loss_tolerance == 0.0
    ):
        raise ValueError(
            "stage_transition uses best_auc_with_loss_tolerance with positive "
            "loss_tolerance_from_best_loss, but resolved loss_tolerance is 0.0."
        )
    stage_specific_transition = transition_cfg.get("stage_specific", {})

    best_auc_value = -float("inf")
    best_auc_epoch = None
    best_auc_stage = None
    best_val_loss_stage_checkpoints = {}
    selected_transition_checkpoints = {}
    last_metrics = None
    global_epoch = global_epoch_start
    final_optimizer = None
    selected_model_epoch = None
    train_subset_size, train_subset_prev = dataset_prevalence(train_loader)
    full_val_size, full_val_prev = dataset_prevalence(val_loader)

    for stage_index, stage_cfg in enumerate(effective_config.get("stagewise_stages", []), start=1):
        stage_label = stage_cfg["name"]
        stage_transition_cfg = stage_specific_transition.get(stage_label, {})
        rollback_to = stage_transition_cfg.get("selection", default_rollback_to)
        loss_tolerance = float(
            stage_transition_cfg.get(
                "loss_tolerance",
                stage_transition_cfg.get("loss_tolerance_from_best_loss", default_loss_tolerance),
            )
        )
        configured_positive_tolerance = float(stage_transition_cfg.get("loss_tolerance_from_best_loss", 0.0)) > 0.0
        configured_positive_tolerance = configured_positive_tolerance or (
            not stage_transition_cfg and float(transition_cfg.get("loss_tolerance_from_best_loss", 0.0)) > 0.0
        )
        if rollback_to == "best_auc_with_loss_tolerance" and configured_positive_tolerance and loss_tolerance == 0.0:
            raise ValueError(
                f"Stage {stage_label} uses best_auc_with_loss_tolerance with positive "
                "loss_tolerance_from_best_loss, but resolved loss_tolerance is 0.0."
            )
        max_allowed_val_loss = stage_transition_cfg.get("max_allowed_val_loss", transition_cfg.get("max_allowed_val_loss"))
        if max_allowed_val_loss is not None:
            max_allowed_val_loss = float(max_allowed_val_loss)
        trainable_blocks = stage_cfg.get("trainable_blocks", [])
        configure_stage_trainability(model, trainable_blocks, freeze_all_bn=freeze_all_bn)
        apply_bn_affine_policy(model, effective_config)
        optimizer = make_stage_optimizer(model, effective_config, stage_cfg)
        scheduler = build_scheduler(effective_config, optimizer)
        final_optimizer = optimizer
        stage_best_loss = float("inf")
        stage_best_loss_epoch = None
        stage_best_loss_auc = None
        stage_best_auc = -float("inf")
        stage_best_auc_epoch = None
        stage_best_auc_loss = None
        stage_bad_transition_epochs = 0
        stage_best_transition_value = float("inf") if transition_mode == "min" else -float("inf")
        stage_best_loss_path = output_dir / "checkpoints" / f"{phase_name}_stage_{stage_index}_{stage_label}_best_val_loss.pth"
        stage_best_auc_path = output_dir / "checkpoints" / f"{phase_name}_stage_{stage_index}_{stage_label}_best_val_auc.pth"
        stage_epoch_records = []

        for stage_epoch in range(1, int(stage_cfg.get("max_epochs", 1)) + 1):
            start = time.time()
            train_m = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                frozen_bn_blocks=[],
                freeze_all_bn=freeze_all_bn,
                grad_clip_max_norm=(effective_config.get("training", {}).get("gradient_clipping", {}).get("max_norm", 1.0) if effective_config.get("training", {}).get("gradient_clipping", {}).get("enabled", False) else None),
            )
            if freeze_all_bn:
                print(f"[batchnorm] stage={stage_label} epoch={stage_epoch} {batchnorm_status(model)}")
            val_m = checked_metrics(evaluate(model, val_loader, criterion, device))
            last_metrics = val_m

            lrs_before_scheduler = [group["lr"] for group in optimizer.param_groups]
            sched_monitor = effective_config.get("scheduler", {}).get("monitor", "val_loss")
            step_scheduler(scheduler, monitor_value(val_m, sched_monitor))
            lr_reduced = any(group["lr"] < before for group, before in zip(optimizer.param_groups, lrs_before_scheduler))

            global_epoch += 1
            model._checkpoint_context = {"phase": phase_name, "stage": stage_label, "loader": phase_cfg.get("loader", "train")}
            event_parts = []
            if stage_epoch == 1:
                event_parts.extend(["optimizer_reset", "scheduler_reset"])
            if lr_reduced:
                event_parts.append("lr_reduced")

            loss_improved = val_m["loss"] < stage_best_loss
            if loss_improved:
                stage_best_loss = val_m["loss"]
                stage_best_loss_epoch = global_epoch
                stage_best_loss_auc = val_m["auc"]
                save_checkpoint(stage_best_loss_path, model, optimizer, global_epoch, val_m, effective_config)
                event_parts.append("stage_best_val_loss")

            transition_value = monitor_value(val_m, transition_monitor)
            if transition_value is None:
                raise ValueError(f"Transition monitor {transition_monitor!r} was not found in validation metrics.")
            if transition_mode == "min":
                transition_improved = transition_value < stage_best_transition_value - transition_min_delta
            else:
                transition_improved = transition_value > stage_best_transition_value + transition_min_delta
            if transition_improved:
                stage_best_transition_value = transition_value
                stage_bad_transition_epochs = 0
            else:
                stage_bad_transition_epochs += 1

            if val_m["auc"] > stage_best_auc:
                stage_best_auc = val_m["auc"]
                stage_best_auc_epoch = global_epoch
                stage_best_auc_loss = val_m["loss"]
                save_checkpoint(stage_best_auc_path, model, optimizer, global_epoch, val_m, effective_config)
                event_parts.append("stage_best_val_auc")
            epoch_checkpoint_path = output_dir / "checkpoints" / f"{phase_name}_stage_{stage_index}_{stage_label}_epoch{global_epoch}.pth"
            save_checkpoint(epoch_checkpoint_path, model, optimizer, global_epoch, val_m, effective_config)
            stage_epoch_records.append(
                {
                    "epoch": global_epoch,
                    "stage_epoch": stage_epoch,
                    "val_loss": float(val_m["loss"]),
                    "val_auc": float(val_m["auc"]),
                    "val_pr_auc": float(val_m["pr_auc"]),
                    "val_f1": float(val_m["f1@0.5"]),
                    "path": epoch_checkpoint_path,
                }
            )

            elapsed = time.time() - start
            row = phase_rows_common(
                effective_config,
                phase_name,
                stage_epoch,
                global_epoch,
                train_m,
                val_m,
                optimizer,
                model,
                ",".join(event_parts),
                elapsed,
            )
            row["stage_name"] = stage_label
            row["stage_epoch"] = stage_epoch
            row["stage_transition_counter"] = stage_bad_transition_epochs
            row["train_data_type"] = phase_cfg.get("train_data_type", phase_cfg.get("loader", "train"))
            row["optimizer_reset"] = stage_epoch == 1
            row["scheduler_reset"] = stage_epoch == 1
            row["rollback_to_best_val_loss"] = False
            history.append(row)
            wandb_payload = {k.replace("val_", "val/").replace("train_", "train/"): v for k, v in row.items()}
            wandb_payload.update({"epoch": global_epoch, "phase": phase_name, "stage": stage_label})
            wandb_logger.log(wandb_payload)

            checkpoint_manager.update(model, optimizer, global_epoch, val_m)
            if val_m["auc"] > best_auc_value:
                best_auc_value = val_m["auc"]
                best_auc_epoch = global_epoch
                best_auc_stage = stage_label

            print(
                f"[{config['experiment_name']}:{stage_label}] stage_epoch {stage_epoch} global_epoch {global_epoch} "
                f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} "
                f"val_auc={val_m['auc']:.4f} event={','.join(event_parts) or '-'}"
            )

            if stage_epoch >= int(stage_cfg.get("min_epochs", 1)) and stage_bad_transition_epochs >= transition_patience:
                print(
                    f"[stage transition] stage={stage_label} monitor={transition_monitor} plateau, "
                    f"selection_source={stage_best_loss_path}"
                )
                break

        selected = None
        tolerance_satisfied = None
        if rollback_to == "best_auc_with_loss_tolerance" and stage_epoch_records:
            selected = select_guarded(stage_epoch_records, stage_best_loss, loss_tolerance, max_allowed_val_loss)
            tolerance_satisfied = True
            reason = "best_auc_within_loss_tolerance"
            if selected and selected["epoch"] == stage_best_loss_epoch:
                reason = "best_val_loss"
            elif selected and selected["epoch"] == stage_best_auc_epoch:
                reason = "best_val_auc"
        elif stage_best_loss_path.exists():
            selected = next((r for r in stage_epoch_records if r["epoch"] == stage_best_loss_epoch), None)
            reason = "best_val_loss"
        else:
            reason = "none"

        if selected and Path(selected["path"]).exists():
            selected_row = next((r for r in stage_epoch_records if r["epoch"] == selected["epoch"]), None)
            if selected_row is None:
                raise RuntimeError(f"Selected transition epoch {selected['epoch']} was not found in stage history.")
            if abs(float(selected["val_loss"]) - float(selected_row["val_loss"])) > 1e-8:
                raise RuntimeError("Selected transition val_loss does not match the in-memory stage history row.")
            if abs(float(selected["val_auc"]) - float(selected_row["val_auc"])) > 1e-8:
                raise RuntimeError("Selected transition val_auc does not match the in-memory stage history row.")
            selected_path = Path(selected["path"])
            transition_path = output_dir / "checkpoints" / f"{phase_name}_stage_{stage_index}_{stage_label}_selected_transition_checkpoint.pth"
            shutil.copy2(selected_path, transition_path)
            metrics_src = selected_path.with_suffix(selected_path.suffix + ".metrics.json")
            if metrics_src.exists():
                shutil.copy2(metrics_src, transition_path.with_suffix(transition_path.suffix + ".metrics.json"))
            state = torch.load(transition_path, map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            last_metrics = state["metrics"]
            selected_model_epoch = state["epoch"]
            model._checkpoint_context = state.get("context", {})
            best_val_loss_stage_checkpoints[stage_label] = str(stage_best_loss_path)
            selected_transition_checkpoints[stage_label] = {
                "path": str(transition_path),
                "epoch": selected["epoch"],
                "val_loss": selected["val_loss"],
                "val_auc": selected["val_auc"],
                "val_pr_auc": selected.get("val_pr_auc"),
                "val_f1": selected.get("val_f1"),
                "reason": reason,
                "selection_rule": rollback_to,
                "loss_tolerance": loss_tolerance if rollback_to == "best_auc_with_loss_tolerance" else None,
                "max_allowed_val_loss": max_allowed_val_loss,
                "tolerance_satisfied": tolerance_satisfied,
                "stage_best_val_loss": stage_best_loss,
                "stage_best_val_loss_epoch": stage_best_loss_epoch,
                "stage_best_val_loss_auc": stage_best_loss_auc,
                "stage_best_val_auc": stage_best_auc,
                "stage_best_val_auc_epoch": stage_best_auc_epoch,
                "stage_best_val_auc_loss": stage_best_auc_loss,
            }
            print(
                "[stage transition] "
                f"stage={stage_label} "
                f"selected_epoch={selected['epoch']} selected_loss={selected['val_loss']:.6f} "
                f"selected_auc={selected['val_auc']:.6f} reason={reason} path={transition_path} "
                f"best_loss={stage_best_loss:.6f}@{stage_best_loss_epoch} best_loss_auc={stage_best_loss_auc:.6f} "
                f"best_auc={stage_best_auc:.6f}@{stage_best_auc_epoch} best_auc_loss={stage_best_auc_loss:.6f} "
                f"selection_rule={rollback_to} loss_tolerance={loss_tolerance if rollback_to == 'best_auc_with_loss_tolerance' else None} "
                f"max_allowed_val_loss={max_allowed_val_loss} "
                f"tolerance_satisfied={tolerance_satisfied} rollback_ok=True"
            )
            rollback_row = {
                "experiment_name": config["experiment_name"],
                "phase": phase_name,
                "stage_name": stage_label,
                "epoch": stage_epoch,
                "global_epoch": global_epoch,
                "unfreeze_event": f"rollback_to_{stage_label}_{reason}",
                "rollback_to_best_val_loss": reason == "best_val_loss",
                "rollback_checkpoint": str(transition_path),
                "selected_transition_epoch": selected["epoch"],
                "selected_transition_val_loss": selected["val_loss"],
                "selected_transition_val_auc": selected["val_auc"],
                "selected_transition_val_pr_auc": selected.get("val_pr_auc"),
                "selected_transition_val_f1": selected.get("val_f1"),
                "selected_transition_reason": reason,
                "transition_selection_rule": rollback_to,
                "transition_loss_tolerance": loss_tolerance if rollback_to == "best_auc_with_loss_tolerance" else None,
                "transition_max_allowed_val_loss": max_allowed_val_loss,
                "transition_tolerance_satisfied": tolerance_satisfied,
                "stage_best_val_loss": stage_best_loss,
                "stage_best_val_loss_epoch": stage_best_loss_epoch,
                "stage_best_val_loss_auc": stage_best_loss_auc,
                "stage_best_val_auc": stage_best_auc,
                "stage_best_val_auc_epoch": stage_best_auc_epoch,
                "stage_best_val_auc_loss": stage_best_auc_loss,
                "transition_sanity_check": "passed",
            }
            history.append(rollback_row)
            keep_paths = {stage_best_loss_path, stage_best_auc_path, transition_path}
            for record in stage_epoch_records:
                path = Path(record["path"])
                if path not in keep_paths and path.exists():
                    path.unlink()
                    metrics_path = path.with_suffix(path.suffix + ".metrics.json")
                    if metrics_path.exists():
                        metrics_path.unlink()

    _, final_unfrozen = frozen_blocks(model)
    conv1_reached = "conv1" in final_unfrozen
    if conv1_reached and final_optimizer is not None:
        save_checkpoint(
            output_dir / "checkpoints" / "final_conv1_completed.pth",
            model,
            final_optimizer,
            selected_model_epoch,
            last_metrics or {},
            effective_config,
        )

    phase_summary = {
        "phase": phase_name,
        "subset_size": train_subset_size,
        "subset_prevalence": train_subset_prev,
        "val_size": full_val_size,
        "val_prevalence": full_val_prev,
        "last_metrics": last_metrics,
        "loss_meta": loss_meta,
        "best_monitor": "val_auc",
        "best_monitor_value": best_auc_value,
        "best_auc_value": best_auc_value,
        "best_auc_epoch": best_auc_epoch,
        "best_auc_stage": best_auc_stage,
        "best_val_loss_stage_checkpoints": best_val_loss_stage_checkpoints,
        "selected_transition_checkpoints": selected_transition_checkpoints,
        "final_stage": "head_only" if not final_unfrozen else "+".join([*final_unfrozen, "head"]),
        "conv1_reached": conv1_reached,
        "conv1_completed": conv1_reached,
        "global_epoch": global_epoch,
    }
    return global_epoch, phase_summary


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
