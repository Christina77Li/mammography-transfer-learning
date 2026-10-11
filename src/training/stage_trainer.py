"""Stagewise execution with selected-checkpoint rollback."""
from __future__ import annotations

import time
from pathlib import Path
from src.training.checkpointing import (
    CheckpointManager, save_checkpoint, copy_selected_checkpoint,
    restore_selected_checkpoint, prune_epoch_checkpoints,
)
from src.training.gradual_unfreeze import frozen_blocks
from src.training.metrics import dataset_prevalence, monitor_value, phase_rows_common
from src.training.optimizers import configure_stage_trainability, make_stage_optimizer, apply_bn_affine_policy
from src.training.scheduler import build_scheduler, step_scheduler
from src.training.stage_selector import StageSelector, PlateauTracker
from src.training.trainer import Trainer
from src.utils.config import effective_phase, bn_policy

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
    trainer = Trainer(model, train_loader, val_loader, effective_config, device)
    loss_meta = trainer.loss_meta
    checkpoint_manager = checkpoint_manager or CheckpointManager(output_dir / "phases" / phase_name, config=effective_config)
    transition_cfg = effective_config.get("stage_transition", {})
    transition_patience = int(transition_cfg.get("patience", 3))
    transition_min_delta = float(transition_cfg.get("min_delta", 0.001))
    transition_monitor = transition_cfg.get("monitor", "val_loss")
    transition_mode = transition_cfg.get("mode", "min")
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
        selector = StageSelector.from_config(transition_cfg, stage_label)
        rollback_to = selector.selection
        loss_tolerance = selector.loss_tolerance
        max_allowed_val_loss = selector.absolute_limit
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
        plateau = PlateauTracker(mode=transition_mode, min_delta=transition_min_delta)
        stage_best_loss_path = output_dir / "checkpoints" / f"{phase_name}_stage_{stage_index}_{stage_label}_best_val_loss.pth"
        stage_best_auc_path = output_dir / "checkpoints" / f"{phase_name}_stage_{stage_index}_{stage_label}_best_val_auc.pth"
        stage_epoch_records = []

        for stage_epoch in range(1, int(stage_cfg.get("max_epochs", 1)) + 1):
            start = time.time()
            train_m, val_m = trainer.run_epoch(
                optimizer, label=f"stage={stage_label} epoch={stage_epoch}"
            )
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
            plateau.step(transition_value)

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
            row["stage_transition_counter"] = plateau.bad_epochs
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

            if stage_epoch >= int(stage_cfg.get("min_epochs", 1)) and plateau.bad_epochs >= transition_patience:
                print(
                    f"[stage transition] stage={stage_label} monitor={transition_monitor} plateau, "
                    f"selection_source={stage_best_loss_path}"
                )
                break

        selected = selector.select(stage_epoch_records, stage_best_loss)
        tolerance_satisfied = True if rollback_to == "best_auc_with_loss_tolerance" else None
        reason = "best_auc_within_loss_tolerance"
        if selected["epoch"] == stage_best_loss_epoch:
            reason = "best_val_loss"
        elif selected["epoch"] == stage_best_auc_epoch:
            reason = "best_val_auc"

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
            copy_selected_checkpoint(selected_path, transition_path)
            selected_model_epoch, last_metrics = restore_selected_checkpoint(
                transition_path, model, optimizer, device
            )
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
            prune_epoch_checkpoints(stage_epoch_records, keep_paths)

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


