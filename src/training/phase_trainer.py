"""Ordinary phases with early stopping and optional controller unfreezing."""
from __future__ import annotations

import time
from pathlib import Path
from src.training.checkpointing import CheckpointManager, save_checkpoint, copy_selected_checkpoint
from src.training.checkpoint_policy import select_guarded
from src.training.early_stopping import EarlyStopping
from src.training.gradual_unfreeze import GradualUnfreezeController, freeze_backbone_train_head, frozen_blocks, unfreeze_all
from src.training.metrics import monitor_value, dataset_prevalence, phase_rows_common
from src.training.optimizers import make_optimizer, apply_bn_affine_policy
from src.training.scheduler import build_scheduler, step_scheduler
from src.training.trainer import Trainer

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

    apply_bn_affine_policy(model, config)
    optimizer = make_optimizer(model, config, gradual=gradual)
    trainer = Trainer(model, train_loader, val_loader, config, device)
    loss_meta = trainer.loss_meta
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
        train_m, val_m = trainer.run_epoch(
            optimizer, frozen_bn_blocks=frozen_bn_blocks, label=f"epoch={epoch}"
        )
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
            copy_selected_checkpoint(selected_path, transition_path)
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


