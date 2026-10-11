"""Configuration validation shared by the CLI and Python entry point."""
from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path

from src.utils.paths import project_path

MONITORS = {"val_loss": "min", "val_auc": "max", "val_f1": "max", "val_f1@0.5": "max"}
BLOCKS = {"conv1", "layer1", "layer2", "layer3", "layer4"}
RULES = {"best_val_loss", "best_auc_with_loss_tolerance"}


def positive(value, key, integer=False, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number.")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be an integer.")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} must be {'nonnegative' if allow_zero else 'positive'}.")


def effective_phase(config, phase):
    result = copy.deepcopy(config)
    for key in ("loss", "optimizer", "scheduler", "stage_transition", "stagewise_stages", "early_stopping", "training", "batch_norm", "data"):
        if key in phase:
            if isinstance(phase[key], dict):
                result[key] = {**result.get(key, {}), **phase[key]}
            else:
                result[key] = copy.deepcopy(phase[key])
    return result


def resolve_config(config):
    config = copy.deepcopy(config)
    for owner in [config, *config.get("phases", [])]:
        data = owner.get("data", {})
        for key in ("image_dir", "label_csv_path", "split_path"):
            if data.get(key):
                data[key] = str(project_path(data[key]).resolve())
        small = data.get("small_subset", {})
        if small.get("split_path"):
            small["split_path"] = str(project_path(small["split_path"]).resolve())
    config["output_dir"] = str(project_path(config.get("output_dir", f"outputs/{config['experiment_name']}" )).resolve())
    for owner in (config, config.get("model", {})):
        if owner.get("init_checkpoint"):
            owner["init_checkpoint"] = str(project_path(owner["init_checkpoint"]).resolve())
    return config


def bn_policy(config):
    bn = config.get("batch_norm", {})
    freeze = bool(config.get("training", {}).get("freeze_all_batchnorm", False))
    freeze = freeze or bool(bn.get("freeze_running_stats", False)) or bool(bn.get("keep_bn_eval_during_training", False))
    return freeze, bool(bn.get("train_bn_affine", not freeze))


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(config.get("experiment_name", ""))):
        raise ValueError("experiment_name must be a nonempty filename-safe name.")
    for key in ("data", "model", "training", "optimizer", "loss", "scheduler", "early_stopping", "batch_norm", "save_best", "report"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{key} must be a mapping.")
    positive(config.get("seed", 42), "seed", integer=True, allow_zero=True)
    device = config.get("device", "auto")
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be auto, cpu, cuda or cuda:<index>.")
    if config.get("model", {}).get("name", "resnet101") != "resnet101" or config.get("model", {}).get("num_classes", 1) != 1:
        raise ValueError("This project supports binary ResNet101 only.")
    if config.get("model", {}).get("architecture", "resnet101") != "resnet101":
        raise ValueError("model.architecture conflicts with ResNet101.")
    for key, value in config.get("save", {}).items():
        if key not in {"save_best_val_loss", "save_best_val_auc", "save_selected_transition", "save_final", "save_history_csv"} or value is not True:
            raise ValueError("save settings describe required outputs and must be true; disabling them is unsupported.")
    phases = config.get("phases")
    if not isinstance(phases, list) or not phases or any(not isinstance(p, dict) for p in phases):
        raise ValueError("phases must be a nonempty list of mappings.")
    check_booleans(config)
    names = [p.get("name") for p in phases]
    if len(names) != len(set(names)) or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Phase names must be unique and filename-safe.")
    report = config.get("report", {})
    if report.get("phase") and report["phase"] not in names:
        raise ValueError("report.phase is not a configured phase.")
    for p in phases:
        cfg = effective_phase(config, p)
        check_booleans(cfg)
        data = cfg.get("data", {})
        if not data.get("image_dir") or not data.get("label_csv_path"):
            raise ValueError("data.image_dir and data.label_csv_path are required.")
        positive(data.get("image_size", 256), "image_size", integer=True)
        positive(cfg.get("training", {}).get("batch_size", data.get("batch_size", 128)), "batch_size", integer=True)
        positive(data.get("num_workers", 4), "num_workers", integer=True, allow_zero=True)
        if not isinstance(data.get("val_pct", .2), (int, float)) or not 0 < data.get("val_pct", .2) < 1:
            raise ValueError("val_pct must be between zero and one.")
        small = data.get("small_subset", {})
        if small.get("enabled"):
            positive(small.get("pos_neg_ratio", 1), "pos_neg_ratio")
            if not 0 < small.get("pos_fraction", 1) <= 1:
                raise ValueError("pos_fraction must be in (0, 1].")
            if small.get("max_pos_groups") is not None:
                positive(small["max_pos_groups"], "max_pos_groups", integer=True)
        if p.get("loader", "train") not in {"train", "small"} or (p.get("loader") == "small" and not small.get("enabled")):
            raise ValueError(f"Unavailable train loader for phase {p['name']}.")
        opt = cfg.get("optimizer", {})
        if str(opt.get("type", "adam")).lower() not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer.type must be adam, adamw or sgd.")
        for key in ("lr", "head_lr", "backbone_lr", "weight_decay"):
            if key in opt:
                positive(opt[key], f"optimizer.{key}", allow_zero=(key == "weight_decay"))
        clip = cfg.get("training", {}).get("gradient_clipping", {})
        if clip.get("enabled"):
            positive(clip.get("max_norm", 1), "gradient_clipping.max_norm")
        if str(cfg.get("loss", {}).get("type", "bce")).replace("-", "_") not in {"bce", "bce_pos", "focal", "focal_pos"}:
            raise ValueError("Unsupported loss.type.")
        if cfg.get("loss", {}).get("pos_weight") is not None:
            positive(cfg["loss"]["pos_weight"], "pos_weight")
        for key in ("scheduler", "early_stopping", "save_best", "stage_transition"):
            settings = cfg.get(key, {})
            if key == "scheduler" and settings.get("type", "reduce_on_plateau") not in {None, "none", "reduce_on_plateau"}:
                raise ValueError("Unsupported scheduler.type.")
            monitor = settings.get("monitor", "val_loss")
            mode = settings.get("mode", MONITORS.get(monitor))
            if monitor not in MONITORS or mode != MONITORS[monitor]:
                raise ValueError(f"Invalid {key} monitor/mode: {monitor}/{mode}.")
            for field in ("patience", "start_epoch", "cooldown", "min_delta", "threshold", "min_lr"):
                if field in settings:
                    positive(settings[field], f"{key}.{field}", integer=field in {"patience", "start_epoch", "cooldown"}, allow_zero=True)
            if key == "scheduler" and not 0 < settings.get("factor", .5) < 1:
                raise ValueError("scheduler.factor must be in (0, 1).")
        if p.get("stagewise_rollback"):
            transition = cfg.get("stage_transition", {})
            for key in ("reset_optimizer_on_new_stage", "reset_scheduler_on_new_stage"):
                if transition.get(key, True) is not True:
                    raise ValueError(f"{key}=false is unsupported; stage transitions reset optimizer and scheduler.")
            stages = cfg.get("stagewise_stages", [])
            if not stages or any(not isinstance(s, dict) for s in stages):
                raise ValueError("stagewise_stages must be a nonempty list of mappings.")
            stage_names = [s.get("name") for s in stages]
            if len(stage_names) != len(set(stage_names)) or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in stage_names):
                raise ValueError("Stage names must be unique and filename-safe.")
            if set(transition.get("stage_specific", {})) - set(stage_names):
                raise ValueError("stage_specific references an unknown stage.")
            for stage in stages:
                positive(stage.get("max_epochs", 1), "stage.max_epochs", integer=True)
                positive(stage.get("min_epochs", 1), "stage.min_epochs", integer=True)
                if stage.get("min_epochs", 1) > stage.get("max_epochs", 1):
                    raise ValueError("min_epochs exceeds max_epochs.")
                blocks = stage.get("trainable_blocks", [])
                if len(blocks) != len(set(blocks)) or set(blocks) - BLOCKS:
                    raise ValueError("Invalid or duplicate trainable_blocks.")
                positive(stage.get("head_lr", opt.get("lr", .001)), "stage.head_lr")
                for block in blocks:
                    positive(stage.get(f"{block}_lr"), f"stage.{block}_lr")
                selection = {**transition, **transition.get("stage_specific", {}).get(stage["name"], {})}
                rule = selection.get("selection", transition.get("rollback_to", transition.get("default_selection", "best_val_loss")))
                validate_selector(selection, rule)
        else:
            positive(p.get("epochs", cfg.get("training", {}).get("epochs", 1)), "phase.epochs", integer=True)
        selector = p.get("checkpoint_selection", config.get("checkpoint_selection", {}))
        if selector:
            validate_selector(selector, selector.get("selection", "best_val_loss"))
    monitor = report.get("monitor", config.get("save_best", {}).get("monitor", config.get("early_stopping", {}).get("monitor", "val_loss")))
    if monitor not in MONITORS:
        raise ValueError("Unsupported report.monitor.")


def validate_selector(settings, rule):
    if rule not in RULES:
        raise ValueError(f"Unsupported checkpoint selection: {rule}.")
    for key in ("loss_tolerance", "loss_tolerance_from_best_loss", "max_allowed_val_loss"):
        if key in settings:
            positive(settings[key], key, allow_zero=True)
    if "loss_tolerance" in settings and "loss_tolerance_from_best_loss" in settings and settings["loss_tolerance"] != settings["loss_tolerance_from_best_loss"]:
        raise ValueError("Conflicting loss tolerance aliases.")
    for key, expected in (("monitor_auc", "val_auc"), ("monitor_loss", "val_loss")):
        if settings.get(key, expected) != expected:
            raise ValueError(f"{key} must be {expected}.")
    if settings.get("on_no_eligible", "error") != "error":
        raise ValueError("Only on_no_eligible=error is supported; loss guards are mandatory.")


def check_booleans(config):
    fields = {"": ["deterministic"], "training": ["freeze_all_batchnorm"], "batch_norm": ["freeze_running_stats", "keep_bn_eval_during_training", "train_bn_affine"], "model": ["imagenet_weights"], "data": ["use_weighted_sampler"], "loss": ["sqrt_pos_weight"], "logging": ["use_wandb"]}
    for owner, keys in fields.items():
        values = config if not owner else config.get(owner, {})
        if not isinstance(values, dict):
            raise ValueError(f"{owner} must be a mapping.")
        for key in keys:
            if key in values and not isinstance(values[key], bool):
                raise ValueError(f"{owner}.{key} must be a YAML boolean.")
    for phase in config.get("phases", []):
        if isinstance(phase, dict):
            for key in ("stagewise_rollback", "freeze_backbone", "gradual_unfreeze", "save_final_conv1_completed"):
                if key in phase and not isinstance(phase[key], bool):
                    raise ValueError(f"phase.{key} must be a YAML boolean.")
    for owner in [config.get("data", {}).get("small_subset", {}), config.get("training", {}).get("gradient_clipping", {})]:
        if not isinstance(owner, dict) or ("enabled" in owner and not isinstance(owner["enabled"], bool)):
            raise ValueError("enabled must be a YAML boolean in subset/clipping settings.")


def preflight(config):
    """Validate paths and exact CSV/PNG identities, without creating split/output files."""
    from src.data.splits import load_rsna_dataframe
    validate_config(config)
    config = resolve_config(config)
    output = Path(config["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK):
        raise ValueError(f"Output parent is not writable: {ancestor}")
    seen = set()
    for p in config["phases"]:
        cfg = effective_phase(config, p)
        data = cfg["data"]
        pair = (data["label_csv_path"], data["image_dir"])
        if pair in seen:
            continue
        seen.add(pair)
        df = load_rsna_dataframe(pair[0])
        images = sorted(Path(pair[1]).rglob("*.png"))
        stems = [image.stem for image in images]
        if not stems or len(stems) != len(set(stems)):
            raise ValueError("PNG directory is empty or contains duplicate stems.")
        missing, extra = set(df.image_id) - set(stems), set(stems) - set(df.image_id)
        if missing or extra:
            raise ValueError(f"CSV/PNG mismatch: {len(missing)} missing PNGs, {len(extra)} unlabeled PNGs.")
    checkpoint = config.get("model", {}).get("init_checkpoint") or config.get("init_checkpoint")
    if checkpoint and not Path(checkpoint).is_file():
        raise ValueError(f"Initial checkpoint does not exist: {checkpoint}")
    return config
