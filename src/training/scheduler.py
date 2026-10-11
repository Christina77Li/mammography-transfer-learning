from __future__ import annotations

from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(config: dict, optimizer):
    cfg = config.get("scheduler", {})
    if cfg.get("type", "reduce_on_plateau") in {None, "none"}:
        return None
    if cfg.get("type", "reduce_on_plateau") != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler type: {cfg.get('type')}")
    return ReduceLROnPlateau(
        optimizer,
        mode=cfg.get("mode", "min"),
        factor=float(cfg.get("factor", 0.5)),
        patience=int(cfg.get("patience", 3)),
        min_lr=float(cfg.get("min_lr", 1e-8)),
        threshold=float(cfg.get("threshold", 1e-4)),
        cooldown=int(cfg.get("cooldown", 0)),
    )


def step_scheduler(scheduler, monitor_value):
    if scheduler is not None:
        scheduler.step(monitor_value)
