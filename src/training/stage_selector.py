"""Stage selection rules and plateau tracking without model or file mutation."""
from __future__ import annotations

from dataclasses import dataclass

from src.training.checkpoint_policy import select_guarded


@dataclass(frozen=True)
class StageSelector:
    selection: str
    loss_tolerance: float = 0.0
    absolute_limit: float | None = None

    @classmethod
    def from_config(cls, transition, stage_name):
        default = transition.get(
            "selection", transition.get("rollback_to", transition.get("default_selection", "best_val_loss"))
        )
        default_tolerance = float(transition.get("loss_tolerance", transition.get("loss_tolerance_from_best_loss", 0.0)))
        stage = transition.get("stage_specific", {}).get(stage_name, {})
        selection = stage.get("selection", default)
        tolerance = float(stage.get("loss_tolerance", stage.get("loss_tolerance_from_best_loss", default_tolerance)))
        positive_default = float(transition.get("loss_tolerance_from_best_loss", 0.0)) > 0.0
        if default == "best_auc_with_loss_tolerance" and positive_default and default_tolerance == 0.0:
            raise ValueError("stage_transition has a positive configured tolerance but resolves to 0.0.")
        positive_stage = float(stage.get("loss_tolerance_from_best_loss", 0.0)) > 0.0
        if selection == "best_auc_with_loss_tolerance" and (positive_stage or (not stage and positive_default)) and tolerance == 0.0:
            raise ValueError(f"Stage {stage_name} has a positive configured tolerance but resolves to 0.0.")
        limit = stage.get("max_allowed_val_loss", transition.get("max_allowed_val_loss"))
        return cls(selection, tolerance, None if limit is None else float(limit))

    def select(self, records, best_loss):
        if not records:
            raise ValueError("Cannot select a checkpoint from an empty stage.")
        if self.selection == "best_auc_with_loss_tolerance":
            return select_guarded(records, best_loss, self.loss_tolerance, self.absolute_limit)
        if self.selection == "best_val_loss":
            return min(records, key=lambda row: row["val_loss"])
        raise ValueError(f"Unknown stage selection: {self.selection}")


class PlateauTracker:
    """Track meaningful improvements separately from the exact minimum loss."""

    def __init__(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError(f"Unknown plateau mode: {mode}")
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad_epochs = 0

    def step(self, value):
        improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
