from __future__ import annotations

import math


class EarlyStopping:
    def __init__(self, monitor="val_loss", mode="min", patience=10, min_delta=1e-3, start_epoch=10):
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.start_epoch = int(start_epoch)
        self.best_score = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_best = False

    def _improved(self, metric):
        if self.best_score is None:
            return True
        if self.mode == "max":
            return metric > self.best_score + self.min_delta
        return metric < self.best_score - self.min_delta

    def step(self, metric, epoch: int) -> bool:
        self.save_best = False
        if epoch < self.start_epoch or metric is None or not math.isfinite(float(metric)):
            return False
        metric = float(metric)
        if self._improved(metric):
            self.best_score = metric
            self.num_bad_epochs = 0
            self.save_best = True
        else:
            self.num_bad_epochs += 1
        self.early_stop = self.num_bad_epochs >= self.patience
        return self.early_stop
