from __future__ import annotations

import os


class WandbLogger:
    def __init__(self, config: dict, experiment_name: str):
        self.run = None
        logging_cfg = config.get("logging", {})
        if not logging_cfg.get("use_wandb", False):
            return
        os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=logging_cfg.get("project", "breast-cancer-detection"),
            name=experiment_name,
            config=config,
            resume=logging_cfg.get("resume", "allow"),
        )

    def log(self, data: dict):
        if self.run is not None:
            self.wandb.log(data)

    def finish(self):
        if self.run is not None:
            self.wandb.finish()
