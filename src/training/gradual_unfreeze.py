from __future__ import annotations

import math

import torch.nn as nn


UNFREEZE_PLAN = [
    {"block": "layer4", "lr": 3e-5, "head_lr": 3e-5, "cooldown": 5},
    {"block": "layer3", "lr": 2e-6, "head_lr": 8e-6, "cooldown": 7},
    {"block": "layer2", "lr": 5e-7, "head_lr": 5e-6, "cooldown": 8},
    {"block": "layer1", "lr": 2e-7, "head_lr": 2e-6, "cooldown": 10},
    {"block": "conv1", "lr": 1e-7, "head_lr": 1e-6, "cooldown": 10},
]

BACKBONE_BLOCKS = ["conv1", "layer1", "layer2", "layer3", "layer4"]


def modules_for_block(model, block: str):
    """Return modules that belong to a named ResNet backbone block."""
    modules = []
    module = getattr(model, block, None)
    if module is not None:
        modules.append(module)
    if block == "conv1":
        bn1 = getattr(model, "bn1", None)
        if bn1 is not None:
            modules.append(bn1)
    return modules


def freeze_backbone_train_head(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    set_frozen_bn_eval(model, BACKBONE_BLOCKS)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def set_frozen_bn_eval(model, frozen_block_names):
    """Keep BatchNorm in frozen backbone blocks from updating statistics."""
    for block in frozen_block_names:
        for block_module in modules_for_block(model, block):
            for module in block_module.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False


def set_all_bn_eval(model):
    """Freeze running statistics; affine trainability is set independently."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def batchnorm_status(model) -> dict:
    total = 0
    eval_count = 0
    train_count = 0
    trainable_params = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            total += 1
            if module.training:
                train_count += 1
            else:
                eval_count += 1
            trainable_params += sum(1 for p in module.parameters() if p.requires_grad)
    return {
        "bn_total": total,
        "bn_eval": eval_count,
        "bn_train": train_count,
        "bn_trainable_param_tensors": trainable_params,
        "bn_all_eval": total == eval_count,
    }




def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def frozen_blocks(model, blocks=None):
    blocks = blocks or BACKBONE_BLOCKS
    frozen = []
    unfrozen = []
    for block in blocks:
        modules = modules_for_block(model, block)
        if not modules:
            continue
        if any(p.requires_grad for module in modules for p in module.parameters()):
            unfrozen.append(block)
        else:
            frozen.append(block)
    return frozen, unfrozen


class GradualUnfreezeController:
    def __init__(
        self,
        model,
        optimizer,
        plan=None,
        monitor="val_loss",
        mode="min",
        patience=3,
        min_delta=1e-3,
        recovery_tolerance=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.plan = plan or UNFREEZE_PLAN
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.recovery_tolerance = None if recovery_tolerance is None else float(recovery_tolerance)
        self.recovery_floor = None
        self.best = None
        self.bad_epochs = 0
        self.cooldown = 0
        self.cooldown_block = None
        self.index = 0
        self.param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        self.ramp = None

    def _improved(self, value):
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value, pre_unfreeze_callback=None):
        if value is None or not math.isfinite(float(value)):
            return ""
        value = float(value)
        if self._improved(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown == 0 and self.cooldown_block:
                block = self.cooldown_block
                self.cooldown_block = None
                return f"{block}_cooldown_complete"
            return ""
        if self.bad_epochs < self.patience or self.index >= len(self.plan):
            return ""
        if self.mode == "max" and self.recovery_floor is not None and value < self.recovery_floor:
            return ""

        if pre_unfreeze_callback is not None:
            pre_unfreeze_callback(self.plan[self.index]["block"])
        event = self.unfreeze_next()
        self.bad_epochs = 0
        return event

    def unfreeze_next(self):
        item = self.plan[self.index]
        block = item["block"]
        modules = modules_for_block(self.model, block)
        if not modules:
            raise AttributeError(f"Model has no block named {block!r}")
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True
            module.train()
        frozen, _ = frozen_blocks(self.model)
        set_frozen_bn_eval(self.model, frozen)

        new_params = []
        for module in modules:
            for p in module.parameters():
                if p.requires_grad and id(p) not in self.param_ids:
                    new_params.append(p)
                    self.param_ids.add(id(p))
        if new_params:
            self.optimizer.add_param_group(
                {
                    "params": new_params,
                    "lr": float(item["lr"]),
                    "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0.0),
                    "name": block,
                }
            )
        for name, lr in item.get("group_lrs", {}).items():
            for group in self.optimizer.param_groups:
                if group.get("name") == name:
                    group["lr"] = float(lr)
        for group in self.optimizer.param_groups:
            if group.get("name") == "head":
                group["lr"] = float(item["head_lr"])
        if not any(group.get("name") == "head" for group in self.optimizer.param_groups):
            self.optimizer.param_groups[0]["lr"] = float(item["head_lr"])
        if self.mode == "max" and self.recovery_tolerance is not None and self.best is not None:
            self.recovery_floor = float(self.best) - self.recovery_tolerance
        self.cooldown = int(item.get("cooldown", 0))
        self.cooldown_block = block if self.cooldown > 0 else None
        self.index += 1
        if item.get("ramp_after_epochs") and item.get("ramp_lrs"):
            self.ramp = {
                "block": block,
                "remaining": int(item["ramp_after_epochs"]),
                "lrs": dict(item["ramp_lrs"]),
            }
        print(
            "[gradual_unfreeze] "
            + ", ".join(f"{group.get('name', f'group_{i}')}={group['lr']}" for i, group in enumerate(self.optimizer.param_groups))
        )
        return f"{block}_unfrozen"

    def maybe_apply_ramp(self):
        if not self.ramp:
            return ""
        if self.ramp["remaining"] > 0:
            self.ramp["remaining"] -= 1
            return ""
        for name, lr in self.ramp["lrs"].items():
            for group in self.optimizer.param_groups:
                if group.get("name") == name:
                    group["lr"] = float(lr)
        block = self.ramp["block"]
        self.ramp = None
        print(
            "[gradual_unfreeze:ramp] "
            + ", ".join(f"{group.get('name', f'group_{i}')}={group['lr']}" for i, group in enumerate(self.optimizer.param_groups))
        )
        return f"{block}_ramp_complete"
