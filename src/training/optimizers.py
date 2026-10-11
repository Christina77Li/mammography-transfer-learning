"""Trainable blocks, BatchNorm affine policy and optimizer construction."""
from __future__ import annotations

import torch
from src.training.gradual_unfreeze import modules_for_block, set_all_bn_eval
from src.utils.config import bn_policy

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


def apply_bn_affine_policy(model, config):
    _, train_affine = bn_policy(config)
    if not train_affine:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                for parameter in module.parameters():
                    parameter.requires_grad = False


