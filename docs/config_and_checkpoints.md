# Configuration and checkpoint semantics

## Effective settings

All experiment configurations use AdamW.

| Setting | Runtime behavior |
| --- | --- |
| `device` | Explicit `cpu` / `cuda[:index]`; `auto` chooses an available device. CLI `--device` overrides it. |
| `data.*` and phase `data.*` | Project-root paths; full CSV/PNG identity and cached split validation. |
| `optimizer.type` | Adam, AdamW or SGD; unknown names fail. |
| phase loss/optimizer/scheduler/training/BN overrides | Applied to that phase before constructing the criterion and optimizer. |
| `training.freeze_all_batchnorm` | Freeze BN running statistics; affine parameters default frozen in this case. |
| `batch_norm.freeze_running_stats` / `keep_bn_eval_during_training` | Keep BN in evaluation mode after `model.train()`. |
| `batch_norm.train_bn_affine` | Independently enables/disables affine optimization in trainable blocks. Parameters in frozen blocks remain frozen. |
| `early_stopping.*` | Stops ordinary phases; its monitor is distinct from final reporting. |
| stage `min_epochs/max_epochs` and `stage_transition.*` | Bound stages and control plateau transitions. True minimum loss tracking does not use `min_delta`. |
| `reset_optimizer_on_new_stage/reset_scheduler_on_new_stage` | True is supported; false fails explicitly. Each stage begins with fresh optimizer/scheduler state. |
| `checkpoint_selection` / stage selection | Select highest AUC within the relative and optional absolute loss bounds. No eligible checkpoint means an error, with no ineligible fallback. |
| `save_best.monitor` | Chooses the final reported best checkpoint unless `report.monitor` overrides it. |
| `report.phase/monitor` | Reporting only; defaults to the last full-data phase and an explicitly resolved metric. |
| `save.*` | Required output declarations. True values are supported; false or unknown declarations fail. |
| phase `source_*`, `checkpoint_reference`, model `architecture` | Descriptive fields; they do not override measured checkpoint metadata. Architecture must match ResNet101. |
| top-level `training.min_epochs/max_epochs` | Top-level descriptive bounds; selected ordinary phases use `phase.epochs` and `early_stopping.start_epoch`. Stagewise phases use stage bounds. |

Extra metadata remains in `config_used.yaml`. The validator checks supported execution settings; arbitrary extra metadata is not an additional executable option. Controller and focal-loss support are optional features outside the configured experiment routes. [Training architecture](training_architecture.md) explains why the optional controller and stagewise rollback have different behavior and which routes the configurations use.

## Checkpoint metadata and selection

A checkpoint contains model state, optimizer state, measured metrics, configuration, epoch and phase/stage context. The adjacent `.metrics.json` repeats its epoch, metrics and context. The payload is authoritative when loading; a raw state dictionary is also supported, with optional sidecar metadata. Missing or unexpected model keys fail under strict loading.

`initialization.json` records loaded checkpoint metadata separately from descriptive configuration fields. History rows use the measured epoch and metrics from the loaded checkpoint.

Per-phase best files live in `phases/<phase>/checkpoints/`. `final_metrics.json` reads one of those sidecars and carries the exact path and matching phase/stage/epoch. Warmup results remain in their own phase summaries. Full-data loss divergence flags use only the reported phase.

After stage rollback, the selected model and matching optimizer are restored. `last_metrics` refers to this restored model. `final_conv1_completed.pth` uses the selected model epoch, which can be earlier than the number of executed epochs. The summary's `global_epoch` remains the number of executed epochs. Optimizer restoration makes the saved payload internally consistent; the next stage still resets it.

Checkpoints do not contain scheduler state or all RNG states and advertise `resume_supported: false`. They support evaluation and model-weight transfer, not exact process resume.

## Split/cache compatibility

Grouping uses `patient_id + laterality`. New candidate lists are sorted before seeded shuffling. When a cache is absent, training creates a reproducible split/subset from the configured seed. Existing valid caches are reused without rewriting. Invalid caches are never deleted or regenerated silently.

Cached splits must have unique disjoint train/validation groups, exactly cover current groups and match seed/validation fraction. Both datasets must contain both classes. Cached subsets must match the requested seed, ratio, fraction, group limit and train membership. Cache validation checks membership and sampling settings; it does not fingerprint every image.

Nonempty output directories fail before writes. A failed run retains diagnostic output, including `run_failed.json`; retry with a new directory after resolving its cause. Use a new output directory for each run.
