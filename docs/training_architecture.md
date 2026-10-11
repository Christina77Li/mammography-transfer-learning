# Training architecture

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `experiment_runner.py` | Validate inputs, initialize the model, dispatch phases and assemble the final report. |
| `trainer.py` | Run training and validation with one phase's criterion, BatchNorm and clipping settings. |
| `phase_trainer.py` / `stage_trainer.py` | Execute ordinary or stagewise phases and write their histories. |
| `stage_selector.py` | Resolve per-stage selection settings and track plateau transitions. |
| `checkpoint_policy.py` | Enforce loss guards and choose the reporting phase and monitor. |
| `checkpointing.py` | Save, load, copy and restore matching checkpoint weights, optimizer and metadata. |
| `optimizers.py` / `metrics.py` | Construct parameter groups and produce checked metric/history fields. |

Trainer does not choose checkpoints. Selection rules do not mutate the model. The stage executor applies the selected rollback and starts the next stage with a fresh optimizer and scheduler.

## Execution routes

**Exp1, Exp2a and Exp2b** use ordinary phases. Their configurations set `gradual_unfreeze: false`; all backbone blocks are trainable. Exp2b loads the selected Exp2a weights before full-data training.

**Exp3 and Exp4** use stagewise phases. Each stage declares its trainable blocks and learning rates. A plateau after the minimum stage duration, or the maximum duration, ends the stage. The selected checkpoint is restored before the next stage begins. Exp4's warm-up and head recalibration use the same stage executor with an empty backbone-block list.

`GradualUnfreezeController` is an optional route within ordinary phases, enabled only by `gradual_unfreeze: true`. It adds optimizer groups progressively and can ramp their learning rates without resetting the whole optimizer. None of the five published configurations enables it. Stagewise rollback is a different execution policy: it restores selected weights and explicitly resets optimizer/scheduler state. Keeping the optional controller does not give Exp1–2 a second active unfreezing mechanism.

## Selection and reporting

For guarded selection, an eligible checkpoint satisfies `val_loss ≤ best_loss + tolerance` and any absolute loss cap. Selection takes the highest AUC among eligible checkpoints, breaking AUC ties by lower loss. The tolerance is an absolute loss difference, not a percentage. If no checkpoint qualifies, execution raises an error.

The exact minimum loss is tracked independently of the plateau tracker's `min_delta`. A small loss improvement can therefore update the best-loss checkpoint while still counting as a plateau epoch. Rollback restores the selected model and matching optimizer, epoch, metrics and context; the executed global epoch count remains separate.

The final report uses one checkpoint from the specified phase, defaulting to the last full-data phase. Warm-up metrics cannot win over full-data results merely because their losses use a different criterion. See [checkpoint semantics](config_and_checkpoints.md) for output paths and transfer behavior.
