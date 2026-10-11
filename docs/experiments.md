# Experiments

Configuration filenames and run identifiers were standardized for readability; the experiment hyperparameters are unchanged by this naming cleanup, and Exp1–Exp4 still identify the experiments reported here. Output/cache paths and the Exp2a → Exp2b initialization path were updated to match the new names.

## Training strategies

| Experiment | Configuration | Strategy |
| --- | --- | --- |
| Exp1 | `configs/exp1_baseline.yaml` | Direct full-data fine-tuning |
| Exp2a | `configs/exp2a_balanced_warmup.yaml` | Full-model warm-up on a positive-enriched subset |
| Exp2b | `configs/exp2b_full_data_transfer.yaml` | Full-data transfer from the selected Exp2a checkpoint |
| Exp3 | `configs/exp3_gradual_unfreezing.yaml` | Gradual unfreezing |
| Exp4 | `configs/exp4_combined_strategy.yaml` | Head warm-up, full-data recalibration and gradual unfreezing |

All configurations use AdamW. Warm-up sampling targets three negative groups per positive group; the resulting image ratio can differ from 1:3. Validation stays fixed throughout training. Transition rules use either minimum validation loss or ROC-AUC within validation-loss bounds; see the configurations for each stage.

Exp1 fine-tunes the full ResNet101 with batch size 128, a maximum of 100 epochs, learning rate 0.001 and weight decay 0.0005. Its ReduceLROnPlateau scheduler uses patience 3 and factor 0.5.

## Data and preprocessing

| Partition | Positive images | Negative images | Total | Positive fraction |
| --- | ---: | ---: | ---: | ---: |
| Training | 936 | 42,792 | 43,728 | 2.14% |
| Validation | 222 | 10,756 | 10,978 | 2.02% |
| Warm-up subset | 936 | 2,767 | 3,703 | 25.28% |

The split groups by patient ID and laterality, matching the breast-specific label and keeping the views of one breast in the same partition. A patient's opposite breast can still enter another partition, so this is breast-group isolation rather than patient isolation; correlations within a patient remain a limitation. A patient-level split is planned for stronger generalization assessment.

Training uses rotation, horizontal flipping and ImageNet normalization; validation uses deterministic transforms. Inputs are resized to 256 × 256 because of compute constraints. Higher-resolution or patch-based inputs are planned to better preserve small-lesion detail. The DICOM converter applies polarity correction, per-image min–max normalization and resizing; it does not apply VOI windowing or modality LUTs.

The validation positive fraction is `222 / 10,978 ≈ 0.02022`, giving a no-skill AP reference of approximately 0.0202. This is a prevalence reference, not an additional trained model.

## Warm-up and transfer checkpoints

| Phase | Checkpoint | Epoch | ROC-AUC | PR-AUC (AP) | Val. loss | Loss criterion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Exp2a | Highest AUC | 7 | 0.6487 | 0.0385 | 0.3016 | BCE |
| Exp2a | Selected transition | 4 | 0.6436 | 0.0347 | 0.2644 | BCE |
| Exp2b | Loss-controlled alternative | 3 | 0.6487 | 0.0390 | 0.4207 | Weighted BCE, w≈6.76 |
| Exp4 warm-up | Highest AUC | 20 | 0.6086 | 0.0322 | 0.3452 | BCE |
| Exp4 warm-up | Selected transition | 17 | 0.6026 | 0.0320 | 0.3284 | BCE |

Validation uses the training criterion for that phase. Ordinary BCE warm-up losses are not directly comparable with weighted full-data losses. The four main checkpoints and the full-data stage checkpoints use approximately the same positive weight (`sqrt(42,792 / 936) ≈ 6.7615`); Exp3 specifies that weight explicitly. CSVs include `val_loss_criterion` and `val_pos_weight`, with effective weight 1 for ordinary BCE.

Exp2b starts from the selected Exp2a transition checkpoint and uses positive-weighted BCE on full data. Exp4 performs head recalibration before gradually unfreezing the backbone. The main results are in `results/reported_metrics.csv`; additional checkpoints are in `results/additional_checkpoints.csv`.

With the default output directory, a successful Exp2a run writes `outputs/exp2a_balanced_warmup/checkpoints/selected_transition_checkpoint.pth` and its `.metrics.json` sidecar. Exp2b's `model.init_checkpoint` points to that file. If no epoch meets Exp2a's loss guard, the run fails without creating an ineligible transition checkpoint. If Exp2a uses a custom `--output-dir`, update Exp2b's initialization path accordingly.

## Gradual unfreezing

![Stage checkpoints for Exp3 and Exp4](../results/figures/stage_checkpoints.png)

Each point is a selected stage checkpoint. The ring marks the highest ROC-AUC checkpoint for that experiment. Later stages can improve PR-AUC without improving ROC-AUC: Exp4's conv1 checkpoint has PR-AUC 0.0550, ROC-AUC 0.6667 and loss 0.4111. Its metrics belong to that checkpoint and are not combined with the layer3 result.

## Baseline training

![Exp1 baseline training curves](../results/figures/exp1_training_curves.png)

Exp1's selected checkpoint is epoch 9. At threshold 0.5, validation accuracy is 0.9798 while recall is zero, showing why accuracy alone is insufficient under severe class imbalance. Thresholds selected on validation data require separate evaluation before deployment.

## Evaluation scope

The results compare single training runs on a fixed validation partition. Validation is used for scheduling, stage transitions and checkpoint selection. Very small rates in early backbone blocks aim to limit changes to pretrained features; absolute loss tolerances define which checkpoints are eligible. These motivations do not establish that the exact rates or tolerances are optimal.

Validation-based development and selection do not replace an untouched test set or external validation. The configurations use `deterministic: false`, so fixed seeds do not guarantee identical reruns. Repeated-seed runs, patient-level separation and independent evaluation are future work.
