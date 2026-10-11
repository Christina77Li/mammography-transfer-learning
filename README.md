# Mammography Classification with Transfer Learning

A comparison of four fine-tuning strategies for ImageNet-pretrained **ResNet101** on the highly imbalanced RSNA screening mammography dataset. This project explores how early class distribution and progressive backbone adaptation affect image-level cancer classification.

For a visual walkthrough, [view the research presentation (PDF)](docs/slides/research_presentation.pdf).

## Results

Each row reports ROC-AUC, average precision and validation loss from the same checkpoint.

| Experiment | Training strategy | ROC-AUC ↑ | PR-AUC (AP) ↑ | Val. loss ↓ |
| --- | --- | ---: | ---: | ---: |
| Exp1 | Baseline: direct full-data fine-tuning | 0.5780 | 0.0242 | 0.4115 |
| Exp2b | 1:3 subset warm-up → full-data transfer | 0.6591 | 0.0426 | 0.4341 |
| Exp3 | Gradual unfreezing | 0.6601 | **0.0486** | 0.4062 |
| Exp4 | Warm-up → head recalibration → gradual unfreezing | **0.6711** | 0.0482 | **0.3980** |

![Validation results by training strategy](results/figures/strategy_comparison.png)

The combined strategy achieved the highest ROC-AUC and lowest loss among these checkpoints. Gradual unfreezing alone had the highest PR-AUC. The best ROC-AUC stages were **layer2 for Exp3** and **layer3 for Exp4**. See [experiment details](docs/experiments.md) for the training stages and additional checkpoints.

## Training design

Warm-up uses a positive-enriched subset sampled from training groups. Exp2a updates the whole model; Exp4's warm-up updates the head only. Full-data validation remains fixed. Gradual unfreezing trains the head first, then releases `layer4 → layer3 → layer2 → layer1 → conv1`, with smaller learning rates for earlier blocks and selected-checkpoint rollback at stage transitions.

The training pipeline includes AdamW, phase-specific BCE/positive-weighted BCE, plateau scheduling, fixed BatchNorm statistics and loss-constrained checkpoint selection. It validates CSV/PNG consistency and cached splits before training, and records matching checkpoint weights, epochs and metrics.

## Quick start

**Python 3.12 · CPU** — create a virtual environment, then run:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke_test.py
```

The smoke test trains ResNet101 on synthetic PNGs for one epoch, reloads its checkpoint and verifies the metrics. It needs no dataset or pretrained-weight download. [Setup instructions](docs/setup.md) include Windows commands, CUDA installation and input preparation.

To train with your dataset:

```bash
python scripts/train.py --config configs/exp1_baseline.yaml --check
python scripts/train.py --config configs/exp1_baseline.yaml --output-dir outputs/exp1_new_run
```

Exp2–Exp4 configurations request CUDA; add `--device cpu` when needed. Exp2b requires the selected Exp2a checkpoint. Paths resolve from the project root; use a new output directory for each run.

## Evaluation and project structure

Results are single-run validation measurements without confidence intervals, significance testing or external validation. The split groups by **patient ID + laterality**, keeping views of one breast together; it does not guarantee that both breasts of every patient share a partition. These results do not establish clinical screening performance.

| Path | Purpose |
| --- | --- |
| `configs/` | Experiment configurations |
| `src/` | Data loading, model, training and evaluation |
| `scripts/` | Training, DICOM conversion, smoke test and plotting |
| `tests/` | Data, configuration and checkpoint regression tests |
| `results/` | Experiment metrics and figures |
| `docs/` | Setup, experiments and checkpoint behavior |

Regenerate figures with `python scripts/plot_reported_results.py`. Datasets, weights and run outputs are excluded from Git. [Checkpoint documentation](docs/config_and_checkpoints.md) describes selection and weight transfer; [test documentation](docs/verification.md) explains automated validation.

**Author:** Ruijing Li · Sophia University
