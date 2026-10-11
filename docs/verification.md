# Automated validation

GitHub Actions runs CPU validation on pushes to `main` and on pull requests.

```bash
python -m pytest -q
python scripts/smoke_test.py
```

The 45 regression tests cover input identifiers and labels, PNG/CSV consistency, split and subset caches, DICOM polarity, configuration validation, checkpoint selection, stage rollback and final metric consistency. Selection tests check inclusive loss bounds, absolute caps and AUC tie-breaking. Closed-form zero-logit tests verify both ordinary and positive-weighted validation BCE against the phase's training criterion. An Exp2a → Exp2b integration test generates the configured transition filename and checks that transfer loads its measured epoch, metrics and context; it uses synthetic data, a small network and a relaxed synthetic-only loss cap.

The smoke test trains ResNet101 on synthetic images, checks that preflight creates no outputs, reloads the checkpoint and verifies that all reevaluated metrics match within `1e-6`. Synthetic test metrics are not research results. Temporary files are removed unless a new `--work-dir` is provided.

CPU tests do not establish full-dataset or GPU performance. Run input validation and training in the target environment before relying on new outputs.
