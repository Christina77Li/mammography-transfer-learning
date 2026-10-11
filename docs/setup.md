# Setup

## CPU environment

Python 3.12.14; torch 2.10.0+cpu; torchvision 0.25.0+cpu. Other direct dependencies are pinned in `requirements.txt`; test dependencies are in `requirements-dev.txt`. 

On Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test.py
```

To retain synthetic test evidence, pass a new directory:

```bash
.venv/bin/python scripts/smoke_test.py --work-dir outputs/cpu_smoke_001
```

It includes synthetic inputs, checkpoints and `smoke_result.json`; it must not be reused for another run. ResNet101 checkpoints are large even in a one-epoch smoke test.

## Windows and CUDA

First run these CPU checks in Windows PowerShell (activation is unnecessary):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

For a compatible NVIDIA GPU, install the CUDA 12.6 wheels:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe scripts/train.py --config configs/exp3_gradual_unfreezing.yaml --check
```

Confirm CUDA availability and run preflight against your data and cached splits. `imagenet_weights: true` downloads torchvision's ResNet101 V2 weights unless they are already cached; the smoke test sets this false. `model.init_checkpoint` transfers model weights with strict architecture checking and creates a new optimizer/scheduler; it does not resume the exact training process.

Official wheel commands: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

## DICOM decoder support

The tests decode generated, uncompressed DICOMs with pydicom. Depending on the transfer syntax, compressed files need an additional decoder. JPEG 2000 commonly uses:

```powershell
.\.venv\Scripts\python.exe -m pip install pylibjpeg pylibjpeg-openjpeg
```

The converter applies MONOCHROME1 inversion, per-image min–max scaling to uint8 and bicubic resize. It applies no VOI window, modality LUT or new scientific preprocessing.

Official decoder guide: [pydicom decoder plugins](https://pydicom.github.io/pydicom/stable/guides/decoding/decoder_plugins.html).

## Research inputs and preflight

The default configuration expects:

```text
input/train.csv
input/images_as_pngs_512/train_images_processed_512/<patient>/<image_id>.png
input/rsna_split_seed42.json
```

The CSV must contain `patient_id`, `image_id`, `laterality` (L/R) and `cancer` (0/1). One PNG stem must match each unique CSV image ID, with no extra or duplicate PNG IDs. Set the subset-cache paths in Exp2a/Exp4. Training creates reproducible new caches when absent; `--check` only validates and does not create them.

For a first data check, use:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/exp3_gradual_unfreezing.yaml --device cpu --check
```

Then use a new output directory for each execution. Exp2b needs the selected Exp2a checkpoint; adjust the initialization path in a configuration copy if Exp2a's output directory changes. Run outputs and weights are excluded from Git.

W&B integration is disabled by default. Enabling it additionally requires the `wandb` package and account configuration; CPU tests run without W&B.
