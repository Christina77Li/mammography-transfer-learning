"""Run or preflight an experiment from any working directory."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import preflight
from src.utils.paths import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Validate config, data, cached splits and checkpoint without training.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override YAML device.")
    parser.add_argument("--output-dir", type=Path, help="Use a new output directory; existing results cannot be overwritten.")
    args = parser.parse_args()
    try:
        config_path = args.config if args.config.is_absolute() or args.config.exists() else project_path(args.config)
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if args.device:
            config["device"] = args.device
        if args.output_dir:
            config["output_dir"] = str(args.output_dir)
        config = preflight(config)
        from src.training.experiment_runner import check_runtime, run_experiment
        check_runtime(config)
        if args.check:
            from src.data.dataset import build_dataloaders
            from src.utils.config import effective_phase
            for phase in config["phases"]:
                build_dataloaders(effective_phase(config, phase), create_splits=False)
            # Check model compatibility without downloading pretrained weights.
            checkpoint = config.get("model", {}).get("init_checkpoint") or config.get("init_checkpoint")
            if checkpoint:
                from src.models.resnet import get_model
                from src.training.checkpointing import load_initial_checkpoint
                load_initial_checkpoint(get_model(imagenet_weights=False), checkpoint, "cpu")
            print("Preflight passed; no training or output files created.")
        else:
            run_experiment(config_path, config=config)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
