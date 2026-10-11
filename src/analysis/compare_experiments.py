from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compare(outputs):
    outputs = Path(outputs)
    figures = outputs / "comparison_figures"
    figures.mkdir(parents=True, exist_ok=True)
    histories = sorted(outputs.glob("*/history.csv"))
    if not histories:
        print(f"No history.csv files found under {outputs}")
        return

    for metric, ylabel, filename in [
        ("val_loss", "Validation loss", "comparison_val_loss.png"),
        ("val_auc", "Validation AUC", "comparison_val_auc.png"),
        ("val_acc", "Validation accuracy", "comparison_val_accuracy.png"),
    ]:
        plt.figure(figsize=(9, 5))
        for history in histories:
            df = pd.read_csv(history)
            if metric in df.columns:
                plt.plot(df["global_epoch"], df[metric], label=history.parent.name)
        plt.xlabel("Global epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    compare(args.outputs)


if __name__ == "__main__":
    main()
