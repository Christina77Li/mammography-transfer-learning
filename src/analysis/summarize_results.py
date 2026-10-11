from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(outputs):
    outputs = Path(outputs)
    rows = []
    for metrics_path in sorted(outputs.glob("*/final_metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "experiment": metrics.get("experiment", metrics_path.parent.name),
                "best_epoch": metrics.get("best_epoch"),
                "best_monitor": metrics.get("best_monitor"),
                "val_loss": metrics.get("loss"),
                "acc@0.5": metrics.get("acc@0.5"),
                "recall@0.5": metrics.get("recall@0.5"),
                "f1@0.5": metrics.get("f1@0.5"),
                "auc": metrics.get("auc"),
                "pr_auc": metrics.get("pr_auc"),
                "sensitivity": metrics.get("sensitivity@0.5"),
                "specificity": metrics.get("specificity@0.5"),
                "best_f1": metrics.get("best_f1"),
                "best_threshold": metrics.get("best_th"),
                "training_time_total_min": metrics.get("training_time_total_min"),
            }
        )
    df = pd.DataFrame(rows)
    outputs.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs / "summary_results.csv", index=False)
    try:
        markdown = df.to_markdown(index=False) if not df.empty else ""
    except Exception:
        markdown = df.to_csv(index=False)
    (outputs / "summary_results.md").write_text(markdown)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    summarize(args.outputs)


if __name__ == "__main__":
    main()
