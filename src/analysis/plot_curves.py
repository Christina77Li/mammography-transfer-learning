from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NO_ANNOTATION_EXPERIMENTS = {
    "exp2a_subset_warmup_1to3_auc_loss_guard",
    "exp2b_from_exp2a_selected_epoch4",
}

UNFREEZE_STAGE_ONLY_EXPERIMENTS = {
    "exp3m_late_stage_auc_tolerance_rollback",
    "exp4_1to3_head_warmup_loss_guard_fixed_tolerance",
}

UNFREEZE_STAGE_NAMES = {"layer4", "layer3", "layer2", "layer1", "conv1"}


def _experiment_name(df):
    if "experiment_name" not in df.columns or df.empty:
        return ""
    values = [v for v in df["experiment_name"].dropna().astype(str).unique().tolist() if v]
    return values[0] if values else ""


def _event_rows(df):
    events = df["unfreeze_event"].fillna("") if "unfreeze_event" in df.columns else pd.Series([""] * len(df))
    return df[events != ""]


def _unfreeze_stage_rows(df):
    if "stage_name" not in df.columns or "global_epoch" not in df.columns:
        return pd.DataFrame()
    rows = df[df["stage_name"].fillna("").isin(UNFREEZE_STAGE_NAMES)].copy()
    if rows.empty:
        return rows
    rows = rows.dropna(subset=["global_epoch"])
    rows = rows.loc[rows["stage_name"].ne(rows["stage_name"].shift())]
    return rows


def _annotate_unfreeze_stages(df, y_top):
    for _, row in _unfreeze_stage_rows(df).iterrows():
        x = row["global_epoch"]
        stage = str(row["stage_name"])
        label = f"epoch {int(x)}: {stage}"
        plt.axvline(x, linestyle="--", linewidth=1, color="gray", alpha=0.75)
        plt.text(x, y_top, label, rotation=90, va="top", ha="right", fontsize=8, color="gray")


def _annotate_events(df, y_top):
    for _, row in _event_rows(df).iterrows():
        x = row["global_epoch"]
        event = str(row["unfreeze_event"])
        label = f"unfreeze {event}" if event.startswith(("conv", "layer")) else event
        plt.axvline(x, linestyle="--", linewidth=1, color="gray", alpha=0.75)
        plt.text(x, y_top, label, rotation=90, va="top", ha="right", fontsize=8, color="gray")


def _annotate_phase_boundaries(df, y_top):
    if "phase" not in df.columns:
        return
    phase_starts = df.groupby("phase", sort=False)["global_epoch"].min()
    for phase, x in phase_starts.iloc[1:].items():
        plt.axvline(x, linestyle="--", linewidth=1.2, color="black", alpha=0.65)
        label = f"transfer to {phase}"
        plt.text(x, y_top, label, rotation=90, va="top", ha="left", fontsize=8, color="black")


def _plot_metric(df, output_path, columns, ylabel, title, labels=None):
    present = [c for c in columns if c in df.columns]
    if not present:
        return
    experiment_name = _experiment_name(df)
    plt.figure(figsize=(9, 5))
    for col in present:
        plt.plot(
            df["global_epoch"],
            df[col],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=(labels or {}).get(col, col),
        )
    values = df[present].apply(pd.to_numeric, errors="coerce")
    y_top = float(values.max().max()) if not values.empty else 1.0
    if experiment_name in UNFREEZE_STAGE_ONLY_EXPERIMENTS:
        _annotate_unfreeze_stages(df, y_top)
    elif experiment_name not in NO_ANNOTATION_EXPERIMENTS:
        _annotate_events(df, y_top)
        _annotate_phase_boundaries(df, y_top)
    plt.xlabel("Global epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _lr_label_map(df, lr_cols):
    labels = {}
    event_names = _event_rows(df)["unfreeze_event"].dropna().astype(str).tolist()
    for col in lr_cols:
        name_col = f"{col}_name"
        if name_col in df.columns:
            names = [n for n in df[name_col].dropna().astype(str).unique().tolist() if n]
            if names:
                labels[col] = names[-1]
                continue
        idx = int(col.replace("lr_group_", ""))
        if idx == 0:
            labels[col] = "head/fc" if event_names else "all trainable"
        elif idx - 1 < len(event_names):
            labels[col] = event_names[idx - 1]
        else:
            labels[col] = f"group {idx}"
    return labels


def plot_experiment_curves(history_csv, figures_dir):
    history_csv = Path(history_csv)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not history_csv.exists():
        return
    df = pd.read_csv(history_csv)
    if df.empty:
        return
    _plot_metric(df, figures_dir / "loss_curve.png", ["train_loss", "val_loss"], "Loss", "Loss curve")
    _plot_metric(df, figures_dir / "auc_curve.png", ["val_auc"], "AUC", "Validation AUC")
    _plot_metric(df, figures_dir / "accuracy_curve.png", ["train_acc", "val_acc"], "Accuracy", "Accuracy curve")
    lr_cols = [c for c in df.columns if c.startswith("lr_group_") and not c.endswith("_name")]
    _plot_metric(
        df,
        figures_dir / "lr_curve.png",
        lr_cols,
        "Learning rate",
        "Learning rate by parameter group",
        labels=_lr_label_map(df, lr_cols),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--figures", required=True)
    args = parser.parse_args()
    plot_experiment_curves(args.history, args.figures)


if __name__ == "__main__":
    main()
