from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def load_rsna_dataframe(label_csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(label_csv_path, dtype={"patient_id": "string", "image_id": "string", "laterality": "string"})
    required = ["patient_id", "image_id", "laterality", "cancer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv is missing required columns: {missing}")
    df = df[required].copy()
    if df.empty or df.isna().any().any():
        raise ValueError("Label CSV must be nonempty and contain no missing IDs, laterality or labels.")
    for key in ("patient_id", "image_id", "laterality"):
        if df[key].str.strip().eq("").any() or not df[key].str.strip().eq(df[key]).all():
            raise ValueError(f"Invalid whitespace in {key}.")
        df[key] = df[key].astype(str)
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in label CSV.")
    if not df["laterality"].isin(["L", "R"]).all():
        raise ValueError("laterality must be L or R.")
    if not df["cancer"].isin([0, 1]).all():
        raise ValueError("cancer must contain only binary labels 0 and 1.")
    df["cancer"] = df["cancer"].astype(int)
    return df


def build_group_index(df: pd.DataFrame, stem_to_index: dict[str, int]):
    group_to_indices: dict[str, list[int]] = {}
    group_labels: dict[str, int] = {}
    missing = 0

    for (pid, laterality), group in df.groupby(["patient_id", "laterality"], sort=False):
        indices = []
        for image_id in group["image_id"].values:
            idx = stem_to_index.get(str(image_id))
            if idx is None:
                missing += 1
            else:
                indices.append(idx)
        if indices:
            key = f"{pid}_{laterality}"
            group_to_indices[key] = indices
            group_labels[key] = int(group["cancer"].max())

    return group_to_indices, group_labels, missing


def get_or_create_group_split(
    group_to_indices: dict[str, list[int]],
    group_labels: dict[str, int],
    split_path: str | Path,
    val_pct: float,
    seed: int,
    create_cache: bool = True,
):
    if not 0 < val_pct < 1:
        raise ValueError("val_pct must be between 0 and 1.")
    split_path = Path(split_path)
    if create_cache:
        split_path.parent.mkdir(parents=True, exist_ok=True)

    all_group_keys = set(group_to_indices.keys())
    pos_groups = sorted(g for g, label in group_labels.items() if label == 1)
    neg_groups = sorted(g for g, label in group_labels.items() if label == 0)

    if len(pos_groups) < 2 or len(neg_groups) < 2:
        raise ValueError("Split requires at least two positive and two negative breast groups.")
    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_list = split.get("train_group_keys", [])
        val_list = split.get("val_group_keys", [])
        train_group_keys, val_group_keys = set(train_list), set(val_list)
        if len(train_list) != len(train_group_keys) or len(val_list) != len(val_group_keys):
            raise ValueError("Cached split contains duplicate groups.")
        if train_group_keys & val_group_keys:
            raise ValueError("Cached split leaks groups between train and validation.")
        if (train_group_keys | val_group_keys) != all_group_keys:
            raise ValueError("Cached split does not exactly cover current CSV/PNG groups. Use the matching data or a new split path.")
        if split.get("seed") != seed or split.get("val_pct") != val_pct:
            raise ValueError("Cached split seed/val_pct differs from configuration.")
        source = "loaded"
    else:
        rng = random.Random(seed)
        rng.shuffle(pos_groups)
        rng.shuffle(neg_groups)
        n_val_pos = max(1, int(round(len(pos_groups) * val_pct))) if pos_groups else 0
        n_val_neg = max(1, int(round(len(neg_groups) * val_pct))) if neg_groups else 0
        if len(pos_groups) > 1:
            n_val_pos = min(n_val_pos, len(pos_groups) - 1)
        if len(neg_groups) > 1:
            n_val_neg = min(n_val_neg, len(neg_groups) - 1)
        val_group_keys = set(pos_groups[:n_val_pos] + neg_groups[:n_val_neg])
        train_group_keys = all_group_keys - val_group_keys
        source = "created"

    train_indices = indices_for_groups(train_group_keys, group_to_indices)
    val_indices = indices_for_groups(val_group_keys, group_to_indices)
    if set(train_indices) & set(val_indices):
        raise ValueError("Image indices overlap between train and validation.")
    for keys in (train_group_keys, val_group_keys):
        if not keys or {group_labels[g] for g in keys} != {0, 1}:
            raise ValueError("Train and validation must each contain positive and negative groups.")

    if source == "created" and create_cache:
        split_path.write_text(json.dumps({"seed": seed, "val_pct": val_pct, "train_group_keys": sorted(train_group_keys), "val_group_keys": sorted(val_group_keys)}, indent=2))

    meta = {
        "split_path": str(split_path),
        "split_source": source,
        "train_group_count": len(train_group_keys),
        "val_group_count": len(val_group_keys),
        "pos_group_count": len(pos_groups),
        "neg_group_count": len(neg_groups),
    }
    return train_indices, val_indices, train_group_keys, val_group_keys, meta


def get_or_create_small_subset(
    train_group_keys: set[str],
    val_group_keys: set[str],
    group_to_indices: dict[str, list[int]],
    group_labels: dict[str, int],
    split_path: str | Path,
    pos_neg_ratio: float,
    seed: int,
    pos_fraction: float = 1.0,
    max_pos_groups: int | None = None,
    create_cache: bool = True,
):
    split_path = Path(split_path)
    if create_cache:
        split_path.parent.mkdir(parents=True, exist_ok=True)
    if not pos_neg_ratio > 0 or not 0 < pos_fraction <= 1 or (max_pos_groups is not None and max_pos_groups < 1):
        raise ValueError("Invalid subset ratio, positive fraction or positive group limit.")
    train_pos = sorted(g for g in train_group_keys if group_labels[g] == 1)
    train_neg = sorted(g for g in train_group_keys if group_labels[g] == 0)

    if split_path.exists():
        info = json.loads(split_path.read_text())
        cached_groups = set(info.get("small_group_keys", []))
        leaked = cached_groups & val_group_keys
        stale = cached_groups - train_group_keys
        cached_ratio = float(info.get("pos_neg_ratio", pos_neg_ratio))
        cached_fraction = float(info.get("pos_fraction", 1.0))
        cached_max_pos = info.get("max_pos_groups")
        current_pos = sum(group_labels[g] == 1 for g in cached_groups if g in train_group_keys)
        current_neg = sum(group_labels[g] == 0 for g in cached_groups if g in train_group_keys)
        current_indices = indices_for_groups(cached_groups, group_to_indices)
        if (
            info.get("chosen_pos_groups") != current_pos
            or info.get("chosen_neg_groups") != current_neg
            or info.get("small_num_samples") != len(current_indices)
            or current_pos == 0 or current_neg == 0
            or info.get("seed") != seed
            or len(info.get("small_group_keys", [])) != len(cached_groups)
            or not cached_groups
            or leaked
            or stale
            or abs(cached_ratio - float(pos_neg_ratio)) > 1e-12
            or abs(cached_fraction - float(pos_fraction)) > 1e-12
            or cached_max_pos != max_pos_groups
        ):
            raise ValueError("Cached subset disagrees with split/configuration; use a matching cache or a new subset path.")
        else:
            small_groups = cached_groups
            return indices_for_groups(small_groups, group_to_indices), info

    if not train_pos or not train_neg:
        raise ValueError("Cannot create small subset because both train classes are required.")

    rng = random.Random(seed + 999)
    pos_shuffled = train_pos.copy()
    rng.shuffle(pos_shuffled)
    target_pos = max(1, int(round(len(pos_shuffled) * float(pos_fraction))))
    if max_pos_groups is not None:
        target_pos = min(target_pos, int(max_pos_groups))
    target_pos = min(target_pos, len(pos_shuffled))
    chosen_pos = pos_shuffled[:target_pos]

    target_neg = int(round(len(chosen_pos) * float(pos_neg_ratio)))
    target_neg = max(1, min(target_neg, len(train_neg))) if train_neg else 0
    neg_shuffled = train_neg.copy()
    rng.shuffle(neg_shuffled)
    small_groups = sorted(chosen_pos + neg_shuffled[:target_neg])
    small_indices = indices_for_groups(set(small_groups), group_to_indices)
    info = {
        "seed": seed,
        "pos_neg_ratio": float(pos_neg_ratio),
        "pos_fraction": float(pos_fraction),
        "max_pos_groups": max_pos_groups,
        "train_pos_groups": len(train_pos),
        "train_neg_groups": len(train_neg),
        "chosen_pos_groups": len(chosen_pos),
        "chosen_neg_groups": target_neg,
        "small_group_keys": small_groups,
        "small_num_samples": len(small_indices),
    }
    if create_cache:
        split_path.write_text(json.dumps(info, indent=2))
    return small_indices, info


def indices_for_groups(group_keys, group_to_indices: dict[str, list[int]]) -> list[int]:
    result = set()
    for key in group_keys:
        result.update(group_to_indices.get(key, []))
    return sorted(result)
