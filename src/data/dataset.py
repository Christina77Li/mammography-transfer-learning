from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.splits import (
    build_group_index,
    get_or_create_group_split,
    get_or_create_small_subset,
    load_rsna_dataframe,
)
from src.data.transforms import build_transforms


class CancerDataset(Dataset):
    """RSNA PNG image dataset keyed by image_id filename stem."""

    def __init__(self, image_dir: str | Path, label_dict: dict[str, int], transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob("**/*.png"))
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.label_dict[img_path.stem]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class TransformingSubset(Dataset):
    """Subset wrapper that applies its own transform to a raw base dataset."""

    def __init__(self, indices, transform, base: CancerDataset):
        self.idx = list(indices)
        self.transform = transform
        self.base = base

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img, y = self.base[self.idx[i]]
        return self.transform(img), y


def labels_for_indices(indices, base: CancerDataset) -> np.ndarray:
    labels = [base.label_dict[base.image_files[idx].stem] for idx in indices]
    return np.asarray(labels, dtype=int)


def labels_from_dataset(dataset) -> np.ndarray:
    if isinstance(dataset, TransformingSubset):
        return labels_for_indices(dataset.idx, dataset.base)
    if isinstance(dataset, CancerDataset):
        return labels_for_indices(range(len(dataset)), dataset)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_labels = labels_from_dataset(dataset.dataset)
        return base_labels[np.asarray(dataset.indices)]
    raise TypeError(f"Cannot infer labels from dataset type {type(dataset)!r}")


def _make_loader(dataset, batch_size, num_workers, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _weighted_sampler(labels: np.ndarray):
    class_counts = np.bincount(labels, minlength=2).astype(float)
    class_counts[class_counts == 0] = 1.0
    weights = (1.0 / class_counts)[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def build_dataloaders(config: dict, create_splits=True):
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    seed = int(config.get("seed", 42))
    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(train_cfg.get("batch_size", data_cfg.get("batch_size", 128)))
    num_workers = int(data_cfg.get("num_workers", 4))

    df = load_rsna_dataframe(data_cfg["label_csv_path"])
    label_dict = dict(zip(df["image_id"], df["cancer"]))
    base = CancerDataset(data_cfg["image_dir"], label_dict, transform=None)
    stems = [p.stem for p in base.image_files]
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("PNG directory is empty or has duplicate image_id stems.")
    csv_ids, png_ids = set(label_dict), set(stems)
    if csv_ids != png_ids:
        raise ValueError(f"CSV/PNG mismatch: {len(csv_ids - png_ids)} missing PNGs; {len(png_ids - csv_ids)} unlabeled PNGs.")
    stem_to_index = {p.stem: i for i, p in enumerate(base.image_files)}
    group_to_indices, group_labels, missing = build_group_index(df, stem_to_index)

    split_path = data_cfg.get("split_path")
    if not split_path:
        split_path = str(Path(data_cfg["label_csv_path"]).with_name(f"rsna_split_seed{seed}.json"))

    train_indices, val_indices, train_groups, val_groups, split_meta = get_or_create_group_split(
        group_to_indices,
        group_labels,
        split_path=split_path,
        val_pct=float(data_cfg.get("val_pct", 0.2)),
        seed=seed,
        create_cache=create_splits,
    )

    train_tf, val_tf = build_transforms(image_size)
    train_ds = TransformingSubset(train_indices, train_tf, base)
    val_ds = TransformingSubset(val_indices, val_tf, base)

    train_labels = labels_for_indices(train_indices, base)
    val_labels = labels_for_indices(val_indices, base)
    for name, labels in (("train", train_labels), ("validation", val_labels)):
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"{name} must contain both binary classes.")
    use_weighted = bool(data_cfg.get("use_weighted_sampler", False))
    sampler = _weighted_sampler(train_labels) if use_weighted else None
    train_loader = _make_loader(train_ds, batch_size, num_workers, shuffle=not use_weighted, sampler=sampler)
    val_loader = _make_loader(val_ds, batch_size, num_workers, shuffle=False)

    small_loader = None
    small_meta = None
    small_cfg = data_cfg.get("small_subset", {})
    if bool(small_cfg.get("enabled", False)):
        ratio = float(small_cfg.get("pos_neg_ratio", 1.0))
        small_split_path = small_cfg.get("split_path")
        if not small_split_path:
            safe_ratio = str(ratio).replace(".", "p")
            safe_fraction = str(float(small_cfg.get("pos_fraction", 1.0))).replace(".", "p")
            small_split_path = str(
                Path(data_cfg["label_csv_path"]).with_name(
                    f"rsna_small_subset_pos1_neg{safe_ratio}_posfrac{safe_fraction}_seed{seed}.json"
                )
            )
        small_indices, small_meta = get_or_create_small_subset(
            train_groups,
            val_groups,
            group_to_indices,
            group_labels,
            split_path=small_split_path,
            pos_neg_ratio=ratio,
            seed=seed,
            create_cache=create_splits,
            pos_fraction=float(small_cfg.get("pos_fraction", 1.0)),
            max_pos_groups=small_cfg.get("max_pos_groups"),
        )
        small_ds = TransformingSubset(small_indices, train_tf, base)
        small_labels = labels_for_indices(small_indices, base)
        small_loader = _make_loader(small_ds, batch_size, num_workers, shuffle=True)
        small_meta["prevalence"] = float(small_labels.mean()) if len(small_labels) else 0.0
        small_meta["positive_image_ids"] = [
            base.image_files[idx].stem for idx, label in zip(small_indices, small_labels) if int(label) == 1
        ]
        small_meta["negative_image_ids"] = [
            base.image_files[idx].stem for idx, label in zip(small_indices, small_labels) if int(label) == 0
        ]


    meta = {
        **split_meta,
        "image_dir": str(data_cfg["image_dir"]),
        "label_csv_path": str(data_cfg["label_csv_path"]),
        "missing_csv_images": int(missing),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "weighted_sampler": use_weighted,
        "train_n": int(len(train_labels)),
        "val_n": int(len(val_labels)),
        "train_prevalence": float(train_labels.mean()) if len(train_labels) else 0.0,
        "val_prevalence": float(val_labels.mean()) if len(val_labels) else 0.0,
        "small_subset": small_meta,
    }
    print(
        f"[data] train={meta['train_n']} prev={meta['train_prevalence']:.4f} | "
        f"val={meta['val_n']} prev={meta['val_prevalence']:.4f} | split={meta['split_path']}"
    )
    if small_meta:
        print(
            f"[data] small_subset={small_meta['small_num_samples']} "
            f"ratio=1:{small_meta['pos_neg_ratio']} prev={small_meta['prevalence']:.4f}"
        )

    return {"train": train_loader, "val": val_loader, "small": small_loader, "metadata": meta}
