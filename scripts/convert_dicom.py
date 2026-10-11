"""Export a DICOM tree as normalized grayscale PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_pixels(pixels: np.ndarray, photometric: str) -> np.ndarray:
    """Apply polarity correction and min–max normalization."""
    data = np.asarray(pixels, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("Expected a non-empty, single-frame grayscale image.")
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f"Unsupported photometric interpretation: {photometric}")
    if not np.isfinite(data).all():
        raise ValueError("Pixel array contains non-finite values.")
    if photometric == "MONOCHROME1":
        data = data.max() - data
    data -= data.min()
    maximum = data.max()
    if maximum == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    return (data / maximum * 255).astype(np.uint8)


def convert_directory(source: Path, destination: Path, size: int, overwrite: bool = False) -> tuple[int, int]:
    import pydicom

    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory not found: {source}")
    if size <= 0:
        raise ValueError("Image size must be positive.")
    if source == destination or source in destination.parents:
        raise ValueError("Output must be outside the input directory tree.")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".dcm", ".dicom", ""})
    if not files:
        raise ValueError("No .dcm, .dicom, or extensionless DICOM candidates found.")
    output_paths = [destination / p.relative_to(source).with_suffix(".png") for p in files]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Multiple inputs map to the same PNG path.")
    converted, skipped = 0, 0
    for path, output in zip(files, output_paths):
        if output.exists() and not overwrite:
            skipped += 1
            continue
        dataset = pydicom.dcmread(path)
        pixels = normalize_pixels(dataset.pixel_array, str(dataset.PhotometricInterpretation))
        image = Image.fromarray(pixels).resize((size, size), resample=Image.Resampling.BICUBIC)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        converted += 1
    return converted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing PNGs; otherwise skip them.")
    args = parser.parse_args()
    try:
        converted, skipped = convert_directory(args.input, args.output, args.size, args.overwrite)
    except (ValueError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(f"Converted {converted} images; skipped {skipped} existing PNGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
