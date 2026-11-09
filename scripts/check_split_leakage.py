"""Inspect YOLO train/val/test splits for duplicates and similarity candidates.

Checks:
  1. exact duplicate files (MD5)
  2. similar images across splits (dHash + Hamming distance)
  3. val/test label coordinates shared with a train label file

Image and label similarity require manual review. Exit codes: 0 for no exact
cross-split file duplicates, 1 for exact duplicates, 2 for missing or empty splits.

Usage:
    python scripts/check_split_leakage.py --root data/processed
"""
import argparse
import csv
import hashlib
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("split_leakage")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def find_splits(root: Path) -> dict:
    """Return {split: (images_dir, labels_dir)} for either common layout."""
    splits = {}
    for name, canon in SPLIT_NAMES.items():
        if canon in splits:
            continue
        candidates = (
            (root / name / "images", root / name / "labels"),
            (root / "images" / name, root / "labels" / name),
        )
        for img_dir, lbl_dir in candidates:
            if img_dir.is_dir():
                splits[canon] = (img_dir, lbl_dir)
                break
    return splits


def list_images(directory: Path) -> list:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(path: Path, size: int = 16) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten().astype(np.float32)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two sets of 0/1 vectors."""
    return a @ (1 - b).T + (1 - a) @ b.T


def label_signature(path: Path) -> frozenset:
    if not path.exists():
        return frozenset()
    boxes = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls = int(float(parts[0]))
            boxes.add((cls, *(round(float(v), 3) for v in parts[1:5])))
    return frozenset(boxes)


def save_pair(a: Path, b: Path, out: Path, height: int = 320) -> None:
    images = []
    for p in (a, b):
        img = Image.open(p).convert("RGB")
        images.append(img.resize((int(img.width * height / img.height), height)))
    canvas = Image.new("RGB", (sum(i.width for i in images) + 10, height), "white")
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + 10
    canvas.save(out)


def check_exact_duplicates(files: dict) -> int:
    logger.info("[1] Exact duplicates (MD5)")
    seen, dups = {}, []
    for split, paths in files.items():
        for p in paths:
            h = file_md5(p)
            if h in seen and seen[h][0] != split:
                dups.append((seen[h], (split, p)))
            seen.setdefault(h, (split, p))
    logger.info("    cross-split duplicates: %d", len(dups))
    for (s1, p1), (s2, p2) in dups[:10]:
        logger.info("    %s/%s == %s/%s", s1, p1.name, s2, p2.name)
    return len(dups)


def check_near_duplicates(files: dict, args) -> None:
    logger.info("[2] Image similarity candidates (dHash, 256 bit)")
    hashes = {s: np.stack([dhash(p) for p in paths]) for s, paths in files.items()}
    train = hashes["train"]

    if len(train) > 1:
        within = hamming_matrix(train, train)
        np.fill_diagonal(within, 1e9)
        nn_within = within.min(axis=1)
        logger.info("    train->train nearest: min=%d p5=%d median=%d",
                    nn_within.min(), np.percentile(nn_within, 5), np.median(nn_within))
    else:
        logger.info("    train->train nearest: skipped (one image)")

    rows = []
    val_test_rows = []
    comparisons = [(split, "train") for split in hashes if split != "train"]
    if "val" in hashes and "test" in hashes:
        comparisons.append(("val", "test"))

    for split, reference in comparisons:
        dist = hamming_matrix(hashes[split], hashes[reference])
        nn, idx = dist.min(axis=1), dist.argmin(axis=1)
        close = np.where(nn <= args.near_thresh)[0]
        logger.info("    %s->%s nearest: min=%d p5=%d median=%d | <=%d bits: %d/%d",
                    split, reference, nn.min(), np.percentile(nn, 5), np.median(nn),
                    args.near_thresh, len(close), len(nn))
        for i in close:
            pair = (files[split][i].name, files[reference][idx[i]].name, int(nn[i]))
            if reference == "train":
                rows.append((split, *pair))
            else:
                val_test_rows.append(pair)
        prefix = split if reference == "train" else f"{split}_{reference}"
        for rank, i in enumerate(close[np.argsort(nn[close])[:3]]):
            save_pair(files[split][i], files[reference][idx[i]],
                      args.out / f"near_{prefix}_{rank}_d{int(nn[i])}.png")

    rows.sort(key=lambda r: r[3])
    with open(args.out / "near_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "file", "train_neighbor", "hamming"])
        writer.writerows(rows)
    logger.info("    %d pairs written to %s", len(rows), args.out / "near_duplicates.csv")

    if "val" in hashes and "test" in hashes:
        val_test_rows.sort(key=lambda r: r[2])
        val_test_path = args.out / "val_test_near_duplicates.csv"
        with val_test_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["val_file", "test_file", "hamming"])
            writer.writerows(val_test_rows)
        logger.info("    %d pairs written to %s", len(val_test_rows), val_test_path)


def check_label_overlap(files: dict, splits: dict, args) -> int:
    logger.info("[3] Label-coordinate overlap candidates with train (manual review)")
    signatures = {}
    box_index = defaultdict(set)
    for split, paths in files.items():
        for p in paths:
            sig = label_signature(splits[split][1] / f"{p.stem}.txt")
            signatures[(split, p.name)] = sig
            for box in sig:
                box_index[box].add((split, p.name))

    hits = []
    for (split, name), sig in signatures.items():
        if split == "train" or not sig:
            continue
        shared = defaultdict(int)
        for box in sig:
            for other_split, other in box_index[box]:
                if other_split == "train":
                    shared[other] += 1
        for other, k in shared.items():
            jaccard = k / (len(sig) + len(signatures[("train", other)]) - k)
            if k >= args.min_shared_boxes and jaccard >= args.jaccard:
                hits.append((jaccard, split, name, other))
    hits.sort(reverse=True)
    logger.info("    label-overlap candidates: %d", len(hits))
    for jaccard, split, name, other in hits[:10]:
        logger.info("    %s/%s ~ train/%s (jaccard=%.2f)", split, name, other, jaccard)
    return len(hits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("reports/split_leakage"))
    parser.add_argument("--near-thresh", type=int, default=20)
    parser.add_argument("--min-shared-boxes", type=int, default=2)
    parser.add_argument("--jaccard", type=float, default=0.5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    splits = find_splits(args.root)
    if "train" not in splits or len(splits) < 2:
        logger.error("Could not find train plus at least one other split under %s", args.root)
        return 2
    files = {s: list_images(img_dir) for s, (img_dir, _) in splits.items()}
    for split, paths in files.items():
        logger.info("%-5s %d images", split, len(paths))
    empty_splits = [split for split, paths in files.items() if not paths]
    if empty_splits:
        logger.error("No images found in split(s): %s", ", ".join(empty_splits))
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    hard = check_exact_duplicates(files)
    check_near_duplicates(files, args)
    check_label_overlap(files, splits, args)

    # Similarity candidates require review; only exact file duplicates fail the run.
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
