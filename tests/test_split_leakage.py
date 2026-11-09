import csv
import logging

import numpy as np
import pytest
from PIL import Image

from scripts import check_split_leakage as checker


def write_image(root, split, name, pixels):
    directory = root / split / "images"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.fromarray(pixels).save(path)
    return path


def noise(seed):
    return np.random.default_rng(seed).integers(0, 256, size=(32, 32), dtype=np.uint8)


def test_label_matches_are_review_candidates(tmp_path, caplog):
    root = tmp_path / "data"
    output = tmp_path / "report"
    for split, seed in [("train", 1), ("valid", 2)]:
        image = write_image(root, split, f"{split}_board.png", noise(seed))
        labels = root / split / "labels"
        labels.mkdir()
        (labels / f"{image.stem}.txt").write_text(
            "0 0.2 0.2 0.1 0.1\n1 0.7 0.7 0.1 0.1\n", encoding="utf-8",
        )

    caplog.set_level(logging.INFO, logger="split_leakage")
    result = checker.main(["--root", str(root), "--out", str(output)])

    assert "label-overlap candidates: 1" in caplog.text
    assert result == 0


def test_exact_duplicates_fail_the_check(tmp_path):
    root = tmp_path / "data"
    train = write_image(root, "train", "train.png", noise(1))
    valid = write_image(root, "valid", "valid.png", noise(2))
    valid.write_bytes(train.read_bytes())

    assert checker.main(["--root", str(root), "--out", str(tmp_path / "report")]) == 1


def test_val_test_similarity_is_reported_separately(tmp_path):
    root = tmp_path / "data"
    output = tmp_path / "report"
    gradient = np.tile(np.arange(64, dtype=np.uint8) * 2, (64, 1))
    write_image(root, "train", "train.png", gradient[:, ::-1])
    val = write_image(root, "valid", "val.png", gradient)
    test = write_image(root, "test", "test.png", gradient + 20)
    assert val.read_bytes() != test.read_bytes()

    assert checker.main([
        "--root", str(root), "--out", str(output), "--near-thresh", "0",
    ]) == 0

    with (output / "near_duplicates.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["split", "file", "train_neighbor", "hamming"]
        assert list(reader) == []
    with (output / "val_test_near_duplicates.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{
            "val_file": "val.png", "test_file": "test.png", "hamming": "0",
        }]
    assert (output / "near_val_test_0_d0.png").exists()


def test_train_similarity_report_keeps_its_format(tmp_path):
    root = tmp_path / "data"
    output = tmp_path / "report"
    gradient = np.tile(np.arange(64, dtype=np.uint8) * 2, (64, 1))
    write_image(root, "train", "train.png", gradient)
    write_image(root, "valid", "val.png", gradient + 20)

    assert checker.main([
        "--root", str(root), "--out", str(output), "--near-thresh", "0",
    ]) == 0

    with (output / "near_duplicates.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{
            "split": "val", "file": "val.png", "train_neighbor": "train.png", "hamming": "0",
        }]
    assert (output / "near_val_0_d0.png").exists()
    assert not (output / "val_test_near_duplicates.csv").exists()


@pytest.mark.parametrize("empty_split", ["train", "valid"])
def test_empty_split_returns_clear_input_error(tmp_path, caplog, empty_split):
    root = tmp_path / "data"
    output = tmp_path / "report"
    for split in ["train", "valid"]:
        (root / split / "images").mkdir(parents=True)
        if split != empty_split:
            write_image(root, split, "board.png", noise(1))

    caplog.set_level(logging.ERROR, logger="split_leakage")
    assert checker.main(["--root", str(root), "--out", str(output)]) == 2
    assert "No images found in split(s)" in caplog.text
    assert not output.exists()


@pytest.mark.parametrize("splits", [("train",), ("valid", "test")])
def test_missing_required_splits_return_input_error(tmp_path, splits):
    root = tmp_path / "data"
    output = tmp_path / "report"
    for seed, split in enumerate(splits):
        write_image(root, split, "board.png", noise(seed))

    assert checker.main(["--root", str(root), "--out", str(output)]) == 2
    assert not output.exists()


def test_image_listing_excludes_directories(tmp_path):
    image = write_image(tmp_path, "train", "board.png", noise(1))
    (image.parent / "directory.png").mkdir()

    assert checker.list_images(image.parent) == [image]
