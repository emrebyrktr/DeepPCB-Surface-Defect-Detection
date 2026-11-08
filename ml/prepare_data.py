import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

IMAGE_SIZE = 640
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class OutputRecoveryError(RuntimeError):
    """Publication failed and the original outputs require manual recovery."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        super().__init__(f"Output recovery failed; original backups preserved at {workspace}")


def convert_one_label(src_path: Path, dst_path: Path, img_size: int = IMAGE_SIZE) -> int:
    """Validate pixel boxes and convert them to normalized YOLO labels."""
    if img_size <= 0:
        raise ValueError("Image size must be positive")
    lines_out = []
    for line_number, line in enumerate(src_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        context = f"{src_path}:{line_number}"
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{context}: expected xmin ymin xmax ymax class_id")
        try:
            x_min, y_min, x_max, y_max, class_id = map(int, parts)
        except ValueError as exc:
            raise ValueError(f"{context}: coordinates and class_id must be integers") from exc
        if not 1 <= class_id <= len(CLASS_NAMES):
            raise ValueError(f"{context}: class_id must be between 1 and {len(CLASS_NAMES)}")
        if not (0 <= x_min < x_max <= img_size and 0 <= y_min < y_max <= img_size):
            raise ValueError(f"{context}: box must have positive area within {img_size}x{img_size}")

        x_center = (x_min + x_max) / 2 / img_size
        y_center = (y_min + y_max) / 2 / img_size
        width = (x_max - x_min) / img_size
        height = (y_max - y_min) / img_size
        lines_out.append(
            f"{class_id - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines_out), encoding="utf-8")
    return len(lines_out)


def count_classes(labels_dir: Path) -> Counter:
    counter = Counter()
    for label_file in sorted(labels_dir.glob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counter[int(line.split()[0])] += 1
    return counter


def _raw_pairs(raw_dir: Path, split: str) -> list[tuple[Path, Path]]:
    images_dir = raw_dir / split / "images"
    labels_dir = raw_dir / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"{split}: raw images and labels directories must both exist")

    images = {}
    for image in sorted(images_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            if image.stem in images:
                raise ValueError(f"{split}: multiple images have stem {image.stem!r}")
            images[image.stem] = image
    labels = {path.stem: path for path in sorted(labels_dir.glob("*.txt")) if path.is_file()}
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(
            f"{split}: missing labels for {missing_labels}; missing images for {missing_images}"
        )
    if not images:
        raise ValueError(f"{split}: raw split contains no image-label pairs")
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def _build_processed_split(
    split: str, raw_dir: Path, interim_dir: Path, processed_dir: Path
) -> dict:
    pairs = _raw_pairs(raw_dir, split)
    for image, label in pairs:
        interim_label = interim_dir / split / "labels" / label.name
        convert_one_label(label, interim_label)
        image_target = processed_dir / split / "images" / image.name
        label_target = processed_dir / split / "labels" / label.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_target)
        shutil.copy2(interim_label, label_target)
    return {"images": len(pairs), "labels": len(pairs)}


def _publish_outputs(replacements: list[tuple[Path, Path]], workspace: Path) -> None:
    """Swap only the checked sibling output directories; restore them on failure."""
    backups = []
    published = []
    try:
        for index, (staged, target) in enumerate(replacements):
            if target.parent != workspace.parent or staged.parent != workspace:
                raise ValueError("Output replacement must stay inside the preparation directory")
            if target.is_symlink():
                raise ValueError(f"Output directory cannot be a symlink: {target}")
            if target.exists():
                backup = workspace / f"previous-{index}"
                backups.append((backup, target))
                target.replace(backup)
            published.append((staged, target))
            staged.replace(target)
    except BaseException:
        try:
            for index, (staged, target) in enumerate(reversed(published)):
                if not staged.exists() and target.exists():
                    target.replace(workspace / f"failed-{index}")
            for backup, target in reversed(backups):
                if backup.exists():
                    backup.replace(target)
        except BaseException as recovery_error:
            raise OutputRecoveryError(workspace) from recovery_error
        raise


def prepare_dataset(raw_dir: Path, interim_dir: Path, processed_dir: Path) -> dict:
    """Build fresh outputs, publishing only after every split passes validation."""
    if interim_dir.is_symlink() or processed_dir.is_symlink():
        raise ValueError("Output directories cannot be symlinks")
    raw_dir, interim_dir, processed_dir = (
        Path(path).resolve() for path in (raw_dir, interim_dir, processed_dir)
    )
    if interim_dir == processed_dir or interim_dir.parent != processed_dir.parent:
        raise ValueError("Interim and processed must be distinct sibling directories")
    for output in (interim_dir, processed_dir):
        if raw_dir == output or raw_dir in output.parents or output in raw_dir.parents:
            raise ValueError("Output directories must not overlap raw data")
        if output.exists() and not output.is_dir():
            raise ValueError(f"Output path is not a directory: {output}")
    output_parent = processed_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    workspace = Path(mkdtemp(prefix=".prepare-data-", dir=output_parent)).resolve()
    preserve_workspace = False
    try:
        if workspace.parent != output_parent:
            raise ValueError("Preparation workspace must stay inside the output parent")
        staged_interim = workspace / "interim"
        staged_processed = workspace / "processed"
        total_counts = Counter()
        stats_report = {"splits": {}, "total_class_distribution": {}}
        for split in SPLITS:
            split_info = _build_processed_split(split, raw_dir, staged_interim, staged_processed)
            counts = count_classes(staged_processed / split / "labels")
            total_counts += counts
            stats_report["splits"][split] = {
                "num_images": split_info["images"],
                "class_distribution": {
                    name: counts.get(class_id, 0) for class_id, name in enumerate(CLASS_NAMES)
                },
            }
        stats_report["total_class_distribution"] = {
            name: total_counts.get(class_id, 0) for class_id, name in enumerate(CLASS_NAMES)
        }
        (staged_processed / "data_stats.json").write_text(
            json.dumps(stats_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _publish_outputs(
            [(staged_interim, interim_dir), (staged_processed, processed_dir)], workspace
        )
    except OutputRecoveryError:
        preserve_workspace = True
        raise
    except BaseException as error:
        # Keep any original outputs even if cancellation interrupts transaction bookkeeping.
        preserve_workspace = True
        if any(workspace.glob("previous-*")):
            error.add_note(f"Original output backups preserved at {workspace}")
        else:
            preserve_workspace = False
        raise
    finally:
        if not preserve_workspace:
            if workspace.parent != output_parent or not workspace.name.startswith(".prepare-data-"):
                raise ValueError("Refusing to clean a workspace outside the output parent")
            shutil.rmtree(workspace)
    return stats_report


def main():
    stats = prepare_dataset(RAW_DIR, INTERIM_DIR, PROCESSED_DIR)
    for split, values in stats["splits"].items():
        print(f"[{split.upper()}] {values['num_images']} image-label pairs prepared")
    print(f"[OK] Data statistics saved to {PROCESSED_DIR / 'data_stats.json'}")


if __name__ == "__main__":
    main()
