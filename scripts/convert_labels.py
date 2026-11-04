from pathlib import Path

IMAGE_SIZE = 640
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
SPLITS = ["train", "valid", "test"]

def convert_one_label(src_path: Path, dst_path: Path, img_size: int = IMAGE_SIZE):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []

    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            x_min, y_min, x_max, y_max, class_id = map(int, parts)

            x_center = (x_min + x_max) / 2 / img_size
            y_center = (y_min + y_max) / 2 / img_size
            width    = (x_max - x_min) / img_size
            height   = (y_max - y_min) / img_size
            cls      = class_id - 1  # 1-indexed → 0-indexed

            lines_out.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out))

def main():
    for split in SPLITS:
        src_labels_dir = RAW_DIR / split / "labels"
        dst_labels_dir = INTERIM_DIR / split / "labels"

        label_files = list(src_labels_dir.glob("*.txt"))
        print(f"{split}: {len(label_files)} label dönüştürülüyor...")

        for src_file in label_files:
            dst_file = dst_labels_dir / src_file.name
            convert_one_label(src_file, dst_file)

        print(f"{split}: tamamlandı.")

if __name__ == "__main__":
    main()