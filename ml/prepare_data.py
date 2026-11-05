import json
import shutil
from pathlib import Path
from collections import Counter

IMAGE_SIZE = 640
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Koordinatların [0.0, 1.0] aralığı dışına taşmasını engeller."""
    return max(min_val, min(val, max_val))


def convert_one_label(src_path: Path, dst_path: Path, img_size: int = IMAGE_SIZE) -> int:
    """Pascal VOC piksel formatını YOLO normalize formata çevirir."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            x_min, y_min, x_max, y_max, class_id = map(int, parts)

            # Geçersiz kutuları (genişliği veya yüksekliği <= 0 olanları) atla
            if x_max <= x_min or y_max <= y_min:
                continue

            x_center = clamp((x_min + x_max) / 2 / img_size)
            y_center = clamp((y_min + y_max) / 2 / img_size)
            width    = clamp((x_max - x_min) / img_size)
            height   = clamp((y_max - y_min) / img_size)
            cls      = class_id - 1  # 1-indexed -> 0-indexed

            if 0 <= cls < len(CLASS_NAMES):
                lines_out.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    return len(lines_out)


def count_classes(labels_dir: Path) -> Counter:
    """Belirtilen klasördeki etiketlerin sınıf dağılımını sayar."""
    counter = Counter()
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counter[cls_id] += 1
    return counter


def build_processed_split(split: str) -> dict:
    """
    1. raw/{split}/labels -> interim/{split}/labels (YOLO dönüşümü)
    2. raw/{split}/images ve interim/{split}/labels -> processed/{split}/ (Eşleştirme ve Kopyalama)
    """
    src_labels_dir = RAW_DIR / split / "labels"
    src_images_dir = RAW_DIR / split / "images"
    interim_labels_dir = INTERIM_DIR / split / "labels"

    proc_labels_dir = PROCESSED_DIR / split / "labels"
    proc_images_dir = PROCESSED_DIR / split / "images"
    proc_labels_dir.mkdir(parents=True, exist_ok=True)
    proc_images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Aşama: Etiketleri dönüştür ve interim'e yaz
    label_files = list(src_labels_dir.glob("*.txt"))
    print(f"[{split.upper()}] {len(label_files)} label dönüştürülüyor...")

    for src_file in label_files:
        interim_file = interim_labels_dir / src_file.name
        convert_one_label(src_file, interim_file)

    # 2. Aşama: Görselleri ve dönüştürülmüş etiketleri processed klasörüne taşı
    copied_images = 0
    for img_file in src_images_dir.glob("*.jpg"):
        corresponding_label = interim_labels_dir / f"{img_file.stem}.txt"
        if corresponding_label.exists():
            shutil.copy2(img_file, proc_images_dir / img_file.name)
            shutil.copy2(corresponding_label, proc_labels_dir / corresponding_label.name)
            copied_images += 1

    print(f"[{split.upper()}] {copied_images} görsel-etiket çifti 'data/processed/{split}' altına kopyalandı.")
    return {"images": copied_images, "labels": len(label_files)}


def main():
    total_counts = Counter()
    stats_report = {"splits": {}, "total_class_distribution": {}}

    for split in SPLITS:
        split_info = build_processed_split(split)
        counts = count_classes(PROCESSED_DIR / split / "labels")
        total_counts += counts

        class_dist = {name: counts.get(cls_id, 0) for cls_id, name in enumerate(CLASS_NAMES)}
        stats_report["splits"][split] = {
            "num_images": split_info["images"],
            "class_distribution": class_dist,
        }

        print(f"\n--- {split.upper()} ---")
        for name, count in class_dist.items():
            print(f"  {name:12s}: {count}")

    print("\n--- TOPLAM ---")
    for cls_id, name in enumerate(CLASS_NAMES):
        total_val = total_counts.get(cls_id, 0)
        stats_report["total_class_distribution"][name] = total_val
        print(f"  {name:12s}: {total_val}")

    # İstatistikleri JSON olarak kaydet (MLOps Denetlenebilirliği)
    stats_path = PROCESSED_DIR / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Veri istatistik raporu kaydedildi: {stats_path}")


if __name__ == "__main__":
    main()