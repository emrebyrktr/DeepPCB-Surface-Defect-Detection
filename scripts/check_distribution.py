from pathlib import Path
from collections import Counter

INTERIM_DIR = Path("data/interim")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def count_classes(labels_dir: Path) -> Counter:
    counter = Counter()
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counter[cls_id] += 1
    return counter

def main():
    total = Counter()

    for split in SPLITS:
        labels_dir = INTERIM_DIR / split / "labels"
        counts = count_classes(labels_dir)
        total += counts

        print(f"\n--- {split.upper()} ---")
        for cls_id, name in enumerate(CLASS_NAMES):
            print(f"  {name:12s}: {counts.get(cls_id, 0)}")

    print(f"\n--- TOPLAM ---")
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {total.get(cls_id, 0)}")

if __name__ == "__main__":
    main()