from pathlib import Path
from PIL import Image
import json
from collections import Counter

ROOT = Path("data/raw/chest_xray")
OUTPUT = Path("reports/dataset_audit.json")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "test"]

results = {
    "dataset_root": str(ROOT),
    "splits": {},
    "total_images": 0,
}

for split in splits:
    split_dir = ROOT / split

    if not split_dir.exists():
        continue

    split_info = {
        "total": 0,
        "classes": {},
        "extensions": {},
        "modes": {},
        "width": {
            "min": None,
            "max": None,
            "average": None,
        },
        "height": {
            "min": None,
            "max": None,
            "average": None,
        },
        "corrupted": 0,
    }

    widths = []
    heights = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        files = list(class_dir.rglob("*"))

        image_files = [
            f for f in files
            if f.is_file()
            and f.suffix.lower() in {".jpeg", ".jpg", ".png"}
        ]

        split_info["classes"][class_name] = len(image_files)
        split_info["total"] += len(image_files)

        for image_path in image_files:
            ext = image_path.suffix.lower()
            split_info["extensions"][ext] = (
                split_info["extensions"].get(ext, 0) + 1
            )

            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)

                    mode = img.mode
                    split_info["modes"][mode] = (
                        split_info["modes"].get(mode, 0) + 1
                    )

            except Exception:
                split_info["corrupted"] += 1

    if widths:
        split_info["width"]["min"] = min(widths)
        split_info["width"]["max"] = max(widths)
        split_info["width"]["average"] = round(sum(widths) / len(widths), 2)

    if heights:
        split_info["height"]["min"] = min(heights)
        split_info["height"]["max"] = max(heights)
        split_info["height"]["average"] = round(sum(heights) / len(heights), 2)

    results["splits"][split] = split_info
    results["total_images"] += split_info["total"]

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("=" * 60)
print("PNEUMOVISION DATASET AUDIT")
print("=" * 60)

for split, info in results["splits"].items():
    print(f"\n{split.upper()}")
    print("-" * 40)
    print(f"Total: {info['total']}")
    print(f"Classes: {info['classes']}")
    print(f"Extensions: {info['extensions']}")
    print(f"Modes: {info['modes']}")
    print(f"Width: {info['width']}")
    print(f"Height: {info['height']}")
    print(f"Corrupted: {info['corrupted']}")

print("\n" + "=" * 60)
print(f"TOTAL IMAGES: {results['total_images']}")
print(f"Report saved to: {OUTPUT}")
print("=" * 60)