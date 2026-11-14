"""Download PlantVillage and export its images into class folders."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def save_image_if_missing(image, image_path):
    image_path = Path(image_path)
    if image_path.exists():
        if not image_path.is_file():
            raise FileExistsError(
                f"Expected an image file but found another path: {image_path}"
            )
        try:
            with Image.open(image_path) as existing:
                existing.verify()
            return False
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            pass

    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=image_path.parent,
            prefix=".plantvillage-",
            suffix=".jpg",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        image.save(temporary_path, format="JPEG", quality=95)
        os.replace(temporary_path, image_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download PlantVillage using TensorFlow Datasets and export images "
            "to folders that can be browsed in the Streamlit uploader."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "plantvillage",
        help="Directory for exported class/image folders.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "tensorflow_datasets",
        help="TensorFlow Datasets cache directory.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help=(
            "Export at most this many new images per run (0 exports the full "
            "dataset). Existing valid images are kept and skipped."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_images < 0:
        raise SystemExit("--max-images must be zero or a positive integer.")

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow Datasets is required. Install it with "
            "'pip install -r requirements-dataset.txt'."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    dataset, info = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        with_info=True,
        data_dir=str(args.cache_dir),
        shuffle_files=False,
    )
    class_names = info.features["label"].names
    counts = [0] * len(class_names)
    total = 0

    for image, label in tfds.as_numpy(dataset):
        class_index = int(label)
        if class_index < 0 or class_index >= len(class_names):
            raise RuntimeError(f"Dataset returned invalid class index {class_index}.")

        class_dir = args.output_dir / class_names[class_index]
        class_dir.mkdir(parents=True, exist_ok=True)
        image_path = class_dir / f"image_{counts[class_index]:05d}.jpg"
        image_to_save = Image.fromarray(np.asarray(image, dtype=np.uint8))
        if save_image_if_missing(
            image_to_save,
            image_path,
        ):
            total += 1
        counts[class_index] += 1

        if total and total % 1000 == 0:
            print(f"Exported {total} images...", flush=True)
        if args.max_images and total >= args.max_images:
            break

    class_names_path = args.output_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps(class_names, indent=2) + "\n",
        encoding="utf-8",
    )
    class_folder_count = sum(
        (args.output_dir / class_name).is_dir()
        for class_name in class_names
    )
    print(f"Exported {total} new images to {args.output_dir}")
    print(f"Dataset has {class_folder_count} class folders.")


if __name__ == "__main__":
    main()
