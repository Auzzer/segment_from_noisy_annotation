"""Generate train/test splits for NIfTI dataset."""

import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def collect_samples(data_dir: Path):
    """
    Collect all samples with matching image/label pairs.
    
    Args:
        data_dir: Root directory containing images/ and labels/ subfolders
        
    Returns:
        Sorted list of image ids (e.g., 'image_001') that have matching labels
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    samples = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        base_name = image_path.stem.split(".")[0]
        suffix = base_name.split("_")[1] if "_" in base_name else base_name
        label_path = labels_dir / f"label_{suffix}.nii.gz"
        if label_path.exists():
            samples.append(base_name)

    if not samples:
        raise RuntimeError(f"No paired image/label volumes found under {data_dir}")

    return samples


def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for item in items:
            fp.write(f"{item}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate train/test splits for NIfTI dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, default=Path("./data"),
                        help="Root directory containing images/ and labels/ subfolders")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directory to save split lists (default: <data_dir>/splits)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of samples assigned to the training split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the split")
    args = parser.parse_args()

    samples = collect_samples(args.data_dir)

    train_ids, test_ids = train_test_split(
        samples,
        train_size=args.train_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    output_dir = args.output_dir or (args.data_dir / "splits")
    train_path = output_dir / "train_list.txt"
    test_path = output_dir / "test_list.txt"

    write_list(train_path, train_ids)
    write_list(test_path, test_ids)

    print(f"Total paired samples: {len(samples)}")
    print(f"Train samples: {len(train_ids)} -> {train_path}")
    print(f"Test samples: {len(test_ids)} -> {test_path}")


if __name__ == "__main__":
    main()

