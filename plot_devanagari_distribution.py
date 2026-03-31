import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Default 48-class Devanagari script order (vowels + consonants), transliterated
# so plots render correctly even when Devanagari fonts are unavailable.
DEFAULT_DEVANAGARI_LABELS_48 = [
    "a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au", "am", "ah",
    "ka", "kha", "ga", "gha", "nga",
    "cha", "chha", "ja", "jha", "nya",
    "tta", "ttha", "dda", "ddha", "nna",
    "ta", "tha", "da", "dha", "na",
    "pa", "pha", "ba", "bha", "ma",
    "ya", "ra", "la", "va",
    "sha", "ssha", "sa", "ha",
    "ksha", "tra", "gya",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Devanagari class frequency graph with script names (not Class 1/2)."
    )
    parser.add_argument(
        "--data-root",
        default="DEVNAGARI_NEW",
        help="Dataset root containing TRAIN/ and optionally TEST/.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which split(s) to include in counts.",
    )
    parser.add_argument(
        "--labels-file",
        default=os.path.join("models", "devanagari_labels.txt"),
        help="Label file path. If this contains 'Class N', script names are auto-mapped.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("figures", "devanagari_script_distribution.png"),
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--csv-output",
        default=os.path.join("figures", "devanagari_script_distribution.csv"),
        help="Path to save per-script counts as CSV.",
    )
    return parser.parse_args()


def list_numeric_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.isdigit()]
    return sorted(dirs, key=lambda x: int(x))


def count_images_in_dir(path: str) -> int:
    if not os.path.isdir(path):
        return 0

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sum(1 for name in os.listdir(path) if name.lower().endswith(valid_exts))


def load_labels(labels_file: str, num_classes: int) -> List[str]:
    labels = []
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    uses_class_prefix = bool(labels) and all(label.lower().startswith("class ") for label in labels)

    if uses_class_prefix and num_classes == 48:
        return DEFAULT_DEVANAGARI_LABELS_48

    if len(labels) >= num_classes:
        return labels[:num_classes]

    if num_classes == 48:
        return DEFAULT_DEVANAGARI_LABELS_48

    # Fallback without using 'Class N' wording.
    return [f"Script {i + 1}" for i in range(num_classes)]


def build_counts(data_root: str, split: str) -> Tuple[List[str], Dict[str, int]]:
    split_dirs = []
    if split in ("train", "all"):
        split_dirs.append(os.path.join(data_root, "TRAIN"))
    if split in ("test", "all"):
        split_dirs.append(os.path.join(data_root, "TEST"))

    class_dirs = []
    for split_dir in split_dirs:
        for class_name in list_numeric_dirs(split_dir):
            if class_name not in class_dirs:
                class_dirs.append(class_name)

    class_dirs = sorted(class_dirs, key=lambda x: int(x))
    counts = {class_name: 0 for class_name in class_dirs}

    for split_dir in split_dirs:
        for class_name in class_dirs:
            counts[class_name] += count_images_in_dir(os.path.join(split_dir, class_name))

    return class_dirs, counts


def main() -> None:
    args = parse_args()

    class_dirs, counts = build_counts(args.data_root, args.split)
    if not class_dirs:
        raise RuntimeError("No numeric class folders found. Check --data-root and dataset structure.")

    labels = load_labels(args.labels_file, len(class_dirs))
    if len(labels) != len(class_dirs):
        raise RuntimeError("Label count does not match class count. Provide a valid labels file.")

    rows = []
    for idx, class_name in enumerate(class_dirs):
        rows.append(
            {
                "class_id": int(class_name),
                "script": labels[idx],
                "count": counts[class_name],
            }
        )

    df = pd.DataFrame(rows).sort_values("class_id")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    df.to_csv(args.csv_output, index=False, encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig_h = max(8, len(df) * 0.34)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    sns.barplot(data=df, y="script", x="count", color="#4C72B0", ax=ax)

    ax.set_title("Devanagari Script Distribution", fontsize=15, weight="bold")
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Script", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure:")
    print(f"- {args.output}")
    print("Saved counts CSV:")
    print(f"- {args.csv_output}")


if __name__ == "__main__":
    main()
