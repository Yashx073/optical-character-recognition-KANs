import argparse
import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope


class KANLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer="ones",
            trainable=True,
        )

    def call(self, inputs):
        linear = tf.matmul(inputs, self.W) + self.b
        nonlinear = tf.math.sin(self.alpha * linear)
        return linear + nonlinear

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


SCRIPT_CONFIGS = {
    "english": {
        "title": "English (A-Z)",
        "model_prefix": "",
        "default_image": os.path.join("test_images", "a.jpg"),
        "cnn_candidates": [
            os.path.join("models", "cls_deneme_model.h5"),
            os.path.join("models", "cls_deneme_model.keras"),
            os.path.join("models", "cnn_model.h5"),
            os.path.join("models", "cnn_model.keras"),
        ],
        "rnn_candidates": [
            os.path.join("models", "rnn_ocr_model.h5"),
            os.path.join("models", "rnn_ocr_model.keras"),
        ],
    },
    "devanagari": {
        "title": "Devanagari",
        "model_prefix": "devanagari_",
        "default_image": None,
        "cnn_candidates": [
            os.path.join("models", "devanagari_cnn_model.h5"),
            os.path.join("models", "devanagari_cnn_model.keras"),
        ],
        "rnn_candidates": [
            os.path.join("models", "devanagari_rnn_model.h5"),
            os.path.join("models", "devanagari_rnn_model.keras"),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interpretability visualizations for CNN + KAN OCR models."
    )
    parser.add_argument(
        "--script",
        default="english",
        choices=["english", "devanagari"],
        help="Dataset/script family to analyze.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a sample image. If omitted, a default sample is used.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("figures", "interpretability"),
        help="Directory where paper figures and notes are saved.",
    )
    parser.add_argument(
        "--max-feature-maps",
        type=int,
        default=16,
        help="Number of CNN feature maps to visualize.",
    )
    parser.add_argument(
        "--num-splines",
        type=int,
        default=6,
        help="Number of KAN transfer curves to plot.",
    )
    return parser.parse_args()


def list_class_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    numeric_folders = [folder for folder in folders if folder.isdigit()]
    return sorted(numeric_folders, key=lambda value: int(value))


def load_labels(script_key: str) -> List[str]:
    if script_key == "english":
        return [chr(index + 65) for index in range(26)]

    label_files = [
        os.path.join("models", "devanagari_labels.txt"),
        os.path.join("DEVNAGARI_NEW", "labels.txt"),
    ]
    for label_file in label_files:
        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as handle:
                labels = [line.strip() for line in handle if line.strip()]
            if labels:
                return labels

    train_dirs = list_class_dirs(os.path.join("DEVNAGARI_NEW", "TRAIN"))
    if train_dirs:
        return [f"Class {folder}" for folder in train_dirs]
    return []


def candidate_paths(prefix: str, base_name: str) -> List[str]:
    return [
        os.path.join("models", f"{prefix}{base_name}.h5"),
        os.path.join("models", f"{prefix}{base_name}.keras"),
    ]


def first_existing(paths: List[str]) -> Optional[str]:
    return next((path for path in paths if os.path.exists(path)), None)


def load_model(paths: List[str], custom_objects=None) -> tf.keras.Model:
    model_path = first_existing(paths)
    if model_path is None:
        raise FileNotFoundError(f"Could not find any model file in: {paths}")

    if custom_objects:
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
    else:
        model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Loaded: {model_path}")
    return model


def ensure_model_ready(model: tf.keras.Model, sample_shape: Tuple[int, ...]) -> None:
    try:
        _ = model.inputs
        if model.inputs:
            return
    except Exception:
        pass

    dummy = np.zeros(sample_shape, dtype=np.float32)
    _ = model(dummy, training=False)


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(raw, (28, 28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    img = img.astype("float32") / 255.0

    if float(np.mean(img)) > 0.5:
        img = 1.0 - img

    return raw, img


def find_first_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise RuntimeError("No Conv2D layer found in CNN model.")


def find_kan_layer(model: tf.keras.Model) -> KANLayer:
    for layer in model.layers:
        if isinstance(layer, KANLayer):
            return layer

    for layer in model.layers:
        if hasattr(layer, "alpha") and hasattr(layer, "W") and hasattr(layer, "b"):
            return layer

    raise RuntimeError("KAN layer not found in classifier model.")


def predict_label(prob: np.ndarray, labels: List[str]) -> Tuple[str, float]:
    idx = int(np.argmax(prob, axis=1)[0])
    conf = float(np.max(prob))
    if 0 <= idx < len(labels):
        return labels[idx], conf
    return f"Class {idx}", conf


def topk(prob: np.ndarray, labels: List[str], k: int = 3) -> List[Tuple[str, float]]:
    p = prob[0]
    k = min(k, len(p))
    order = np.argsort(p)[::-1][:k]
    items = []
    for idx in order:
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        items.append((label, float(p[idx])))
    return items


def plot_feature_maps(feature_map: np.ndarray, out_path: str, max_maps: int = 16) -> None:
    channels = feature_map.shape[-1]
    show_n = min(max_maps, channels)
    cols = int(np.ceil(np.sqrt(show_n)))
    rows = int(np.ceil(show_n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < show_n:
            fmap = feature_map[:, :, i]
            ax.imshow(fmap, cmap="magma")
            ax.set_title(f"Channel {i}", fontsize=10)

    fig.suptitle("CNN Feature Activation Maps", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kan_transfer_curves(
    linear_sample: np.ndarray,
    transformed_sample: np.ndarray,
    alpha: np.ndarray,
    out_path: str,
    num_curves: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.abs(transformed_sample)
    unit_ids = np.argsort(scores)[::-1][:num_curves]

    cols = 3
    rows = int(np.ceil(len(unit_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        ax.grid(alpha=0.25)
        if i >= len(unit_ids):
            ax.axis("off")
            continue

        u = unit_ids[i]
        center = float(linear_sample[u])
        x_min, x_max = center - 2.8, center + 2.8
        x = np.linspace(x_min, x_max, 300)
        y = x + np.sin(alpha[u] * x)

        ax.plot(x, x, linestyle="--", color="#64748B", label="Linear y=x")
        ax.plot(x, y, color="#0EA5E9", linewidth=2.0, label="KAN transfer")
        ax.scatter([linear_sample[u]], [transformed_sample[u]], color="#22C55E", s=22, zorder=5)
        ax.set_title(f"Unit {u} (alpha={alpha[u]:.2f})", fontsize=10)
        ax.set_xlabel("Affine input z")
        ax.set_ylabel("Output z + sin(alpha*z)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("KAN Unit-Wise Nonlinear Transfer Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return unit_ids, scores[unit_ids]


def plot_all_kan_splines(alpha: np.ndarray, out_path: str) -> None:
    x = np.linspace(-6.0, 6.0, 400)
    alpha_min = float(np.min(alpha))
    alpha_max = float(np.max(alpha))

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 7.0))
    cmap = plt.cm.viridis

    for idx, a in enumerate(alpha):
        norm = (float(a) - alpha_min) / (alpha_max - alpha_min + 1e-9)
        y = x + np.sin(float(a) * x)
        ax.plot(x, y, color=cmap(norm), alpha=0.35, linewidth=1.0)

    ax.plot(x, x, linestyle="--", color="#475569", linewidth=1.2, label="Linear baseline y=x")
    ax.set_title("All KAN Unit Transfer Curves in a Single Diagram", fontsize=16, fontweight="bold")
    ax.set_xlabel("Affine input z")
    ax.set_ylabel("KAN output z + sin(alpha*z)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=alpha_min, vmax=alpha_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("alpha value per unit")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_transformation(
    features: np.ndarray,
    linear_sample: np.ndarray,
    transformed_sample: np.ndarray,
    out_path: str,
) -> None:
    show_dim = min(40, len(features))
    x1 = np.arange(show_dim)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    axes[0].plot(x1, features[:show_dim], color="#2563EB", linewidth=1.8)
    axes[0].set_title("CNN Feature Vector (first 40 dims)")
    axes[0].set_xlabel("Feature index")
    axes[0].set_ylabel("Activation value")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(linear_sample, transformed_sample, s=12, alpha=0.8, color="#10B981")
    diag = np.linspace(float(np.min(linear_sample)), float(np.max(linear_sample)), 100)
    axes[1].plot(diag, diag, "--", color="#64748B", linewidth=1.0, label="Linear identity")
    axes[1].set_title("KAN Transformation: Affine Input vs Output")
    axes[1].set_xlabel("Affine input z")
    axes[1].set_ylabel("KAN output z + sin(alpha*z)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("How CNN Features Are Transformed Before Classification", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_summary(
    input_image: np.ndarray,
    feature_map: np.ndarray,
    linear_sample: np.ndarray,
    transformed_sample: np.ndarray,
    prediction_text: str,
    out_path: str,
) -> None:
    avg_map = np.mean(feature_map, axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axes[0].imshow(input_image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(avg_map, cmap="magma")
    axes[1].set_title("CNN Feature Map (avg)")
    axes[1].axis("off")

    idx = np.arange(min(32, len(linear_sample)))
    axes[2].plot(idx, linear_sample[: len(idx)], label="Affine z", color="#64748B")
    axes[2].plot(idx, transformed_sample[: len(idx)], label="KAN output", color="#0EA5E9")
    axes[2].set_title("KAN Transformation")
    axes[2].set_xlabel("Unit index")
    axes[2].set_ylabel("Activation")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    axes[3].axis("off")
    axes[3].text(
        0.02,
        0.88,
        "Prediction",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        transform=axes[3].transAxes,
    )
    axes[3].text(
        0.02,
        0.65,
        prediction_text,
        fontsize=11,
        color="#1E293B",
        transform=axes[3].transAxes,
    )

    fig.text(0.245, 0.5, "->", fontsize=20, fontweight="bold")
    fig.text(0.49, 0.5, "->", fontsize=20, fontweight="bold")
    fig.text(0.735, 0.5, "->", fontsize=20, fontweight="bold")

    fig.suptitle("Interpretability Flow: Input -> CNN Map -> KAN Transform -> Prediction", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_paper_notes(
    out_path: str,
    script_title: str,
    image_path: str,
    top_predictions: List[Tuple[str, float]],
    selected_units: np.ndarray,
    selected_scores: np.ndarray,
    alpha: np.ndarray,
) -> None:
    lines = []
    lines.append("# Interpretability Notes (Paper-Ready)\n")
    lines.append(f"- Script analyzed: {script_title}")
    lines.append(f"- Sample image: `{image_path}`")
    lines.append("- Model pipeline: Input image -> CNN feature map -> KAN nonlinear transfer -> class prediction\n")

    lines.append("## What each KAN curve represents")
    lines.append(
        "Each plotted curve is one KAN hidden unit transfer function. "
        "For unit `u`, the x-axis is the affine input `z_u = w_u^T f + b_u` from CNN features, "
        "and the y-axis is `z_u + sin(alpha_u * z_u)`."
    )
    lines.append(
        "If `alpha_u` is larger, the curve oscillates more rapidly, indicating stronger nonlinearity. "
        "The green marker in each subplot is the current sample's operating point.\n"
    )

    lines.append("## Top predictions")
    for i, (label, conf) in enumerate(top_predictions, start=1):
        lines.append(f"{i}. {label}: {conf * 100:.2f}%")

    lines.append("\n## Most influential KAN units in this sample")
    lines.append("Units were selected by largest absolute transformed activation.")
    for unit, score in zip(selected_units, selected_scores):
        lines.append(
            f"- Unit {int(unit)}: |activation|={float(score):.4f}, alpha={float(alpha[int(unit)]):.4f}"
        )

    lines.append("\n## Suggested figure captions")
    lines.append("- `pipeline_summary.png`: End-to-end interpretable flow from input to decision through CNN and KAN.")
    lines.append("- `cnn_feature_maps.png`: Representative CNN channel activations highlighting spatial evidence used for recognition.")
    lines.append("- `kan_transfer_curves.png`: Unit-wise KAN nonlinear transfer functions with sample operating points.")
    lines.append("- `kan_all_splines_single_diagram.png`: Overlay of all KAN unit transfer curves in one plot (color indicates alpha).")
    lines.append("- `feature_transformation.png`: Relationship between affine unit input and KAN output, showing non-linear shaping.")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cfg = SCRIPT_CONFIGS[args.script]

    os.makedirs(args.output_dir, exist_ok=True)
    labels = load_labels(args.script)
    if not labels:
        raise RuntimeError("No labels found. Add labels file or dataset folders before running visualization.")

    image_path = args.image if args.image else cfg["default_image"]
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(
            "Could not find sample image. Pass --image <path> explicitly."
        )

    prefix = cfg["model_prefix"]
    cnn_model = load_model(cfg["cnn_candidates"])
    feature_extractor = load_model(candidate_paths(prefix, "feature_extractor"))
    kan_model = load_model(candidate_paths(prefix, "kan_model"), custom_objects={"KANLayer": KANLayer})

    raw_image, processed = preprocess_image(image_path)
    x = processed.reshape(1, 28, 28, 1)

    ensure_model_ready(cnn_model, (1, 28, 28, 1))
    ensure_model_ready(feature_extractor, (1, 28, 28, 1))

    conv_name = find_first_conv_layer_name(cnn_model)
    fmap_model = tf.keras.Model(inputs=cnn_model.inputs, outputs=cnn_model.get_layer(conv_name).output)
    feature_map = fmap_model.predict(x, verbose=0)[0]

    features = feature_extractor.predict(x, verbose=0)
    prob = kan_model.predict(features, verbose=0)
    pred_label, pred_conf = predict_label(prob, labels)
    top_predictions = topk(prob, labels, k=3)

    kan_layer = find_kan_layer(kan_model)
    W = kan_layer.W.numpy()
    b = kan_layer.b.numpy()
    alpha = kan_layer.alpha.numpy()

    f = features[0]
    linear = np.dot(f, W) + b
    transformed = linear + np.sin(alpha * linear)

    feature_map_path = os.path.join(args.output_dir, "cnn_feature_maps.png")
    kan_curve_path = os.path.join(args.output_dir, "kan_transfer_curves.png")
    all_splines_path = os.path.join(args.output_dir, "kan_all_splines_single_diagram.png")
    transform_path = os.path.join(args.output_dir, "feature_transformation.png")
    pipeline_path = os.path.join(args.output_dir, "pipeline_summary.png")
    notes_path = os.path.join(args.output_dir, "interpretability_notes.md")

    plot_feature_maps(feature_map, feature_map_path, max_maps=args.max_feature_maps)
    selected_units, selected_scores = plot_kan_transfer_curves(
        linear,
        transformed,
        alpha,
        kan_curve_path,
        num_curves=args.num_splines,
    )
    plot_all_kan_splines(alpha, all_splines_path)
    plot_feature_transformation(f, linear, transformed, transform_path)

    pred_text = f"Predicted: {pred_label}\nConfidence: {pred_conf * 100:.2f}%\n\nTop-3:\n"
    for rank, (label, score) in enumerate(top_predictions, start=1):
        pred_text += f"{rank}. {label}: {score * 100:.2f}%\n"

    plot_pipeline_summary(raw_image, feature_map, linear, transformed, pred_text, pipeline_path)
    save_paper_notes(
        notes_path,
        cfg["title"],
        image_path,
        top_predictions,
        selected_units,
        selected_scores,
        alpha,
    )

    print("\nGenerated interpretability files:")
    print(f"- {feature_map_path}")
    print(f"- {kan_curve_path}")
    print(f"- {all_splines_path}")
    print(f"- {transform_path}")
    print(f"- {pipeline_path}")
    print(f"- {notes_path}")


if __name__ == "__main__":
    main()
