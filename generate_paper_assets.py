import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import custom_object_scope

from train_devanagari import KANLayer, load_split, sorted_class_folders


sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready graphs, tables, and figures.")
    parser.add_argument("--data-dir", default="DEVNAGARI_NEW", help="Dataset root with TRAIN/TEST")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained models")
    parser.add_argument("--output-dir", default=os.path.join("figures", "paper_assets"), help="Output directory")
    parser.add_argument(
        "--logs-dir",
        default=os.path.join("models", "training_logs"),
        help="Directory with *_history.json logs from train_devanagari.py",
    )
    parser.add_argument(
        "--streamlit-screenshot",
        default=None,
        help="Optional path to Streamlit UI screenshot to copy into paper assets.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="If > 0, evaluate only this many test samples for faster runs.",
    )
    return parser.parse_args()


def find_first_existing(paths: List[str]) -> Optional[str]:
    return next((p for p in paths if os.path.exists(p)), None)


def load_model(paths: List[str], custom_objects=None) -> tf.keras.Model:
    model_path = find_first_existing(paths)
    if model_path is None:
        raise FileNotFoundError(f"Model file not found. Checked: {paths}")

    if custom_objects:
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
    else:
        model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Loaded model: {model_path}")
    return model


def load_labels(models_dir: str, class_folders: List[str]) -> List[str]:
    labels_path = os.path.join(models_dir, "devanagari_labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        if labels and labels[0].lower().startswith("class "):
            return [f"Script {i+1}" for i in range(len(class_folders))]
        if len(labels) >= len(class_folders):
            return labels[: len(class_folders)]

    return [f"Script {i+1}" for i in range(len(class_folders))]


def load_test_data(data_dir: str, max_test_samples: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    test_dir = os.path.join(data_dir, "TEST")
    train_dir = os.path.join(data_dir, "TRAIN")

    class_folders = sorted_class_folders(train_dir)
    if not class_folders:
        raise RuntimeError(f"No classes found in {train_dir}")

    x_test, y_test = load_split(test_dir, class_folders)

    if max_test_samples > 0 and len(x_test) > max_test_samples:
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    return x_test, y_test, class_folders


def read_training_logs(logs_dir: str) -> Dict[str, dict]:
    logs = {}
    for model_key in ["cnn", "rnn", "kan"]:
        path = os.path.join(logs_dir, f"{model_key}_history.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                logs[model_key] = json.load(handle)
    return logs


def plot_history_curves(logs: Dict[str, dict], out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    acc_path = os.path.join(out_dir, "accuracy_vs_epochs.png")
    loss_path = os.path.join(out_dir, "loss_vs_epochs.png")

    if not logs:
        return None, None

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_key, payload in logs.items():
        hist = payload.get("history", {})
        if "accuracy" in hist and hist["accuracy"]:
            ax.plot(hist["accuracy"], marker="o", label=f"{model_key.upper()} train")
        if "val_accuracy" in hist and hist["val_accuracy"]:
            ax.plot(hist["val_accuracy"], marker="x", linestyle="--", label=f"{model_key.upper()} val")
    ax.set_title("Accuracy vs Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_key, payload in logs.items():
        hist = payload.get("history", {})
        if "loss" in hist and hist["loss"]:
            ax.plot(hist["loss"], marker="o", label=f"{model_key.upper()} train")
        if "val_loss" in hist and hist["val_loss"]:
            ax.plot(hist["val_loss"], marker="x", linestyle="--", label=f"{model_key.upper()} val")
    ax.set_title("Loss vs Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return acc_path, loss_path


def evaluate_models(
    models_dir: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, int]]:
    cnn = load_model(
        [
            os.path.join(models_dir, "devanagari_cnn_model.h5"),
            os.path.join(models_dir, "devanagari_cnn_model.keras"),
        ]
    )
    rnn = load_model(
        [
            os.path.join(models_dir, "devanagari_rnn_model.h5"),
            os.path.join(models_dir, "devanagari_rnn_model.keras"),
        ]
    )
    feature_extractor = load_model(
        [
            os.path.join(models_dir, "devanagari_feature_extractor.h5"),
            os.path.join(models_dir, "devanagari_feature_extractor.keras"),
        ]
    )
    kan = load_model(
        [
            os.path.join(models_dir, "devanagari_kan_model.h5"),
            os.path.join(models_dir, "devanagari_kan_model.keras"),
        ],
        custom_objects={"KANLayer": KANLayer},
    )

    pred_probs_cnn = cnn.predict(x_test, batch_size=256, verbose=0)
    pred_cnn = np.argmax(pred_probs_cnn, axis=1)

    pred_probs_rnn = rnn.predict(x_test.reshape(-1, 28, 28), batch_size=256, verbose=0)
    pred_rnn = np.argmax(pred_probs_rnn, axis=1)

    feats = feature_extractor.predict(x_test, batch_size=256, verbose=0)
    pred_probs_kan = kan.predict(feats, batch_size=256, verbose=0)
    pred_kan = np.argmax(pred_probs_kan, axis=1)

    preds = {
        "CNN": pred_cnn,
        "RNN": pred_rnn,
        "CNN+KAN": pred_kan,
    }

    metrics = {}
    for name, pred in preds.items():
        acc = float(accuracy_score(y_test, pred))
        cer = float(np.mean(pred != y_test))
        metrics[name] = {
            "accuracy": acc,
            "cer": cer,
        }

    params = {
        "CNN": int(cnn.count_params()),
        "RNN": int(rnn.count_params()),
        "CNN+KAN": int(feature_extractor.count_params() + kan.count_params()),
    }

    return preds, metrics, params


def plot_confusion_matrices(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    labels: List[str],
    out_dir: str,
) -> str:
    out_path = os.path.join(out_dir, "confusion_matrix_comparison.png")
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 6.4))
    axes = np.array(axes).reshape(-1)

    for idx, (name, pred) in enumerate(preds.items()):
        cm = confusion_matrix(y_true, pred, labels=np.arange(len(labels)), normalize="true")
        ax = axes[idx]
        sns.heatmap(cm, cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{name} Confusion Matrix (Normalized)")
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("True")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cer_comparison(metrics: Dict[str, Dict[str, float]], out_dir: str) -> str:
    out_path = os.path.join(out_dir, "cer_comparison.png")
    rows = [{"model": name, "cer": val["cer"]} for name, val in metrics.items()]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.barplot(data=df, x="model", y="cer", hue="model", palette="Set2", legend=False, ax=ax)
    ax.set_title("CER Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Character Error Rate")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_tables(
    metrics: Dict[str, Dict[str, float]],
    params: Dict[str, int],
    logs: Dict[str, dict],
    out_dir: str,
) -> Tuple[str, str]:
    rows = []
    key_map = {"CNN": "cnn", "RNN": "rnn", "CNN+KAN": "kan"}
    for name in ["CNN", "RNN", "CNN+KAN"]:
        k = key_map[name]
        log = logs.get(k, {})
        rows.append(
            {
                "Model": name,
                "Accuracy": metrics[name]["accuracy"],
                "CER": metrics[name]["cer"],
                "Parameter Count": int(params[name]),
                "Training Time (s)": log.get("training_seconds", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "model_comparison_table.csv")
    md_path = os.path.join(out_dir, "model_comparison_table.md")
    df.to_csv(csv_path, index=False)

    headers = list(df.columns)
    sep = ["---"] * len(headers)

    def row_to_md(row_vals: List[str]) -> str:
        return "| " + " | ".join(row_vals) + " |\n"

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(row_to_md(headers))
        handle.write(row_to_md(sep))
        for _, row in df.iterrows():
            vals = [str(row[col]) for col in headers]
            handle.write(row_to_md(vals))

    return csv_path, md_path


def draw_architecture_diagram(out_dir: str) -> str:
    out_path = os.path.join(out_dir, "architecture_diagram.png")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")

    boxes = [
        (0.05, 0.55, 0.16, 0.25, "Input\n28x28x1"),
        (0.28, 0.55, 0.16, 0.25, "CNN\nFeature Extractor"),
        (0.51, 0.55, 0.16, 0.25, "KAN Layer\nNonlinear Transfer"),
        (0.74, 0.55, 0.16, 0.25, "Classifier\nSoftmax"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, color="#E2E8F0", ec="#334155", lw=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrows = [(0.21, 0.67, 0.28, 0.67), (0.44, 0.67, 0.51, 0.67), (0.67, 0.67, 0.74, 0.67)]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.8))

    ax.text(0.5, 0.18, "CNN and RNN are baseline branches; CNN+KAN is the proposed interpretable architecture.",
            ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_workflow_diagram(out_dir: str) -> str:
    out_path = os.path.join(out_dir, "workflow_diagram.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    steps = [
        (0.04, "Dataset\n(TRAIN/TEST)"),
        (0.24, "Preprocessing\n(28x28, threshold)"),
        (0.44, "Model Training\nCNN / RNN / CNN+KAN"),
        (0.64, "Evaluation\nAccuracy, CER, CM"),
        (0.84, "Deployment\nStreamlit App"),
    ]

    for x, text in steps:
        rect = plt.Rectangle((x, 0.45), 0.12, 0.22, fill=True, color="#DCFCE7", ec="#166534", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.06, 0.56, text, ha="center", va="center", fontsize=10)

    for i in range(len(steps) - 1):
        x0 = steps[i][0] + 0.12
        x1 = steps[i + 1][0]
        ax.annotate("", xy=(x1, 0.56), xytext=(x0, 0.56), arrowprops=dict(arrowstyle="->", lw=1.6))

    ax.text(0.5, 0.2, "End-to-end OCR workflow used for experiments and deployment", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def copy_streamlit_screenshot(screenshot_path: Optional[str], out_dir: str) -> Optional[str]:
    if not screenshot_path:
        return None
    if not os.path.exists(screenshot_path):
        return None

    ext = os.path.splitext(screenshot_path)[1].lower() or ".png"
    dst = os.path.join(out_dir, f"streamlit_interface_screenshot{ext}")
    shutil.copy2(screenshot_path, dst)
    return dst


def copy_kan_visualization(out_dir: str) -> Optional[str]:
    src = os.path.join("figures", "interpretability", "kan_all_splines_single_diagram.png")
    if not os.path.exists(src):
        return None
    dst = os.path.join(out_dir, "kan_function_visualization.png")
    shutil.copy2(src, dst)
    return dst


def save_asset_index(out_dir: str, assets: Dict[str, Optional[str]]) -> str:
    index_path = os.path.join(out_dir, "paper_assets_index.md")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write("# Paper Assets Index\n\n")
        for name, path in assets.items():
            if path:
                handle.write(f"- {name}: `{path}`\n")
            else:
                handle.write(f"- {name}: NOT GENERATED (missing input/logs)\n")
    return index_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    x_test, y_test, class_folders = load_test_data(args.data_dir, args.max_test_samples)
    labels = load_labels(args.models_dir, class_folders)

    logs = read_training_logs(args.logs_dir)
    acc_plot, loss_plot = plot_history_curves(logs, args.output_dir)

    preds, metrics, params = evaluate_models(args.models_dir, x_test, y_test)
    cm_plot = plot_confusion_matrices(y_test, preds, labels, args.output_dir)
    cer_plot = plot_cer_comparison(metrics, args.output_dir)
    table_csv, table_md = save_tables(metrics, params, logs, args.output_dir)

    arch_plot = draw_architecture_diagram(args.output_dir)
    workflow_plot = draw_workflow_diagram(args.output_dir)
    streamlit_shot = copy_streamlit_screenshot(args.streamlit_screenshot, args.output_dir)
    kan_viz = copy_kan_visualization(args.output_dir)

    assets = {
        "Accuracy vs Epochs": acc_plot,
        "Loss vs Epochs": loss_plot,
        "CER Comparison": cer_plot,
        "Confusion Matrix": cm_plot,
        "Model Comparison Table (CSV)": table_csv,
        "Model Comparison Table (Markdown)": table_md,
        "Architecture Diagram": arch_plot,
        "Workflow Diagram": workflow_plot,
        "Streamlit Interface Screenshot": streamlit_shot,
        "KAN Function Visualization": kan_viz,
    }
    index_path = save_asset_index(args.output_dir, assets)

    print("\nGenerated paper assets:")
    for key, path in assets.items():
        print(f"- {key}: {path if path else 'NOT GENERATED'}")
    print(f"- Asset Index: {index_path}")


if __name__ == "__main__":
    main()
