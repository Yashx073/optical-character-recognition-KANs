import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Devanagari OCR models (CNN, RNN, and CNN+KAN) for app inference."
    )
    parser.add_argument("--data-dir", default="DEVNAGARI_NEW", help="Dataset root containing TRAIN/ and TEST/ folders.")
    parser.add_argument("--cnn-epochs", type=int, default=10, help="Number of CNN training epochs.")
    parser.add_argument("--rnn-epochs", type=int, default=10, help="Number of RNN training epochs.")
    parser.add_argument("--kan-epochs", type=int, default=12, help="Number of KAN training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output-dir", default="models", help="Directory to store trained model and labels.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If > 0, use only this many training samples for faster experimentation.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="If > 0, use only this many test samples for faster evaluation.",
    )
    return parser.parse_args()


def sorted_class_folders(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.isdigit()]
    return sorted(folders, key=lambda value: int(value))


def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.resize(image, (28, 28))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    image = image.astype("float32") / 255.0

    if np.mean(image) > 0.5:
        image = 1.0 - image

    return image


def load_split(split_dir: str, class_folders: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    class_to_index = {class_name: index for index, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        file_names = [
            name for name in os.listdir(class_path)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        for file_name in file_names:
            image_path = os.path.join(class_path, file_name)
            try:
                image = preprocess_image(image_path)
            except ValueError:
                continue
            images.append(image)
            labels.append(class_to_index[class_name])

    if not images:
        raise RuntimeError(f"No images found in split: {split_dir}")

    x = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28, 1)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def build_cnn_model(num_classes: int, learning_rate: float) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu", name="feature_dense"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_rnn_model(num_classes: int, learning_rate: float) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


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


def build_kan_model(input_dim: int, num_classes: int, learning_rate: float) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = KANLayer(128)(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, model_name: str) -> None:
    if x_test is None or y_test is None:
        return

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"{model_name} test accuracy: {test_accuracy:.4f} | test loss: {test_loss:.4f}")


def save_labels(output_path: str, class_folders: List[str]) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for class_name in class_folders:
            handle.write(f"Class {class_name}\n")


def limit_samples(x: np.ndarray, y: np.ndarray, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(x) <= max_samples:
        return x, y

    return x[:max_samples], y[:max_samples]


def main() -> None:
    args = parse_args()
    tf.random.set_seed(42)
    np.random.seed(42)

    train_dir = os.path.join(args.data_dir, "TRAIN")
    test_dir = os.path.join(args.data_dir, "TEST")

    class_folders = sorted_class_folders(train_dir)
    if not class_folders:
        raise RuntimeError(f"No numeric class folders found in: {train_dir}")

    print(f"Detected {len(class_folders)} classes from {train_dir}")

    x_train, y_train = load_split(train_dir, class_folders)
    print(f"Train samples: {len(x_train)}")
    x_train, y_train = limit_samples(x_train, y_train, args.max_train_samples)
    if args.max_train_samples > 0:
        print(f"Using limited train samples: {len(x_train)}")

    if os.path.isdir(test_dir):
        x_test, y_test = load_split(test_dir, class_folders)
        print(f"Test samples: {len(x_test)}")
        x_test, y_test = limit_samples(x_test, y_test, args.max_test_samples)
        if args.max_test_samples > 0:
            print(f"Using limited test samples: {len(x_test)}")
    else:
        x_test, y_test = None, None
        print("No TEST directory found. Training without test evaluation.")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    num_classes = len(class_folders)

    print("\nTraining CNN model...")
    cnn_model = build_cnn_model(num_classes=num_classes, learning_rate=args.learning_rate)
    cnn_model.summary()
    cnn_model.fit(
        x_train,
        y_train,
        epochs=args.cnn_epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    evaluate_model(cnn_model, x_test, y_test, "CNN")

    print("\nTraining RNN model...")
    rnn_model = build_rnn_model(num_classes=num_classes, learning_rate=args.learning_rate)
    rnn_model.summary()
    rnn_model.fit(
        x_train.reshape(-1, 28, 28),
        y_train,
        epochs=args.rnn_epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    evaluate_model(
        rnn_model,
        None if x_test is None else x_test.reshape(-1, 28, 28),
        y_test,
        "RNN",
    )

    print("\nTraining CNN+KAN model...")
    feature_extractor = models.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.get_layer("feature_dense").output,
    )

    train_features = feature_extractor.predict(x_train, batch_size=256, verbose=0)

    x_train_kan, x_val_kan, y_train_kan, y_val_kan = train_test_split(
        train_features,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )

    kan_model = build_kan_model(
        input_dim=train_features.shape[1],
        num_classes=num_classes,
        learning_rate=args.learning_rate,
    )
    kan_model.summary()
    kan_model.fit(
        x_train_kan,
        y_train_kan,
        epochs=args.kan_epochs,
        batch_size=args.batch_size,
        validation_data=(x_val_kan, y_val_kan),
        callbacks=callbacks,
        verbose=1,
    )

    if x_test is not None and y_test is not None:
        test_features = feature_extractor.predict(x_test, batch_size=256, verbose=0)
        evaluate_model(kan_model, test_features, y_test, "KAN")

    os.makedirs(args.output_dir, exist_ok=True)

    cnn_model_path = os.path.join(args.output_dir, "devanagari_cnn_model.h5")
    rnn_model_path = os.path.join(args.output_dir, "devanagari_rnn_model.h5")
    feature_extractor_path = os.path.join(args.output_dir, "devanagari_feature_extractor.h5")
    kan_model_path = os.path.join(args.output_dir, "devanagari_kan_model.h5")
    labels_path = os.path.join(args.output_dir, "devanagari_labels.txt")

    cnn_model.save(cnn_model_path)
    rnn_model.save(rnn_model_path)
    feature_extractor.save(feature_extractor_path)
    kan_model.save(kan_model_path)
    save_labels(labels_path, class_folders)

    print("\nSaved Devanagari artifacts:")
    print(f"- {cnn_model_path}")
    print(f"- {rnn_model_path}")
    print(f"- {feature_extractor_path}")
    print(f"- {kan_model_path}")
    print(f"- {labels_path}")


if __name__ == "__main__":
    main()
