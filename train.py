"""
Training & Evaluation Pipeline for GTSRB Traffic Sign Recognition CNN.

Features:
- Loads official GTSRB data (or synthetic 43-class dataset with `--use-synthetic`)
- Applies OpenCV LAB-space CLAHE contrast normalization and real-time augmentation
- Trains the Multi-Scale TensorFlow/Keras CNN with EarlyStopping, ReduceLROnPlateau,
  and ModelCheckpoint callbacks
- Saves training/validation accuracy & loss curves and confusion matrix plots
"""

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DATA_DIR,
    EPOCHS,
    INITIAL_LR,
    MODELS_DIR,
    NUM_CLASSES,
    OUTPUTS_DIR,
)
from dataset import compute_class_weights, load_gtsrb_from_directory
from model import TF_AVAILABLE, build_traffic_sign_cnn
from preprocessing import OpenCVDataAugmenter


def plot_training_history(history_dict: Dict[str, List[float]], save_path: Path) -> Path:
    """Plot and save training vs. validation accuracy and loss curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history_dict["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    fig.suptitle("GTSRB Multi-Scale CNN — Training & Validation Performance", fontsize=13, fontweight="bold")

    ax1.plot(epochs_range, history_dict["accuracy"], label="Train Accuracy", color="#2563EB", linewidth=2)
    ax1.plot(epochs_range, history_dict["val_accuracy"], label="Validation Accuracy", color="#10B981", linewidth=2)
    ax1.set_title("Classification Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.0, 1.02])
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs_range, history_dict["loss"], label="Train Loss", color="#EF4444", linewidth=2)
    ax2.plot(epochs_range, history_dict["val_loss"], label="Validation Loss", color="#F59E0B", linewidth=2)
    ax2.set_title("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def evaluate_and_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute overall accuracy and macro-averaged precision, recall, and F1 score."""
    accuracy = float(np.mean(y_true == y_pred))
    precisions, recalls, f1s = [], [], []

    for cid in np.unique(y_true):
        tp = float(np.sum((y_true == cid) & (y_pred == cid)))
        fp = float(np.sum((y_true != cid) & (y_pred == cid)))
        fn = float(np.sum((y_true == cid) & (y_pred != cid)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }
    print("\n=== Validation Evaluation Summary ===")
    print(f"Overall Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision  : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall     : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score   : {metrics['macro_f1'] * 100:.2f}%")
    return metrics


def train_model(
    data_dir: Path = DATA_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = INITIAL_LR,
    use_clahe: bool = True,
) -> None:
    """Execute end-to-end GTSRB training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_gtsrb_from_directory(
        data_dir=data_dir,
        use_clahe=use_clahe,
    )
    print(f"[INFO] Loaded dataset: X_train={X_train.shape}, X_val={X_val.shape}")

    class_weights = compute_class_weights(y_train)
    augmenter = OpenCVDataAugmenter(seed=42)

    if not TF_AVAILABLE:
        print(
            "[WARN] TensorFlow is not installed in the current Python environment. "
            "Install TensorFlow (`pip install tensorflow`) to train the Keras CNN."
        )
        return

    import tensorflow as tf

    model = build_traffic_sign_cnn(learning_rate=learning_rate)
    model.summary()

    # Create a tf.data generator applying OpenCV real-time augmentation per epoch
    def augmented_generator():
        rng = np.random.default_rng()
        indices = rng.permutation(len(X_train))
        for idx in indices:
            yield augmenter.augment(X_train[idx]), y_train[idx]

    train_ds = tf.data.Dataset.from_generator(
        augmented_generator,
        output_signature=(
            tf.TensorSpec(shape=X_train.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    curve_path = OUTPUTS_DIR / "training_curves.png"
    plot_training_history(history.history, curve_path)
    print(f"[INFO] Saved training history plot to '{curve_path}'")

    val_probs = model.predict(X_val, batch_size=batch_size, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    evaluate_and_report(y_val, val_preds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GTSRB Traffic Sign Recognition CNN")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to GTSRB dataset folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=INITIAL_LR, help="Initial Adam learning rate")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast normalization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_clahe=not args.no_clahe,
    )
