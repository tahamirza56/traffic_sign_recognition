"""
Single-Image, Batch, and Real-Time Webcam Inference for Traffic Sign Recognition.

Predicts:
- Exact GTSRB Class ID (0..42) and official Traffic Sign Name
- High-Level Functional Category (Speed Limit, Stop, Yield, Turn / Direction,
  Prohibitory, Danger / Warning, Mandatory, Priority)
- Top-5 Confidence Scores & Annotated Visual Output Card
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    MACRO_CATEGORY_MAP,
    NUM_CLASSES,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    get_sign_metadata,
)
from dataset import generate_sample_road_signs, render_traffic_sign_icon
from model import TF_AVAILABLE
from preprocessing import (
    apply_clahe_color,
    detect_traffic_sign_rois,
    preprocess_image,
)


class TrafficSignPredictor:
    """
    Unified inference engine that uses the trained TensorFlow/Keras CNN model
    (`models/gtsrb_traffic_sign_cnn.keras`) when available, and provides a
    zero-mean normalized cross-correlation (ZNCC) prototype classifier fallback
    for offline demonstration before model weights are trained.
    """

    def __init__(self, model_path: Path = BEST_MODEL_PATH) -> None:
        self.model = None
        self.model_path = model_path
        if TF_AVAILABLE and model_path.exists():
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(model_path))
            print(f"[INFO] Loaded trained Keras model from '{model_path}'")
        else:
            self._build_reference_prototypes()

    def _build_reference_prototypes(self) -> None:
        """Build zero-mean unit-norm CLAHE prototypes for all 43 classes."""
        protos = []
        for cid in range(NUM_CLASSES):
            icon = render_traffic_sign_icon(cid, canvas_size=64)
            proc = preprocess_image(icon, target_size=(IMG_WIDTH, IMG_HEIGHT), is_bgr=False, use_clahe=True)
            vec = proc.ravel().astype(np.float32)
            vec = vec - np.mean(vec)
            protos.append(vec / (np.linalg.norm(vec) + 1e-8))
        self.proto_unit = np.stack(protos, axis=0)

    def predict_preprocessed(self, proc_img: np.ndarray) -> np.ndarray:
        """Return 43-class probability vector for a preprocessed (32, 32, 3) float32 image."""
        if self.model is not None:
            probs = self.model.predict(proc_img[None, ...], verbose=0)[0]
            return np.asarray(probs, dtype=np.float32)

        vec = proc_img.ravel().astype(np.float32)
        vec = vec - np.mean(vec)
        vec_unit = vec / (np.linalg.norm(vec) + 1e-8)
        zncc = self.proto_unit @ vec_unit
        logits = zncc * 32.0
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        return (exp_scores / np.sum(exp_scores)).astype(np.float32)

    def predict_image(
        self,
        image_bgr: np.ndarray,
        auto_detect_roi: bool = False,
    ) -> Dict[str, object]:
        """
        Run full preprocessing + CLAHE contrast normalization + CNN classification
        on an input road-sign image (BGR format from `cv2.imread`).
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        roi_box: Optional[Tuple[int, int, int, int]] = None
        if auto_detect_roi:
            boxes = detect_traffic_sign_rois(rgb)
            if boxes:
                roi_box = boxes[0]

        proc = preprocess_image(
            rgb,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            roi_box=roi_box,
            is_bgr=False,
            use_clahe=True,
            norm_mode="minmax",
        )
        probs = self.predict_preprocessed(proc)
        top5_idx = np.argsort(probs)[::-1][:5]
        best_id = int(top5_idx[0])
        meta = get_sign_metadata(best_id)

        return {
            "class_id": best_id,
            "class_name": meta["class_name"],
            "macro_category": meta["macro_category"],
            "confidence": float(probs[best_id]),
            "roi_box": roi_box,
            "top5": [
                {
                    "class_id": int(idx),
                    "class_name": CLASS_NAMES[int(idx)],
                    "macro_category": MACRO_CATEGORY_MAP[int(idx)],
                    "probability": float(probs[int(idx)]),
                }
                for idx in top5_idx
            ],
            "preprocessed_rgb": (proc * 255.0).astype(np.uint8),
            "clahe_full_rgb": apply_clahe_color(rgb),
        }


def save_prediction_Grid(
    predictor: TrafficSignPredictor,
    image_paths: List[Path],
    output_path: Path,
) -> Path:
    """
    Run inference across multiple sample road signs and generate a visual
    multi-class classification summary grid with top-3 confidence bars.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    cols = min(4, n)
    rows = int(np.ceil(n / float(cols)))

    fig, axes = plt.subplots(rows * 2, cols, figsize=(3.8 * cols, 4.5 * rows), dpi=160)
    fig.suptitle(
        "Traffic Sign Recognition (GTSRB) — Multi-Class CNN Predictions & Category Breakdown",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    if rows * 2 == 2 and cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif axes.ndim == 1:
        axes = axes.reshape(rows * 2, cols)

    for idx, img_path in enumerate(image_paths):
        r = (idx // cols) * 2
        c = idx % cols
        bgr = cv2.imread(str(img_path))
        res = predictor.predict_image(bgr)

        ax_img = axes[r, c]
        ax_img.imshow(res["clahe_full_rgb"])
        ax_img.set_title(
            f"Pred: {res['class_name']}\nCategory: [{res['macro_category']}] ({res['confidence']*100:.1f}%)",
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
        )
        ax_img.axis("off")

        ax_bar = axes[r + 1, c]
        top3 = res["top5"][:3][::-1]
        names = [item["class_name"][:22] for item in top3]
        vals = [item["probability"] * 100.0 for item in top3]
        colors = ["#94A3B8", "#38BDF8", "#10B981"]
        bars = ax_bar.barh(names, vals, color=colors, height=0.55)
        ax_bar.set_xlim(0, 115)
        ax_bar.set_xlabel("Confidence (%)", fontsize=8)
        ax_bar.tick_params(axis="both", labelsize=7.5)
        ax_bar.grid(axis="x", alpha=0.25)
        for bar, v in zip(bars, vals):
            ax_bar.text(min(v + 2.0, 99.0), bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_webcam_realtime(predictor: TrafficSignPredictor, camera_id: int = 0) -> None:
    """Run real-time traffic sign detection and classification from a webcam stream."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with camera_id={camera_id}.")

    print("[INFO] Starting Real-Time Traffic Sign Recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_traffic_sign_rois(rgb, min_area=600)
        for (x1, y1, x2, y2) in boxes[:3]:
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue
            res = predictor.predict_image(crop_bgr)
            if res["confidence"] < 0.45:
                continue
            label = f"{res['macro_category']}: {res['class_name']} ({res['confidence']*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 185, 129), 2)

        cv2.imshow("Real-Time GTSRB Traffic Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Inference (Image / Demo / Webcam)")
    parser.add_argument("--image", type=Path, default=None, help="Path to a road-sign image to classify")
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH, help="Path to trained Keras model")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam inference")
    parser.add_argument("--demo", action="store_true", help="Generate sample road signs and run full visual demo")
    args = parser.parse_args()

    predictor = TrafficSignPredictor(model_path=args.model)

    if args.webcam:
        run_webcam_realtime(predictor)
        return

    if args.image is not None:
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image at '{args.image}'")
        res = predictor.predict_image(bgr)
        print("\n=== Traffic Sign Prediction Result ===")
        print(f"Input Image    : {args.image}")
        print(f"GTSRB Class ID : {res['class_id']}")
        print(f"Sign Name      : {res['class_name']}")
        print(f"Sign Category  : {res['macro_category']}")
        print(f"Confidence     : {res['confidence'] * 100:.2f}%")
        out_card = OUTPUTS_DIR / f"prediction_{args.image.stem}.png"
        save_prediction_Grid(predictor, [args.image], out_card)
        print(f"Saved visual prediction card to '{out_card}'")
        return

    # Default / --demo mode: generate sample signs and classify all of them
    sample_map = generate_sample_road_signs(SAMPLES_DIR)
    sample_paths = list(sample_map.values())
    grid_path = OUTPUTS_DIR / "sample_predictions_grid.png"
    save_prediction_Grid(predictor, sample_paths, grid_path)
    print(f"[INFO] Classified {len(sample_paths)} sample road signs -> '{grid_path}'")
    for cid, p in sample_map.items():
        res = predictor.predict_image(cv2.imread(str(p)))
        print(
            f"  - {p.name:<22} | Category: {res['macro_category']:<16} | "
            f"Predicted: {res['class_name']} ({res['confidence']*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
