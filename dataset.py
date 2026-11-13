"""
Dataset Loader and Synthetic GTSRB Road Sign Generator.

Supports:
1. Loading the official German Traffic Sign Recognition Benchmark (GTSRB) dataset
   from directory structure (Train/<0..42>/*.png) or CSV annotations (Train.csv, Test.csv).
2. Generating realistic synthetic road-sign images across key categories
   (Speed Limit, Stop, Yield, Turn Right/Left, No Entry, Caution, Priority Road, etc.)
   and all 43 GTSRB classes using OpenCV drawing primitives so the pipeline can be
   tested, demonstrated, and verified out of the box.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    CLASS_NAMES,
    DATA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_CLASSES,
    RANDOM_SEED,
    SAMPLES_DIR,
)
from preprocessing import OpenCVDataAugmenter, preprocess_image


def render_traffic_sign_icon(
    class_id: int,
    canvas_size: int = 128,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Render a synthetic road-sign image (RGB, uint8) for a given GTSRB class ID
    with consistent outdoor background and realistic illumination gradient.
    """
    det_rng = np.random.default_rng(RANDOM_SEED + class_id * 17)
    s = canvas_size
    k = s / 96.0
    c = s // 2
    img = np.zeros((s, s, 3), dtype=np.uint8)

    # Consistent sky/asphalt natural background gradient per class
    bg_top = det_rng.integers(75, 135, size=3, dtype=np.uint8)
    bg_bot = det_rng.integers(45, 95, size=3, dtype=np.uint8)
    for y in range(s):
        alpha = y / float(max(s - 1, 1))
        img[y, :] = ((1 - alpha) * bg_top + alpha * bg_bot).astype(np.uint8)

    # Colors in RGB
    RED = (220, 38, 38)
    BLUE = (29, 78, 216)
    YELLOW = (250, 204, 21)
    WHITE = (248, 250, 252)
    BLACK = (20, 24, 30)
    GRAY = (148, 163, 184)

    speed_map = {
        0: "20", 1: "30", 2: "50", 3: "60",
        4: "70", 5: "80", 7: "100", 8: "120",
    }

    if class_id in speed_map:
        # Red-bordered circular speed limit sign
        cv2.circle(img, (c, c), int(s * 0.40), RED, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (c, c), int(s * 0.30), WHITE, -1, lineType=cv2.LINE_AA)
        txt = speed_map[class_id]
        scale = (0.78 if len(txt) <= 2 else 0.58) * k
        thick = max(2, int(round(2.2 * k)))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(
            img, txt, (c - tw // 2, c + th // 2),
            cv2.FONT_HERSHEY_DUPLEX, scale, BLACK, thick, lineType=cv2.LINE_AA
        )

    elif class_id == 14:
        # Octagonal red STOP sign
        angles = np.deg2rad(np.arange(22.5, 360.0, 45.0))
        r_outer = int(s * 0.41)
        r_inner = int(s * 0.37)
        pts_outer = np.stack([c + r_outer * np.cos(angles), c + r_outer * np.sin(angles)], axis=-1).astype(np.int32)
        pts_inner = np.stack([c + r_inner * np.cos(angles), c + r_inner * np.sin(angles)], axis=-1).astype(np.int32)
        cv2.fillPoly(img, [pts_outer], WHITE, lineType=cv2.LINE_AA)
        cv2.fillPoly(img, [pts_inner], RED, lineType=cv2.LINE_AA)
        scale = 0.58 * k
        thick = max(2, int(round(2.0 * k)))
        (tw, th), _ = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(img, "STOP", (c - tw // 2, c + th // 2), cv2.FONT_HERSHEY_DUPLEX, scale, WHITE, thick, cv2.LINE_AA)

    elif class_id == 13:
        # Inverted red/white triangle YIELD sign
        pts_out = np.array([[int(s * 0.12), int(s * 0.20)], [int(s * 0.88), int(s * 0.20)], [c, int(s * 0.86)]], np.int32)
        pts_in = np.array([[int(s * 0.24), int(s * 0.27)], [int(s * 0.76), int(s * 0.27)], [c, int(s * 0.72)]], np.int32)
        cv2.fillPoly(img, [pts_out], RED, lineType=cv2.LINE_AA)
        cv2.fillPoly(img, [pts_in], WHITE, lineType=cv2.LINE_AA)

    elif class_id == 17:
        # No entry: red circle with horizontal white bar
        cv2.circle(img, (c, c), int(s * 0.40), RED, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (int(s * 0.20), int(s * 0.43)), (int(s * 0.80), int(s * 0.57)), WHITE, -1)

    elif class_id == 12:
        # Priority road: yellow diamond inside white border
        pts_out = np.array([[c, int(s * 0.10)], [int(s * 0.90), c], [c, int(s * 0.90)], [int(s * 0.10), c]], np.int32)
        pts_in = np.array([[c, int(s * 0.22)], [int(s * 0.78), c], [c, int(s * 0.78)], [int(s * 0.22), c]], np.int32)
        cv2.fillPoly(img, [pts_out], WHITE, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts_out], True, BLACK, max(2, int(2 * k)), lineType=cv2.LINE_AA)
        cv2.fillPoly(img, [pts_in], YELLOW, lineType=cv2.LINE_AA)

    elif class_id in range(33, 41):
        # Mandatory blue circular turn / direction signs
        cv2.circle(img, (c, c), int(s * 0.41), WHITE, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (c, c), int(s * 0.38), BLUE, -1, lineType=cv2.LINE_AA)
        aw = max(4, int(round(5.5 * k)))
        if class_id == 33:  # Turn right ahead
            cv2.arrowedLine(img, (int(s * 0.32), int(s * 0.65)), (int(s * 0.32), int(s * 0.42)), WHITE, aw, cv2.LINE_AA, tipLength=0.0)
            cv2.arrowedLine(img, (int(s * 0.32), int(s * 0.42)), (int(s * 0.72), int(s * 0.42)), WHITE, aw, cv2.LINE_AA, tipLength=0.35)
        elif class_id == 34:  # Turn left ahead
            cv2.arrowedLine(img, (int(s * 0.68), int(s * 0.65)), (int(s * 0.68), int(s * 0.42)), WHITE, aw, cv2.LINE_AA, tipLength=0.0)
            cv2.arrowedLine(img, (int(s * 0.68), int(s * 0.42)), (int(s * 0.28), int(s * 0.42)), WHITE, aw, cv2.LINE_AA, tipLength=0.35)
        elif class_id == 35:  # Ahead only
            cv2.arrowedLine(img, (c, int(s * 0.72)), (c, int(s * 0.26)), WHITE, aw, cv2.LINE_AA, tipLength=0.35)
        elif class_id == 38:  # Keep right
            cv2.arrowedLine(img, (int(s * 0.34), int(s * 0.34)), (int(s * 0.66), int(s * 0.66)), WHITE, aw, cv2.LINE_AA, tipLength=0.35)
        elif class_id == 39:  # Keep left
            cv2.arrowedLine(img, (int(s * 0.66), int(s * 0.34)), (int(s * 0.34), int(s * 0.66)), WHITE, aw, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.circle(img, (c, c), int(s * 0.20), WHITE, max(3, int(4 * k)), lineType=cv2.LINE_AA)
            cv2.putText(img, str(class_id), (c - int(14 * k), c + int(6 * k)), cv2.FONT_HERSHEY_DUPLEX, 0.5 * k, WHITE, max(2, int(2 * k)), cv2.LINE_AA)

    elif class_id in [11] or class_id in range(18, 32):
        # Upright warning red/white triangle
        pts_out = np.array([[c, int(s * 0.12)], [int(s * 0.88), int(s * 0.82)], [int(s * 0.12), int(s * 0.82)]], np.int32)
        pts_in = np.array([[c, int(s * 0.26)], [int(s * 0.76), int(s * 0.76)], [int(s * 0.24), int(s * 0.76)]], np.int32)
        cv2.fillPoly(img, [pts_out], RED, lineType=cv2.LINE_AA)
        cv2.fillPoly(img, [pts_in], WHITE, lineType=cv2.LINE_AA)
        symbol = "!" if class_id == 18 else str(class_id)
        scale = 0.58 * k
        thick = max(2, int(round(2.0 * k)))
        (tw, th), _ = cv2.getTextSize(symbol, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(img, symbol, (c - tw // 2, int(s * 0.68)), cv2.FONT_HERSHEY_DUPLEX, scale, BLACK, thick, cv2.LINE_AA)

    elif class_id in [6, 32, 41, 42]:
        # End-of-restriction white/gray circle with diagonal black slashes
        cv2.circle(img, (c, c), int(s * 0.40), GRAY, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (c, c), int(s * 0.36), WHITE, -1, lineType=cv2.LINE_AA)
        cv2.line(img, (int(s * 0.24), int(s * 0.76)), (int(s * 0.76), int(s * 0.24)), BLACK, max(4, int(5 * k)), cv2.LINE_AA)
        cv2.putText(img, str(class_id), (c - int(14 * k), c + int(8 * k)), cv2.FONT_HERSHEY_DUPLEX, 0.5 * k, BLACK, max(2, int(2 * k)), cv2.LINE_AA)

    else:
        # Other prohibitory red-ring signs (9, 10, 15, 16)
        cv2.circle(img, (c, c), int(s * 0.40), RED, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (c, c), int(s * 0.28), WHITE, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (int(s * 0.32), int(s * 0.44)), (int(s * 0.68), int(s * 0.56)), RED if class_id in [9, 15] else BLACK, -1)

    # Simulate uneven outdoor shadow gradient across the image so CLAHE contrast
    # normalization visibly restores uniform luminance
    x_grid = np.linspace(0.55, 1.15, s, dtype=np.float32)
    y_grid = np.linspace(0.65, 1.10, s, dtype=np.float32)
    illum = np.outer(y_grid, x_grid)[:, :, None]
    shadowed = np.clip(img.astype(np.float32) * illum, 0, 255).astype(np.uint8)
    return shadowed


def generate_sample_road_signs(samples_dir: Path = SAMPLES_DIR) -> Dict[int, Path]:
    """
    Generate and save representative road-sign sample images covering the prompt's
    key categories (Speed Limit, Stop, Yield, Turn Right/Left, No Entry, Caution)
    for immediate CLI and notebook testing.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    representative_classes = {
        2: "speed_limit_50.png",
        5: "speed_limit_80.png",
        14: "stop_sign.png",
        13: "yield_sign.png",
        33: "turn_right_ahead.png",
        34: "turn_left_ahead.png",
        17: "no_entry.png",
        18: "general_caution.png",
    }
    saved_paths: Dict[int, Path] = {}
    rng = np.random.default_rng(RANDOM_SEED)

    for cid, filename in representative_classes.items():
        rgb = render_traffic_sign_icon(cid, canvas_size=128, rng=rng)
        out_path = samples_dir / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        saved_paths[cid] = out_path

    return saved_paths


def generate_synthetic_gtsrb_dataset(
    samples_per_class: int = 30,
    use_clahe: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a balanced 43-class synthetic GTSRB dataset in memory for fast
    pipeline verification or offline training when the 300MB Kaggle archive is
    not present locally.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    augmenter = OpenCVDataAugmenter(seed=RANDOM_SEED)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for class_id in range(NUM_CLASSES):
        base_icon = render_traffic_sign_icon(class_id, canvas_size=128, rng=rng)
        for _ in range(samples_per_class):
            aug_rgb = augmenter.augment(base_icon)
            proc = preprocess_image(
                aug_rgb,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                is_bgr=False,
                use_clahe=use_clahe,
                norm_mode="minmax",
            )
            X_list.append(proc)
            y_list.append(class_id)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split_idx = int(0.8 * len(X))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def load_gtsrb_from_directory(
    data_dir: Path = DATA_DIR,
    use_clahe: bool = True,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the official GTSRB dataset from `data_dir`.
    Supports:
    - `data_dir/Train/<0..42>/*` directory layout
    - Optional `data_dir/Train.csv` and `data_dir/Test.csv` with ROI coordinates
      (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`, `ClassId`, `Path`).
    Falls back to `generate_synthetic_gtsrb_dataset` if `data_dir` is not found.
    """
    train_csv = data_dir / "Train.csv"
    train_folder = data_dir / "Train"

    if not train_folder.exists() and not train_csv.exists():
        print(
            f"[INFO] Official GTSRB directory not found at '{data_dir}'. "
            "Generating synthetic 43-class GTSRB dataset for immediate execution..."
        )
        return generate_synthetic_gtsrb_dataset(samples_per_class=35, use_clahe=use_clahe)

    images: List[np.ndarray] = []
    labels: List[int] = []

    if train_csv.exists():
        counts: Dict[int, int] = {}
        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["ClassId"])
                if max_samples_per_class and counts.get(cid, 0) >= max_samples_per_class:
                    continue
                img_path = data_dir / row["Path"]
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                roi = (
                    int(row.get("Roi.X1", 0)),
                    int(row.get("Roi.Y1", 0)),
                    int(row.get("Roi.X2", bgr.shape[1])),
                    int(row.get("Roi.Y2", bgr.shape[0])),
                )
                proc = preprocess_image(
                    bgr,
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    roi_box=roi,
                    is_bgr=True,
                    use_clahe=use_clahe,
                )
                images.append(proc)
                labels.append(cid)
                counts[cid] = counts.get(cid, 0) + 1
    else:
        for cid in range(NUM_CLASSES):
            class_dir = train_folder / str(cid)
            if not class_dir.exists():
                continue
            files = sorted(class_dir.glob("*.*"))
            if max_samples_per_class:
                files = files[:max_samples_per_class]
            for img_path in files:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                proc = preprocess_image(
                    bgr,
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    is_bgr=True,
                    use_clahe=use_clahe,
                )
                images.append(proc)
                labels.append(cid)

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split_idx = int(0.8 * len(X))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights to mitigate GTSRB class imbalance
    across the 43 traffic sign categories.
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    total = float(len(y_train))
    n_classes = float(len(unique_classes))
    return {
        int(cid): float(total / (n_classes * max(cnt, 1)))
        for cid, cnt in zip(unique_classes, counts)
    }
