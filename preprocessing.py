"""
Image Preprocessing, Contrast Normalization (CLAHE), and Data Augmentation
using OpenCV, NumPy, and Matplotlib for Traffic Sign Recognition (GTSRB).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
)


def crop_roi(
    image: np.ndarray,
    roi_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Crop the traffic sign Region of Interest (ROI) using GTSRB bounding box
    coordinates (x1, y1, x2, y2) when available.
    """
    if roi_box is None:
        return image
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi_box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return image[y1:y2, x1:x2]


def apply_clahe_color(
    rgb_image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
    luminance (L) channel in CIE LAB color space.

    Operating on the L channel corrects uneven road illumination, shadows,
    glare, and night/fog low contrast while preserving the crucial red/blue/yellow
    chromaticity of traffic signs.
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image * 255.0 if rgb_image.max() <= 1.0 else rgb_image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


def apply_global_hist_eq(rgb_image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization on the Y (luminance) channel in YCrCb space.
    Provided as an alternative contrast normalization baseline.
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def normalize_intensity(
    rgb_image: np.ndarray,
    mode: str = "minmax",
) -> np.ndarray:
    """
    Scale pixel intensities to float32.
    - 'minmax': scales to [0.0, 1.0]
    - 'zscore': per-image zero-mean unit-variance standardization
    """
    img_float = rgb_image.astype(np.float32)
    if mode == "zscore":
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        std = np.std(img_float, axis=(0, 1), keepdims=True) + 1e-6
        return (img_float - mean) / std
    return img_float / 255.0


def preprocess_image(
    image_bgr_or_rgb: np.ndarray,
    target_size: Tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
    roi_box: Optional[Tuple[int, int, int, int]] = None,
    is_bgr: bool = False,
    use_clahe: bool = True,
    norm_mode: str = "minmax",
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single traffic sign image:
    1. Convert BGR -> RGB (if loaded via cv2.imread)
    2. Crop ROI bounding box (if provided)
    3. Resize to target_size (32x32 default) using area/cubic interpolation
    4. Contrast normalization via LAB-space CLAHE
    5. Intensity normalization to float32 [0, 1] or z-score
    """
    img = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_BGR2RGB) if is_bgr else image_bgr_or_rgb.copy()
    img = crop_roi(img, roi_box)

    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (h > target_size[1] or w > target_size[0]) else cv2.INTER_CUBIC
    img_resized = cv2.resize(img, target_size, interpolation=interp)

    if use_clahe:
        img_resized = apply_clahe_color(img_resized)

    return normalize_intensity(img_resized, mode=norm_mode)


class OpenCVDataAugmenter:
    """
    Real-time OpenCV photometric & geometric data augmentation pipeline
    tailored for traffic signs.

    NOTE: Horizontal/vertical flips are intentionally excluded because flipping
    directional traffic signs (e.g. 'Turn right ahead' vs 'Turn left ahead')
    changes their semantic ground-truth class.
    """

    def __init__(
        self,
        rotation_deg: float = 15.0,
        translate_frac: float = 0.10,
        scale_range: Tuple[float, float] = (0.88, 1.12),
        shear_deg: float = 8.0,
        brightness_range: Tuple[float, float] = (0.75, 1.25),
        seed: Optional[int] = None,
    ) -> None:
        self.rotation_deg = rotation_deg
        self.translate_frac = translate_frac
        self.scale_range = scale_range
        self.shear_deg = shear_deg
        self.brightness_range = brightness_range
        self.rng = np.random.default_rng(seed)

    def random_affine(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation, scaling, shear, and translation."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        angle = self.rng.uniform(-self.rotation_deg, self.rotation_deg)
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

        tx = self.rng.uniform(-self.translate_frac, self.translate_frac) * w
        ty = self.rng.uniform(-self.translate_frac, self.translate_frac) * h
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty

        shear_rad = np.deg2rad(self.rng.uniform(-self.shear_deg, self.shear_deg))
        shear_mat = np.array([[1.0, np.tan(shear_rad), 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        warped = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped = cv2.warpAffine(
            warped,
            shear_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped

    def random_photometric(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness/gamma shift and mild motion/Gaussian blur."""
        is_float = image.dtype != np.uint8
        work = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8) if is_float else image.copy()

        # Gamma & brightness jitter
        gamma = self.rng.uniform(self.brightness_range[0], self.brightness_range[1])
        inv_gamma = 1.0 / max(gamma, 1e-4)
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        work = cv2.LUT(work, lut)

        # Occasional mild Gaussian blur simulating motion or distance defocus
        if self.rng.random() < 0.25:
            work = cv2.GaussianBlur(work, (3, 3), sigmaX=0.6)

        return (work.astype(np.float32) / 255.0) if is_float else work

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply combined geometric and photometric augmentation to a single image."""
        out = self.random_affine(image)
        out = self.random_photometric(out)
        return out

    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """Augment a 4D batch of images (N, H, W, C) on the fly."""
        return np.stack([self.augment(img) for img in images], axis=0)


def detect_traffic_sign_rois(
    road_image_rgb: np.ndarray,
    min_area: int = 400,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect candidate traffic sign bounding boxes (x1, y1, x2, y2) in an uncropped
    road scene using HSV chromaticity segmentation (red, blue, yellow traffic sign
    colors) combined with aspect-ratio and contour filtering.
    """
    hsv = cv2.cvtColor(road_image_rgb, cv2.COLOR_RGB2HSV)

    # Red wraps around 0/180 in OpenCV HSV
    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([95, 80, 50]), np.array([135, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([15, 80, 70]), np.array([35, 255, 255]))

    combined_mask = mask_red1 | mask_red2 | mask_blue | mask_yellow
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    h_img, w_img = road_image_rgb.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(max(h, 1))
        if 0.65 <= aspect <= 1.45:
            pad_x = int(0.08 * w)
            pad_y = int(0.08 * h)
            boxes.append((
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(w_img, x + w + pad_x),
                min(h_img, y + h + pad_y),
            ))

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def visualize_preprocessing_pipeline(
    raw_rgb_image: np.ndarray,
    output_path: Path,
    sign_title: str = "Traffic Sign Sample",
) -> Path:
    """
    Generate a publication-quality Matplotlib visual comparing:
    1. Original low-contrast / shadowed road sign image
    2. Global Histogram Equalization (YCrCb)
    3. LAB-space CLAHE Contrast Normalization
    4. Four real-time OpenCV data augmentation variants
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized = cv2.resize(raw_rgb_image, (64, 64), interpolation=cv2.INTER_CUBIC)
    hist_eq = apply_global_hist_eq(resized)
    clahe_img = apply_clahe_color(resized)

    augmenter = OpenCVDataAugmenter(seed=42)
    aug_variants = [augmenter.augment(clahe_img) for _ in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), dpi=160)
    fig.suptitle(
        f"Traffic Sign Preprocessing, CLAHE Contrast Normalization & Augmentation ({sign_title})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Top row: Raw -> Global HistEq -> CLAHE -> Luminance Histograms
    axes[0, 0].imshow(resized)
    axes[0, 0].set_title("1. Raw Input Image\n(Uneven Lighting)", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(hist_eq)
    axes[0, 1].set_title("2. Global Hist. Eq.\n(YCrCb Luminance)", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(clahe_img)
    axes[0, 2].set_title("3. CLAHE Normalized\n(LAB L-Channel, 4x4 Tiles)", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")

    # Luminance histogram comparison
    raw_l = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)[:, :, 0].ravel()
    clahe_l = cv2.cvtColor(clahe_img, cv2.COLOR_RGB2LAB)[:, :, 0].ravel()
    axes[0, 3].hist(raw_l, bins=32, range=(0, 256), alpha=0.55, label="Raw L", color="#64748B", density=True)
    axes[0, 3].hist(clahe_l, bins=32, range=(0, 256), alpha=0.65, label="CLAHE L", color="#0EA5E9", density=True)
    axes[0, 3].set_title("Luminance Distribution", fontsize=10, fontweight="bold")
    axes[0, 3].set_xlabel("Pixel Intensity (0-255)", fontsize=8)
    axes[0, 3].set_ylabel("Density", fontsize=8)
    axes[0, 3].legend(fontsize=8)
    axes[0, 3].grid(alpha=0.25)

    # Bottom row: 4 Augmented variants
    aug_labels = [
        "Aug 1: Rotation + Zoom",
        "Aug 2: Perspective Shear",
        "Aug 3: Gamma / Shadow Shift",
        "Aug 4: Combined Affine + Blur",
    ]
    for idx in range(4):
        axes[1, idx].imshow(aug_variants[idx])
        axes[1, idx].set_title(aug_labels[idx], fontsize=10, fontweight="bold")
        axes[1, idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
