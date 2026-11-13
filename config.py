"""
Configuration, Hyperparameters, and Class Definitions for
German Traffic Sign Recognition Benchmark (GTSRB) CNN Project.
"""

from pathlib import Path
from typing import Dict, Tuple

# Directory & File Paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "GTSRB"
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "gtsrb_traffic_sign_cnn.keras"

# Model & Preprocessing Hyperparameters

IMG_HEIGHT: int = 32
IMG_WIDTH: int = 32
IMG_CHANNELS: int = 3
IMG_SHAPE: Tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
NUM_CLASSES: int = 43

# Contrast Limited Adaptive Histogram Equalization (CLAHE) settings
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (4, 4)

# Training Hyperparameters
BATCH_SIZE: int = 64
EPOCHS: int = 30
INITIAL_LR: float = 1e-3
VALIDATION_SPLIT: float = 0.2
RANDOM_SEED: int = 42

# Official 43 GTSRB Traffic Sign Classes
# Reference: Stallkamp et al., IJCNN 2011 (https://benchmark.ini.rub.de/)

CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

# High-Level Macro Category Mapping
# Maps each of the 43 fine-grained GTSRB classes to core functional road sign
# categories (Speed Limit, Stop, Yield, Turn / Mandatory Direction, etc.)

MACRO_CATEGORY_MAP: Dict[int, str] = {
    **{cid: "Speed Limit" for cid in [0, 1, 2, 3, 4, 5, 7, 8]},
    14: "Stop",
    13: "Yield",
    **{cid: "Turn / Direction" for cid in [33, 34, 35, 36, 37, 38, 39, 40]},
    **{cid: "Prohibitory" for cid in [9, 10, 15, 16, 17]},
    **{cid: "Danger / Warning" for cid in range(18, 32)},
    **{cid: "Priority" for cid in [11, 12]},
    **{cid: "End of Restriction" for cid in [6, 32, 41, 42]},
}


def get_sign_metadata(class_id: int) -> Dict[str, str]:
    """Return human-readable class name and macro category for a GTSRB class ID."""
    if class_id not in CLASS_NAMES:
        raise ValueError(f"Invalid GTSRB class_id {class_id}. Expected 0..42.")
    return {
        "class_id": str(class_id),
        "class_name": CLASS_NAMES[class_id],
        "macro_category": MACRO_CATEGORY_MAP[class_id],
    }
