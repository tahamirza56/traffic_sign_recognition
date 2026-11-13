# Traffic Sign Recognition (GTSRB) — Advanced Computer Vision, Data Augmentation & Real-Time CNNs

An end-to-end multi-class traffic sign classification and real-time recognition system built with [Python](https://www.python.org/), [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/), [OpenCV](https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/), [NumPy](https://numpy.org/doc/stable/), and [Matplotlib](https://matplotlib.org/stable/index.html) on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://zenodo.org/records/13741936).

## Project Highlights & Core Capabilities

1. **Multi-Class & Macro-Category Traffic Sign Recognition**:
   - Classifies road signs across all **43 official classes** of the [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (39,209 training images and 12,630 test images as documented in [Stallkamp et al., IJCNN 2011](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)).
   - Automatically maps fine-grained predictions into functional driving categories: **Speed Limit**, **Stop**, **Yield**, **Turn / Direction**, **Prohibitory**, **Danger / Warning**, **Priority**, and **End of Restriction**.
2. **Image Preprocessing & Contrast Normalization**:
   - Crops Region-of-Interest (ROI) bounding boxes and resizes inputs to `32x32x3` tensors using adaptive area/cubic interpolation.
   - Applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** (`clipLimit=2.0`, `tileGridSize=(4, 4)`) to the luminance ($L$) channel in **CIE LAB color space** via [OpenCV CLAHE](https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/), normalizing harsh shadows, glare, and low-light road conditions while preserving red, blue, and yellow traffic sign hues.
3. **Domain-Safe Real-Time Data Augmentation**:
   - Implements on-the-fly geometric and photometric transformations (`OpenCVDataAugmenter` and in-graph [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)): random rotation ($\pm 15^\circ$), zoom ($0.88\times - 1.12\times$), perspective shear ($\pm 8^\circ$), translation ($\pm 10\%$), gamma illumination jitter, and motion/defocus blur.
   - Intentionally excludes horizontal and vertical flips so directional signs (`Turn right ahead` vs. `Turn left ahead`) retain their true semantic labels.
4. **Multi-Scale Convolutional Neural Network (CNN) Architecture**:
   - Inspired by multi-scale traffic sign architectures in [Stallkamp et al., IJCNN 2011](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf), combining three convolutional stages (`32 -> 64 -> 128` filters with `BatchNormalization`, `ReLU`, `MaxPooling2D`, and `SpatialDropout2D`) with **Multi-Scale Feature Fusion** that concatenates mid-level Stage-2 features and high-level Stage-3 features prior to the dense softmax classifier head.

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `config.py` | Hyperparameters, file paths, all 43 [GTSRB](https://zenodo.org/records/13741936) class names, and functional macro-category mappings (`Speed Limit`, `Stop`, `Yield`, `Turn / Direction`, etc.). |
| `preprocessing.py` | ROI cropping, CIE LAB [OpenCV CLAHE](https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/) contrast normalization, intensity standardization, `OpenCVDataAugmenter`, HSV ROI sign detector, and [Matplotlib](https://matplotlib.org/stable/index.html) pipeline visualizer. |
| `dataset.py` | Official [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) folder/CSV loader (`Train/`, `Train.csv`, `Test.csv`), inverse-frequency class weight calculator, and synthetic 43-class traffic sign generator for instant offline testing. |
| `model.py` | Multi-scale CNN architecture (`GTSRB_MultiScale_CNN`) built with [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) and [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers). |
| `train.py` | End-to-end training and evaluation pipeline with `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, macro precision/recall/F1 reporting, and training curve plots. |
| `predict.py` | CLI inference script supporting single-image prediction (`--image`), batch sample demonstration (`--demo`), and real-time webcam traffic sign detection & classification (`--webcam`). |
| `Traffic_Sign_Recognition_GTSRB.ipynb` | Self-contained Jupyter / Google Colab notebook walking through exploratory visualization, CLAHE preprocessing, augmentation, CNN training, evaluation, and inference. |
| `requirements.txt` | Python package dependencies for local or cloud execution. |

## Quickstart & Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the GTSRB Dataset (or Use Built-in Synthetic Generator)

- **Option A — Official GTSRB Dataset**: Download the dataset from [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) or [Zenodo GTSRB](https://zenodo.org/records/13741936) and extract it into `data/GTSRB/` so that `data/GTSRB/Train/<0..42>/` and `data/GTSRB/Train.csv` are present.
- **Option B — Immediate Out-of-the-Box Execution**: If `data/GTSRB/` is not present, `dataset.py` and `train.py` automatically synthesize a 43-class road-sign dataset and sample test images (`samples/`) so you can verify the entire codebase immediately.

### 3. Train the Multi-Scale CNN

```bash
python train.py --data-dir data/GTSRB --epochs 30 --batch-size 64 --lr 0.001
```

- Best model weights are saved to `models/gtsrb_traffic_sign_cnn.keras`.
- Training and validation accuracy/loss plots are saved to `outputs/training_curves.png`.

### 4. Predict Traffic Sign Category from an Input Image

```bash
# Run the full visual demo across Speed Limit, Stop, Yield, Turn, and Caution signs:
python predict.py --demo

# Classify any specific input road-sign image:
python predict.py --image samples/stop_sign.png

# Launch real-time webcam traffic sign detection & recognition:
python predict.py --webcam
```

## CNN Architecture Summary

| Stage / Layer | Output Shape | Configuration & Details |
| :--- | :--- | :--- |
| Input Layer | `(32, 32, 3)` | CLAHE-normalized RGB tensor in `[0.0, 1.0]` ([NumPy](https://numpy.org/doc/stable/) `float32`). |
| In-Graph Augmentation | `(32, 32, 3)` | `RandomRotation(0.04)`, `RandomTranslation(0.08)`, `RandomZoom(0.10)`, `RandomContrast(0.15)` ([tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)). |
| Conv Stage 1 | `(16, 16, 32)` | `2 x [Conv2D(32, 3x3, same) + BatchNorm + ReLU] -> MaxPool(2x2) -> SpatialDropout2D(0.15)`. |
| Conv Stage 2 | `(8, 8, 64)` | `2 x [Conv2D(64, 3x3, same) + BatchNorm + ReLU] -> MaxPool(2x2) -> SpatialDropout2D(0.20)`. |
| Conv Stage 3 | `(4, 4, 128)` | `2 x [Conv2D(128, 3x3, same) + BatchNorm + ReLU] -> MaxPool(2x2) -> SpatialDropout2D(0.25)`. |
| Multi-Scale Fusion | `(2112,)` | Concatenates `GlobalAveragePooling2D(Stage 2)` (`64-d`) + `Flatten(Stage 3)` (`2048-d`). |
| Classifier Head | `(43,)` | `Dense(256) -> BatchNorm -> ReLU -> Dropout(0.5) -> Dense(43, Softmax)`. |
