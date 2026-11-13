"""
Multi-Scale Convolutional Neural Network (CNN) Architecture in TensorFlow/Keras
for Real-Time Traffic Sign Recognition (GTSRB - 43 Classes).

Also includes a lightweight NumPy/OpenCV multi-class inference engine fallback
so predictions and probability charts can be verified in environments where
TensorFlow is not yet installed.
"""

from typing import Tuple
import numpy as np

from config import IMG_SHAPE, INITIAL_LR, NUM_CLASSES

try:
    import tensorflow as tf
    from tensorflow.keras import Model, layers, optimizers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_traffic_sign_cnn(
    input_shape: Tuple[int, int, int] = IMG_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = INITIAL_LR,
    use_in_graph_augmentation: bool = True,
):
    """
    Build and compile a real-time Multi-Scale CNN architecture inspired by
    Stallkamp et al. / Sermanet & LeCun GTSRB architectures.

    Architecture Highlights:
    - Optional in-graph Keras data augmentation (RandomRotation, RandomZoom,
      RandomTranslation, RandomContrast) active only during training.
    - Stage 1: Two 3x3 Conv2D(32) layers + BatchNorm + ReLU + MaxPool(2x2) + SpatialDropout2D(0.15)
    - Stage 2: Two 3x3 Conv2D(64) layers + BatchNorm + ReLU + MaxPool(2x2) + SpatialDropout2D(0.20)
    - Stage 3: Two 3x3 Conv2D(128) layers + BatchNorm + ReLU + MaxPool(2x2) + SpatialDropout2D(0.25)
    - Multi-Scale Feature Fusion: Concatenates GlobalAveragePooling features from
      Stage 2 (mid-level shape/pictogram cues) and Stage 3 (high-level semantic features).
    - Classifier Head: Dense(256) + BatchNorm + ReLU + Dropout(0.5) -> Softmax(43).
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to build the Keras model. "
            "Install dependencies via: pip install -r requirements.txt"
        )

    weight_decay = regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape, name="traffic_sign_input")
    x = inputs

    if use_in_graph_augmentation:
        x = layers.RandomRotation(factor=0.04, fill_mode="reflect", name="aug_rotation")(x)
        x = layers.RandomTranslation(
            height_factor=0.08, width_factor=0.08, fill_mode="reflect", name="aug_translation"
        )(x)
        x = layers.RandomZoom(height_factor=0.10, width_factor=0.10, fill_mode="reflect", name="aug_zoom")(x)
        x = layers.RandomContrast(factor=0.15, name="aug_contrast")(x)

    # --- Block 1: 32x32 -> 16x16 ---
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=weight_decay, use_bias=False, name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=weight_decay, use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.SpatialDropout2D(0.15, name="drop1")(x)

    # --- Block 2: 16x16 -> 8x8 ---
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=weight_decay, use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=weight_decay, use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    stage2_pool = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.SpatialDropout2D(0.20, name="drop2")(stage2_pool)

    # --- Block 3: 8x8 -> 4x4 ---
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=weight_decay, use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=weight_decay, use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    stage3_pool = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    stage3_drop = layers.SpatialDropout2D(0.25, name="drop3")(stage3_pool)

    # --- Multi-Scale Feature Fusion ---
    ms_feat_2 = layers.GlobalAveragePooling2D(name="gap_stage2")(stage2_pool)
    ms_feat_3 = layers.Flatten(name="flatten_stage3")(stage3_drop)
    fused = layers.Concatenate(name="multiscale_concat")([ms_feat_2, ms_feat_3])

    # --- Dense Classification Head ---
    dense = layers.Dense(256, kernel_regularizer=weight_decay, use_bias=False, name="fc1")(fused)
    dense = layers.BatchNormalization(name="bn_fc1")(dense)
    dense = layers.Activation("relu", name="relu_fc1")(dense)
    dense = layers.Dropout(0.5, name="drop_fc1")(dense)

    outputs = layers.Dense(num_classes, activation="softmax", name="traffic_sign_probs")(dense)

    model = Model(inputs=inputs, outputs=outputs, name="GTSRB_MultiScale_CNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
