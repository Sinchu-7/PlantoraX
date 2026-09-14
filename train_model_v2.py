import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Plant-Disease-Detection"

TRAIN_CSV = os.path.join(BASE_DIR, "data", "splits", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "splits", "validation.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "splits", "test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 38

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("PLANT DISEASE DETECTION - TRAINING V2")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset CSV files...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Training images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Test images       : {len(test_df)}")

# ============================================================
# CLASS NAMES
# ============================================================

class_names_path = os.path.join(OUTPUT_DIR, "class_names.json")

with open(class_names_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print(f"\nNumber of classes: {len(class_names)}")

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, but found {len(class_names)}"
    )

class_to_index = {
    class_name: i
    for i, class_name in enumerate(class_names)
}

train_df["label"] = train_df["class_name"].map(class_to_index)
val_df["label"] = val_df["class_name"].map(class_to_index)
test_df["label"] = test_df["class_name"].map(class_to_index)

if train_df["label"].isna().any():
    raise ValueError("Some training classes could not be mapped.")

if val_df["label"].isna().any():
    raise ValueError("Some validation classes could not be mapped.")

if test_df["label"].isna().any():
    raise ValueError("Some test classes could not be mapped.")

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):
    """
    Load image and apply MobileNetV2 preprocessing.
    """

    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32)

    # MobileNetV2 expects pixels approximately in [-1, 1]
    image = preprocess_input(image)

    return image, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15)
        ),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomContrast(
            0.15
        ),

        layers.RandomBrightness(
            0.10
        ),
    ],
    name="data_augmentation"
)


def augment_image(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")

train_paths = train_df["image_path"].values
train_labels = train_df["label"].values

val_paths = val_df["image_path"].values
val_labels = val_df["label"].values

test_paths = test_df["image_path"].values
test_labels = test_df["label"].values


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)


train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(train_df),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_ds = (
    val_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


test_ds = (
    test_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Datasets ready.")

# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

classes = np.arange(NUM_CLASSES)

raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

# Instead of using the full balanced weights,
# use square-root weighting.
#
# This reduces extremely large weights for rare classes
# while still giving minority classes extra importance.

class_weights_array = np.sqrt(raw_weights)

# Normalize around 1
class_weights_array = (
    class_weights_array /
    np.mean(class_weights_array)
)

class_weights = {
    i: float(class_weights_array[i])
    for i in range(NUM_CLASSES)
}

print("\nClass weights:")

for i, weight in class_weights.items():
    print(
        f"{i:2d} | "
        f"{class_names[i][:55]:55s} | "
        f"{weight:.3f}"
    )

# ============================================================
# BUILD MOBILE NET V2
# ============================================================

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Freeze entire pretrained network initially
base_model.trainable = False

inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="input_image"
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.Dropout(
    0.35,
    name="dropout_1"
)(x)

x = layers.Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = layers.Dropout(
    0.30,
    name="dropout_2"
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="disease_output"
)(x)

model = Model(
    inputs,
    outputs,
    name="PlantDisease_MobileNetV2_V2"
)

print("\nModel created.")

model.summary()

# ============================================================
# PHASE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 1 - TRAINING CLASSIFICATION HEAD")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

phase1_path = os.path.join(
    MODEL_DIR,
    "plant_disease_v2_phase1_best.keras"
)

phase1_checkpoint = ModelCheckpoint(
    phase1_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping_1 = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr_1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=[
        phase1_checkpoint,
        early_stopping_1,
        reduce_lr_1
    ]
)

# ============================================================
# PHASE 2 - CONTROLLED FINE TUNING
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 2 - CONTROLLED FINE-TUNING")
print("=" * 70)

# Unfreeze MobileNetV2
base_model.trainable = True

# Freeze most layers.
# Only the last 30 layers are trainable.

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print(
    f"\nTrainable MobileNetV2 layers: "
    f"{trainable_count}"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-6
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

phase2_path = os.path.join(
    MODEL_DIR,
    "plant_disease_v2_best.keras"
)

phase2_checkpoint = ModelCheckpoint(
    phase2_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping_2 = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr_2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=[
        phase2_checkpoint,
        early_stopping_2,
        reduce_lr_2
    ]
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING BEST V2 MODEL")
print("=" * 70)

if os.path.exists(phase2_path):

    model = tf.keras.models.load_model(
        phase2_path,
        compile=False
    )

    print(
        f"\nBest model loaded from:\n"
        f"{phase2_path}"
    )

elif os.path.exists(phase1_path):

    model = tf.keras.models.load_model(
        phase1_path,
        compile=False
    )

    print(
        f"\nPhase 2 checkpoint unavailable."
        f"\nLoaded Phase 1 best model."
    )

else:

    print(
        "\nWARNING: No checkpoint found."
        "\nUsing current model."
    )

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL V2 TEST EVALUATION")
print("=" * 70)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n")
print("=" * 70)
print("V2 RESULTS")
print("=" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "plant_disease_model_v2.keras"
)

model.save(
    final_model_path
)

print(
    f"\nV2 model saved to:\n"
    f"{final_model_path}"
)

# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

model_info = {

    "model_name":
        "MobileNetV2 Plant Disease Detection V2",

    "architecture":
        "MobileNetV2",

    "input_size":
        "128x128",

    "num_classes":
        NUM_CLASSES,

    "augmentation":
        [
            "RandomFlip",
            "RandomRotation",
            "RandomZoom",
            "RandomTranslation",
            "RandomContrast",
            "RandomBrightness"
        ],

    "class_weighting":
        "sqrt balanced class weights",

    "phase_1_learning_rate":
        0.0005,

    "phase_2_learning_rate":
        0.000005,

    "fine_tuned_layers":
        30,

    "batch_size":
        BATCH_SIZE,

    "test_images":
        len(test_df),

    "test_accuracy":
        float(test_accuracy),

    "test_loss":
        float(test_loss)
}

info_path = os.path.join(
    OUTPUT_DIR,
    "model_info_v2.json"
)

with open(
    info_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {}

for key, values in history_phase1.history.items():

    history_data[
        f"phase1_{key}"
    ] = [
        float(v)
        for v in values
    ]

for key, values in history_phase2.history.items():

    history_data[
        f"phase2_{key}"
    ] = [
        float(v)
        for v in values
    ]

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history_v2.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )

print(
    f"\nTraining history saved to:\n"
    f"{history_path}"
)

print("\n")
print("=" * 70)
print("TRAINING V2 COMPLETED")
print("=" * 70)
print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("\nNext step:")
print(
    "Run evaluate_model.py using the V2 model "
    "to generate detailed classification metrics."
)

print("=" * 70)