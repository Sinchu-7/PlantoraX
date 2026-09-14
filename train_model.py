import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Plant-Disease-Detection"

TRAIN_CSV = os.path.join(
    BASE_DIR, "data", "splits", "train.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR, "data", "splits", "validation.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR, "data", "splits", "test.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "output"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32

NUM_CLASSES = 38

INITIAL_EPOCHS = 15

FINE_TUNE_EPOCHS = 15

RANDOM_SEED = 42


# ============================================================
# SET RANDOM SEEDS
# ============================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("PLANT DISEASE DETECTION - MODEL TRAINING")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("\nGPU detected:")
    print(gpus)

else:

    print("\nNo GPU detected.")
    print("Training will use CPU.")


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading dataset CSV files...")

train_df = pd.read_csv(TRAIN_CSV)

val_df = pd.read_csv(VAL_CSV)

test_df = pd.read_csv(TEST_CSV)


print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images:       {len(test_df)}")


# ============================================================
# COLUMN COMPATIBILITY
# ============================================================

# The split script stores the class under class_name.

if "class_name" not in train_df.columns:

    if "class" in train_df.columns:

        train_df = train_df.rename(
            columns={"class": "class_name"}
        )

        val_df = val_df.rename(
            columns={"class": "class_name"}
        )

        test_df = test_df.rename(
            columns={"class": "class_name"}
        )

    else:

        raise ValueError(
            "Could not find class_name or class column."
        )


# ============================================================
# CREATE CLASS LIST
# ============================================================

class_names = sorted(
    train_df["class_name"].unique()
)

print(
    f"\nNumber of classes: {len(class_names)}"
)

if len(class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(
        f"{i:02d}: {class_name}"
    )


# ============================================================
# CLASS NAME -> INTEGER
# ============================================================

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


train_df["label"] = train_df[
    "class_name"
].map(class_to_index)

val_df["label"] = val_df[
    "class_name"
].map(class_to_index)

test_df["label"] = test_df[
    "class_name"
].map(class_to_index)


# ============================================================
# IMAGE PATH FUNCTION
# ============================================================

def get_full_path(path):

    path = str(path)

    if os.path.isabs(path):

        return path

    return os.path.join(
        BASE_DIR,
        path
    )


train_paths = train_df[
    "image_path"
].apply(get_full_path).values

val_paths = val_df[
    "image_path"
].apply(get_full_path).values

test_paths = test_df[
    "image_path"
].apply(get_full_path).values


train_labels = train_df[
    "label"
].values.astype(np.int32)

val_labels = val_df[
    "label"
].values.astype(np.int32)

test_labels = test_df[
    "label"
].values.astype(np.int32)


# ============================================================
# TF.DATA IMAGE LOADER
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)


train_dataset = (
    train_dataset
    .shuffle(
        buffer_size=5000,
        seed=RANDOM_SEED
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_dataset = (
    val_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


test_dataset = (
    test_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)


class_weights = {
    i: float(weight)
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i, weight in class_weights.items():

    print(
        f"{class_names[i]}: {weight:.3f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.15
        ),

        layers.RandomContrast(
            0.10
        ),

    ],

    name="data_augmentation"
)


# ============================================================
# MOBILE NET V2 BASE MODEL
# ============================================================

print("\nBuilding MobileNetV2 model...")


base_model = MobileNetV2(

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze base model initially

base_model.trainable = False


# ============================================================
# BUILD COMPLETE MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="input_image"
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


x = layers.Dropout(
    0.30
)(x)


x = layers.Dense(
    256,
    activation="relu",
    name="feature_dense"
)(x)


x = layers.Dropout(
    0.30
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="predictions"
)(x)


model = models.Model(
    inputs,
    outputs,
    name="PlantDisease_MobileNetV2"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_plant_disease_model.keras"
)


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        checkpoint_path,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=5,

        mode="max",

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-7,

        verbose=1
    )

]


# ============================================================
# PHASE 1 - TRANSFER LEARNING
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 70)


history_1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 2 - FINE TUNING")
print("=" * 70)


# Unfreeze MobileNetV2

base_model.trainable = True


# Keep early layers frozen.
# Fine-tune only the later layers.

fine_tune_from = 100


for layer in base_model.layers:

    if layer.name.startswith("block"):

        try:

            block_number = int(
                layer.name.split("_")[1]
            )

            if block_number < 13:

                layer.trainable = False

        except:

            pass


# Batch normalization layers are kept frozen
# for stable fine-tuning.

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# Recompile with smaller learning rate

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


history_2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    checkpoint_path
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = (
    best_model.evaluate(
        test_dataset,
        verbose=1
    )
)


print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "plant_disease_model.keras"
)


best_model.save(
    final_model_path
)


print(
    f"\n✓ Model saved to:\n"
    f"{final_model_path}"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    OUTPUT_DIR,
    "class_names.json"
)


with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


print(
    f"✓ Class names saved to:\n"
    f"{class_names_path}"
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "model": "MobileNetV2",

    "input_size": [
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    ],

    "num_classes": NUM_CLASSES,

    "classes": class_names,

    "train_images": len(train_df),

    "validation_images": len(val_df),

    "test_images": len(test_df),

    "test_accuracy": float(
        test_accuracy
    ),

    "test_loss": float(
        test_loss
    ),

    "initial_epochs": INITIAL_EPOCHS,

    "fine_tune_epochs": FINE_TUNE_EPOCHS,

    "class_balancing": True,

    "data_augmentation": True,

    "leakage_aware_split": True

}


model_info_path = os.path.join(
    OUTPUT_DIR,
    "model_info.json"
)


with open(
    model_info_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )


print(
    f"✓ Model information saved to:\n"
    f"{model_info_path}"
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nYour 38-class Plant Disease Detection "
    "model has been trained."
)

print("\nModel:")
print(final_model_path)

print("\nClass names:")
print(class_names_path)

print("\nModel information:")
print(model_info_path)

print("\nNext step:")
print(
    "Generate detailed classification report, "
    "confusion matrix and per-class metrics."
)

print("=" * 70)