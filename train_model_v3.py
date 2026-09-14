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
# PLANT DISEASE DETECTION - TRAINING V3
# ============================================================

print("=" * 70)
print("PLANT DISEASE DETECTION - MODEL TRAINING V3")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Plant-Disease-Detection"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "train.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "validation.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "test.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
# RANDOM SEEDS
# ============================================================

tf.random.set_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# GPU CHECK
# ============================================================

gpus = tf.config.list_physical_devices(
    "GPU"
)

if gpus:

    print("\nGPU detected:")
    print(gpus)

else:

    print("\nNo GPU detected.")
    print("Training will use CPU.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset CSV files...")


train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)


print(
    f"Training images:   {len(train_df)}"
)

print(
    f"Validation images: {len(val_df)}"
)

print(
    f"Test images:       {len(test_df)}"
)


# ============================================================
# COLUMN COMPATIBILITY
# ============================================================

if "class_name" not in train_df.columns:

    if "class" in train_df.columns:

        train_df = train_df.rename(
            columns={
                "class": "class_name"
            }
        )

        val_df = val_df.rename(
            columns={
                "class": "class_name"
            }
        )

        test_df = test_df.rename(
            columns={
                "class": "class_name"
            }
        )

    else:

        raise ValueError(
            "Could not find class_name or class column."
        )


# ============================================================
# CREATE CLASS LIST
# ============================================================

class_names = sorted(
    train_df[
        "class_name"
    ].unique()
)


print(
    f"\nNumber of classes: "
    f"{len(class_names)}"
)


if len(class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


print("\nClasses:")

for i, class_name in enumerate(
    class_names
):

    print(
        f"{i:02d}: {class_name}"
    )


# ============================================================
# CLASS -> INTEGER
# ============================================================

class_to_index = {

    name: index

    for index, name in enumerate(
        class_names
    )
}


train_df["label"] = train_df[
    "class_name"
].map(
    class_to_index
)

val_df["label"] = val_df[
    "class_name"
].map(
    class_to_index
)

test_df["label"] = test_df[
    "class_name"
].map(
    class_to_index
)


if train_df["label"].isna().any():

    raise ValueError(
        "Some training classes could not "
        "be converted to labels."
    )


if val_df["label"].isna().any():

    raise ValueError(
        "Some validation classes could not "
        "be converted to labels."
    )


if test_df["label"].isna().any():

    raise ValueError(
        "Some test classes could not "
        "be converted to labels."
    )


train_labels = train_df[
    "label"
].values.astype(
    np.int32
)

val_labels = val_df[
    "label"
].values.astype(
    np.int32
)

test_labels = test_df[
    "label"
].values.astype(
    np.int32
)


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
].apply(
    get_full_path
).values


val_paths = val_df[
    "image_path"
].apply(
    get_full_path
).values


test_paths = test_df[
    "image_path"
].apply(
    get_full_path
).values


# ============================================================
# CHECK IMAGE PATHS
# ============================================================

print("\nChecking image paths...")

missing_train = [
    p
    for p in train_paths
    if not os.path.exists(p)
]

missing_val = [
    p
    for p in val_paths
    if not os.path.exists(p)
]

missing_test = [
    p
    for p in test_paths
    if not os.path.exists(p)
]


if missing_train:

    raise FileNotFoundError(
        f"Missing training images: "
        f"{len(missing_train)}"
    )


if missing_val:

    raise FileNotFoundError(
        f"Missing validation images: "
        f"{len(missing_val)}"
    )


if missing_test:

    raise FileNotFoundError(
        f"Missing test images: "
        f"{len(missing_test)}"
    )


print("All image paths are valid.")


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(
    path,
    label
):

    image = tf.io.read_file(
        path
    )

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

    # MobileNetV2 preprocessing
    image = preprocess_input(
        image
    )

    return image, label


# ============================================================
# DATASETS
# ============================================================

print(
    "\nCreating TensorFlow datasets..."
)


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
        seed=RANDOM_SEED,
        reshuffle_each_iteration=True
    )

    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    .batch(
        BATCH_SIZE
    )

    .prefetch(
        tf.data.AUTOTUNE
    )
)


val_dataset = (
    val_dataset

    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    .batch(
        BATCH_SIZE
    )

    .prefetch(
        tf.data.AUTOTUNE
    )
)


test_dataset = (
    test_dataset

    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    .batch(
        BATCH_SIZE
    )

    .prefetch(
        tf.data.AUTOTUNE
    )
)


print("Datasets created successfully.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

print(
    "\nCalculating class weights..."
)


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(
        NUM_CLASSES
    ),

    y=train_labels
)


class_weights = {

    i: float(weight)

    for i, weight in enumerate(
        class_weights_array
    )

}


print(
    "\nClass weights:"
)

for i, weight in class_weights.items():

    print(
        f"{class_names[i]}: "
        f"{weight:.3f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

# IMPORTANT:
# This is intentionally kept close to V1.
#
# We are NOT using RandomBrightness.
# We are NOT using RandomTranslation.
#
# This allows us to compare V3 fairly against V1.

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
# BUILD MOBILE NET V2
# ============================================================

print(
    "\nBuilding MobileNetV2 model..."
)


base_model = MobileNetV2(

    input_shape=(

        IMAGE_SIZE[0],

        IMAGE_SIZE[1],

        3

    ),

    include_top=False,

    weights="imagenet"

)


# ============================================================
# FREEZE BASE MODEL
# ============================================================

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

    name="PlantDisease_MobileNetV2_V3"

)


# ============================================================
# PHASE 1 COMPILE
# ============================================================

print(
    "\nCompiling Phase 1..."
)


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

print(
    "\nModel summary:"
)

model.summary()


# ============================================================
# PHASE 1 CALLBACKS
# ============================================================

phase1_checkpoint_path = os.path.join(

    MODEL_DIR,

    "plant_disease_v3_phase1_best.keras"

)


phase1_callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        phase1_checkpoint_path,

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
# PHASE 1 TRAINING
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

    callbacks=phase1_callbacks

)


# ============================================================
# LOAD BEST PHASE 1 MODEL
# ============================================================

print(
    "\nLoading best Phase 1 model..."
)


if os.path.exists(
    phase1_checkpoint_path
):

    model = tf.keras.models.load_model(

        phase1_checkpoint_path,

        compile=False

    )

    print(
        "Best Phase 1 model loaded."
    )

else:

    print(
        "Phase 1 checkpoint not found."
    )


# ============================================================
# FIND MOBILE NET BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        if "MobileNetV2" in layer.name:

            base_model = layer

            break


if base_model is None:

    raise RuntimeError(
        "Could not find MobileNetV2 "
        "base model inside the trained model."
    )


# ============================================================
# PHASE 2 - CONTROLLED FINE TUNING
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 2 - CONTROLLED FINE-TUNING")
print("=" * 70)


# Start by making the MobileNetV2 trainable.

base_model.trainable = True


# ------------------------------------------------------------
# IMPORTANT:
# Freeze ALL layers first.
# ------------------------------------------------------------

for layer in base_model.layers:

    layer.trainable = False


# ------------------------------------------------------------
# Unfreeze ONLY the final 30 layers.
# ------------------------------------------------------------

number_of_layers_to_unfreeze = 30


for layer in base_model.layers[
    -number_of_layers_to_unfreeze:
]:

    layer.trainable = True


# ------------------------------------------------------------
# Keep BatchNormalization layers frozen.
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COUNT TRAINABLE LAYERS
# ============================================================

trainable_layers = [

    layer

    for layer in base_model.layers

    if layer.trainable

]


print(
    "\nTrainable MobileNetV2 layers: "
    f"{len(trainable_layers)}"
)


print(
    "Total MobileNetV2 layers: "
    f"{len(base_model.layers)}"
)


print(
    "\nFine-tuning only the final "
    f"{number_of_layers_to_unfreeze} layers "
    "while keeping BatchNormalization frozen."
)


# ============================================================
# PHASE 2 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# PHASE 2 CALLBACKS
# ============================================================

phase2_checkpoint_path = os.path.join(

    MODEL_DIR,

    "plant_disease_v3_best.keras"

)


phase2_callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        phase2_checkpoint_path,

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
# PHASE 2 TRAINING
# ============================================================

history_2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=phase2_callbacks

)


# ============================================================
# LOAD BEST V3 MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING BEST V3 MODEL")
print("=" * 70)


if os.path.exists(
    phase2_checkpoint_path
):

    best_model = tf.keras.models.load_model(

        phase2_checkpoint_path,

        compile=False

    )

    print(
        "\nBest V3 model loaded from:"
    )

    print(
        phase2_checkpoint_path
    )

elif os.path.exists(
    phase1_checkpoint_path
):

    best_model = tf.keras.models.load_model(

        phase1_checkpoint_path,

        compile=False

    )

    print(
        "\nPhase 2 checkpoint unavailable."
    )

    print(
        "Using best Phase 1 model."
    )

else:

    best_model = model

    print(
        "\nNo checkpoint found."
    )

    print(
        "Using current model."
    )


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL V3 TEST EVALUATION")
print("=" * 70)


best_model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


test_loss, test_accuracy = best_model.evaluate(

    test_dataset,

    verbose=1

)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("V3 RESULTS")
print("=" * 70)


print(
    f"Test Loss     : "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


print("\n")
print("BASELINE COMPARISON")
print("-" * 70)


print(
    "V1 Test Accuracy : 63.98%"
)


print(
    f"V3 Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


difference = (
    test_accuracy * 100
) - 63.98


print(
    f"Difference       : "
    f"{difference:+.2f} percentage points"
)


# ============================================================
# SAVE FINAL V3 MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "plant_disease_model_v3.keras"

)


best_model.save(
    final_model_path
)


print(
    "\nV3 model saved to:"
)


print(
    final_model_path
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
    "\nClass names saved to:"
)

print(
    class_names_path
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "model":
        "MobileNetV2 V3",

    "architecture":
        "MobileNetV2",

    "input_size":
        [
            IMAGE_SIZE[0],
            IMAGE_SIZE[1]
        ],

    "num_classes":
        NUM_CLASSES,

    "train_images":
        len(train_df),

    "validation_images":
        len(val_df),

    "test_images":
        len(test_df),

    "batch_size":
        BATCH_SIZE,

    "initial_epochs":
        INITIAL_EPOCHS,

    "fine_tune_epochs":
        FINE_TUNE_EPOCHS,

    "initial_learning_rate":
        1e-3,

    "fine_tuning_learning_rate":
        1e-5,

    "fine_tuned_last_layers":
        number_of_layers_to_unfreeze,

    "batch_normalization_frozen":
        True,

    "class_balancing":
        True,

    "data_augmentation":
        [
            "RandomFlip",
            "RandomRotation",
            "RandomZoom",
            "RandomContrast"
        ],

    "preprocessing":
        "MobileNetV2 preprocess_input",

    "test_accuracy":
        float(test_accuracy),

    "test_loss":
        float(test_loss),

    "baseline_v1_accuracy":
        0.6398,

    "difference_from_v1_percentage_points":
        float(difference)

}


model_info_path = os.path.join(

    OUTPUT_DIR,

    "model_info_v3.json"

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
    "\nModel information saved to:"
)

print(
    model_info_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {}


for key, values in history_1.history.items():

    history_data[
        f"phase1_{key}"
    ] = [

        float(value)

        for value in values

    ]


for key, values in history_2.history.items():

    history_data[
        f"phase2_{key}"
    ] = [

        float(value)

        for value in values

    ]


history_path = os.path.join(

    OUTPUT_DIR,

    "training_history_v3.json"

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
    "\nTraining history saved to:"
)

print(
    history_path
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING V3 COMPLETED")
print("=" * 70)


print(
    f"\nFinal V3 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"V1 Baseline Accuracy: "
    f"63.98%"
)


print(
    f"Difference: "
    f"{difference:+.2f} percentage points"
)


print("\nFinal V3 model:")
print(final_model_path)


print("\nNext step:")
print(
    "Run the detailed evaluation script "
    "on plant_disease_model_v3.keras."
)


print("=" * 70)