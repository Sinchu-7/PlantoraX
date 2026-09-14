import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Plant-Disease-Detection"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_model.keras"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "test.csv"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "output",
    "class_names.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("=" * 70)
print("LOADING CLASS NAMES")
print("=" * 70)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Number of classes: {len(class_names)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(test_df)}")

print("\nColumns:")
print(test_df.columns.tolist())


# ============================================================
# FIND IMAGE COLUMN
# ============================================================

if "image_path" in test_df.columns:
    image_column = "image_path"

elif "filepath" in test_df.columns:
    image_column = "filepath"

else:
    raise ValueError(
        "Could not find image path column in test.csv"
    )

print(f"\nImage column: {image_column}")


# ============================================================
# PREPARE NUMERIC LABELS
# ============================================================

print("\n" + "=" * 70)
print("PREPARING LABELS")
print("=" * 70)

if "label" in test_df.columns:

    test_labels = (
        test_df["label"]
        .astype(int)
        .values
    )

elif "class_id" in test_df.columns:

    test_labels = (
        test_df["class_id"]
        .astype(int)
        .values
    )

elif "class_name" in test_df.columns:

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    test_labels = (
        test_df["class_name"]
        .map(class_to_index)
        .astype(int)
        .values
    )

else:

    raise ValueError(
        "Could not find class label information in test.csv"
    )

print("Labels successfully prepared.")

print(
    f"Number of labels: {len(test_labels)}"
)


# ============================================================
# CHECK LABELS
# ============================================================

if len(test_labels) != len(test_df):

    raise ValueError(
        "Number of labels does not match number of test images."
    )

if np.any(test_labels < 0) or np.any(
    test_labels >= len(class_names)
):

    raise ValueError(
        "Some labels are outside the valid class range."
    )

print("Label validation successful.")


# ============================================================
# IMAGE LOADING FUNCTION
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

    # MobileNetV2 preprocessing
    image = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image)
    )

    return image, label


# ============================================================
# CREATE TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEST DATASET")
print("=" * 70)

test_paths = (
    test_df[image_column]
    .astype(str)
    .values
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)

print("Test dataset created successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

confidence = np.max(
    predictions,
    axis=1
)

print("Predictions generated successfully.")


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    test_labels,
    predicted_labels
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PLANT DISEASE DETECTION "
        "CLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test Images: {len(test_df)}\n"
    )

    f.write(
        f"Number of Classes: {len(class_names)}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write(report)

print(
    f"\nClassification report saved to:"
)

print(report_path)


# ============================================================
# REPORT AS CSV
# ============================================================

report_dict = classification_report(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_csv_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

report_df.to_csv(
    report_csv_path
)

print(
    f"Classification CSV saved to:"
)

print(report_csv_path)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CREATING CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

plt.figure(
    figsize=(24, 20)
)

plt.imshow(cm)

plt.title(
    "Plant Disease Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    np.arange(len(class_names)),
    class_names,
    fontsize=7
)

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to:"
)

print(cm_path)


# ============================================================
# PER-IMAGE PREDICTIONS
# ============================================================

results_df = pd.DataFrame({

    "image_path": test_paths,

    "actual_class": [
        class_names[i]
        for i in test_labels
    ],

    "predicted_class": [
        class_names[i]
        for i in predicted_labels
    ],

    "confidence": confidence,

    "correct": (
        test_labels == predicted_labels
    )
})


prediction_csv_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

results_df.to_csv(
    prediction_csv_path,
    index=False
)

print(
    f"\nPrediction results saved to:"
)

print(prediction_csv_path)


# ============================================================
# LOW CONFIDENCE PREDICTIONS
# ============================================================

low_confidence = results_df[
    results_df["confidence"] < 0.60
]

low_confidence_path = os.path.join(
    OUTPUT_DIR,
    "low_confidence_predictions.csv"
)

low_confidence.to_csv(
    low_confidence_path,
    index=False
)

print(
    f"\nLow-confidence predictions (<60%): "
    f"{len(low_confidence)}"
)

print(
    f"Saved to:"
)

print(low_confidence_path)


# ============================================================
# WRONG PREDICTIONS
# ============================================================

wrong_predictions = results_df[
    results_df["correct"] == False
]

wrong_path = os.path.join(
    OUTPUT_DIR,
    "wrong_predictions.csv"
)

wrong_predictions.to_csv(
    wrong_path,
    index=False
)

print(
    f"\nWrong predictions: "
    f"{len(wrong_predictions)}"
)

print(
    f"Saved to:"
)

print(wrong_path)


# ============================================================
# BEST AND WORST CLASSES
# ============================================================

class_metrics = report_df.loc[
    class_names,
    [
        "precision",
        "recall",
        "f1-score",
        "support"
    ]
].copy()

class_metrics = class_metrics.sort_values(
    "f1-score"
)


# ============================================================
# 10 WORST CLASSES
# ============================================================

print("\n" + "=" * 70)
print("10 WORST PERFORMING CLASSES")
print("=" * 70)

print(
    class_metrics
    .head(10)
    .to_string()
)


# ============================================================
# 10 BEST CLASSES
# ============================================================

print("\n" + "=" * 70)
print("10 BEST PERFORMING CLASSES")
print("=" * 70)

print(
    class_metrics
    .tail(10)
    .sort_values(
        "f1-score",
        ascending=False
    )
    .to_string()
)


# ============================================================
# SAVE PER-CLASS METRICS
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "per_class_metrics.csv"
)

class_metrics.to_csv(
    metrics_path
)

print(
    f"\nPer-class metrics saved to:"
)

print(metrics_path)


# ============================================================
# SUMMARY FILE
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PLANT DISEASE DETECTION "
        "MODEL EVALUATION SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Number of test images: "
        f"{len(test_df)}\n"
    )

    f.write(
        f"Number of classes: "
        f"{len(class_names)}\n"
    )

    f.write(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Wrong predictions: "
        f"{len(wrong_predictions)}\n"
    )

    f.write(
        f"Low-confidence predictions "
        f"(<60%): {len(low_confidence)}\n"
    )

    f.write("\n")

    f.write(
        "BEST PERFORMING CLASSES\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        class_metrics
        .tail(10)
        .sort_values(
            "f1-score",
            ascending=False
        )
        .to_string()
    )

    f.write("\n\n")

    f.write(
        "WORST PERFORMING CLASSES\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        class_metrics
        .head(10)
        .to_string()
    )


print(
    f"\nEvaluation summary saved to:"
)

print(summary_path)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"""
Test Accuracy: {accuracy * 100:.2f}%

Test Images: {len(test_df)}

Generated files:

1. classification_report.txt
2. classification_report.csv
3. confusion_matrix.png
4. test_predictions.csv
5. low_confidence_predictions.csv
6. wrong_predictions.csv
7. per_class_metrics.csv
8. evaluation_summary.txt

Location:

{OUTPUT_DIR}
"""
)

print("=" * 70)