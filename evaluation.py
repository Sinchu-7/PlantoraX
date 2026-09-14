import streamlit as st
import os
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Model Evaluation")

st.write(
    "Performance evaluation of the MobileNetV2 plant disease "
    "classification model."
)

st.divider()

# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Deep Learning Model",
        "MobileNetV2"
    )

with col2:
    st.metric(
        "Number of Classes",
        "38"
    )

with col3:
    st.metric(
        "Dataset",
        "PlantVillage"
    )

st.divider()

# ============================================================
# EVALUATION METRICS
# ============================================================

st.subheader("📈 Performance Metrics")

# Actual evaluation values obtained from main.ipynb

accuracy = 90.44
precision = 93.26
recall = 87.67

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🎯 Accuracy",
        f"{accuracy:.2f}%"
    )

with col2:
    st.metric(
        "🔎 Precision",
        f"{precision:.2f}%"
    )

with col3:
    st.metric(
        "📌 Recall",
        f"{recall:.2f}%"
    )

st.divider()

# ============================================================
# PERFORMANCE BARS
# ============================================================

st.subheader("📊 Performance Overview")

st.write(f"**Accuracy — {accuracy:.2f}%**")
st.progress(int(accuracy))

st.write(f"**Precision — {precision:.2f}%**")
st.progress(int(precision))

st.write(f"**Recall — {recall:.2f}%**")
st.progress(int(recall))

st.divider()

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔢 Confusion Matrix")

confusion_matrix_path = os.path.join(
    "output",
    "confusion_matrix.png"
)

if os.path.exists(confusion_matrix_path):

    image = Image.open(confusion_matrix_path)

    st.image(
    image,
    caption="Plant Disease Classification Confusion Matrix",
    width=700
)

else:

    st.warning(
        "Confusion matrix image was not found in the output folder."
    )

st.divider()

# ============================================================
# METRIC EXPLANATION
# ============================================================

st.subheader("📚 Understanding the Metrics")

st.markdown("""
### 🎯 Accuracy

Accuracy represents the percentage of correctly classified samples
among all evaluated samples.

**Accuracy: 90.44%**

---

### 🔎 Precision

Precision indicates how many of the samples predicted as a particular
class were actually correct.

**Precision: 93.26%**

Higher precision means fewer false-positive predictions.

---

### 📌 Recall

Recall indicates how many of the actual samples belonging to a class
were successfully identified.

**Recall: 87.67%**

Higher recall means fewer actual disease cases are missed.

---

### 🔢 Confusion Matrix

The confusion matrix shows the relationship between the actual classes
and the classes predicted by the model.

The diagonal represents correct classifications, while values outside
the diagonal represent misclassifications.
""")

st.divider()

# ============================================================
# F1-SCORE
# ============================================================

st.subheader("ℹ️ F1-Score")

st.info(
    "The original evaluation output reports Accuracy, Precision and "
    "Recall. An F1-score was not directly reported in the notebook, "
    "so it is not presented as an evaluated result."
)

st.divider()

# ============================================================
# SUMMARY
# ============================================================

st.subheader("🌱 Evaluation Summary")

st.success(
    "The MobileNetV2 model achieved 90.44% accuracy, "
    "93.26% precision and 87.67% recall on the evaluated test data."
)

st.caption(
    "The evaluation values are taken from the evaluation results "
    "recorded in the project's main.ipynb notebook."
)