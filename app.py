import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
import csv
from datetime import datetime
from io import BytesIO

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    "output",
    "class_names.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    "output",
    "confusion_matrix.png"
)

HISTORY_PATH = "prediction_history.csv"


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "show_register" not in st.session_state:
    st.session_state.show_register = False

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "Admin User",
        "email": "admin@gmail.com",
        "phone": "",
        "photo": None
    }


# ============================================================
# PASSWORD HASH
# ============================================================

def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# USERS
# ============================================================

if "users" not in st.session_state:
    st.session_state.users = {
        "admin": hash_password("admin123")
    }


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0b0f0d;
    color: #f5f5f5;
}

section[data-testid="stSidebar"] {
    background-color: #111713;
    border-right: 1px solid #29362e;
}

section[data-testid="stSidebar"] > div {
    background-color: #111713;
}

h1, h2, h3, h4, h5, h6 {
    color: white;
}

p, label {
    color: #d8ddd9;
}

/* BUTTONS */

.stButton > button {
    width: 100%;
    min-height: 42px;
    border-radius: 8px;
    border: 1px solid #33443a;
    background-color: #18221c;
    color: white;
    font-weight: 500;
}

.stButton > button:hover {
    border-color: #55c878;
    color: #55c878;
}

/* INPUTS */

.stTextInput input {
    background-color: #141a16;
    color: white;
    border: 1px solid #33443a;
    border-radius: 8px;
}

.stTextInput input:focus {
    border-color: #55c878;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background-color: #121914;
    border-radius: 10px;
}

/* PROFILE */

.profile-name {
    font-size: 19px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}

.profile-email {
    font-size: 13px;
    color: #aeb8b1;
    margin-bottom: 3px;
}

.profile-phone {
    font-size: 13px;
    color: #aeb8b1;
}

/* SECTION */

.section-title {
    font-size: 13px;
    font-weight: 700;
    color: #8d9b92;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
    margin-bottom: 8px;
}

/* HOME CARDS */

.home-card {
    background-color: #131a16;
    border: 1px solid #29362e;
    border-radius: 14px;
    padding: 20px;
    height: 100%;
}

.home-card-title {
    font-size: 15px;
    color: #aeb8b1;
    margin-bottom: 8px;
}

.home-card-value {
    font-size: 30px;
    font-weight: 700;
    color: #63d987;
}

/* METRIC CARDS */

.metric-card {
    background-color: #131a16;
    border: 1px solid #29362e;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #63d987;
}

.metric-label {
    font-size: 13px;
    color: #aeb8b1;
}

/* LOGIN */

.login-title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: white;
}

.login-subtitle {
    text-align: center;
    color: #9daaa2;
    margin-bottom: 20px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    st.write("")
    st.write("")
    st.write("")

    left, center, right = st.columns(
        [1, 1.4, 1]
    )

    with center:

        st.markdown(
            '<div class="login-title">🌿 Plant Disease Detection</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="login-subtitle">AI-powered plant disease identification</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 🔐 Login")

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        show_password = st.checkbox(
            "Show password"
        )

        password = st.text_input(
            "Password",
            type="default" if show_password else "password",
            placeholder="Enter password"
        )

        if st.button(
            "🔐 Login",
            use_container_width=True
        ):

            if username in st.session_state.users:

                if (
                    st.session_state.users[username]
                    == hash_password(password)
                ):

                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "🏠 Home"

                    st.experimental_rerun()

                else:

                    st.error(
                        "❌ Incorrect password."
                    )

            else:

                st.error(
                    "❌ Username not found."
                )

        st.write("")

        st.info(
            "Demo Login: Username = admin | Password = admin123"
        )

        st.write("")

        if not st.session_state.show_register:

            if st.button(
                "➕ Create New Account"
            ):

                st.session_state.show_register = True

                st.experimental_rerun()

        else:

            st.markdown(
                "### 📝 Create Account"
            )

            new_username = st.text_input(
                "New Username"
            )

            new_password = st.text_input(
                "New Password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "Create Account"
                ):

                    if not new_username or not new_password:

                        st.error(
                            "Please fill all fields."
                        )

                    elif new_username in st.session_state.users:

                        st.error(
                            "Username already exists."
                        )

                    elif new_password != confirm_password:

                        st.error(
                            "Passwords do not match."
                        )

                    else:

                        st.session_state.users[
                            new_username
                        ] = hash_password(
                            new_password
                        )

                        st.success(
                            "Account created successfully!"
                        )

                        st.session_state.show_register = False

                        st.experimental_rerun()

            with col2:

                if st.button(
                    "← Back to Login"
                ):

                    st.session_state.show_register = False

                    st.experimental_rerun()

    st.stop()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_plant_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return load_model(
        MODEL_PATH
    )


# ============================================================
# CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if os.path.exists(
        CLASS_NAMES_PATH
    ):

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, dict):
            return list(data.values())

        return data

    return [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]


model = load_plant_model()
class_names = load_class_names()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_disease_name(name):

    name = str(name)

    name = name.replace(
        "___",
        " - "
    )

    name = name.replace(
        "_",
        " "
    )

    return name


def get_risk(confidence):

    if confidence >= 80:
        return "High Confidence"

    elif confidence >= 60:
        return "Medium Confidence"

    return "Low Confidence"


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    username,
    filename,
    prediction,
    confidence,
    risk
):

    record = {
        "Date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Username": username,
        "Image": filename,
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%",
        "Risk": risk
    }

    # --------------------------------------------------------
    # SAVE IMMEDIATELY IN SESSION
    # --------------------------------------------------------

    st.session_state.prediction_history.append(
        record
    )

    # --------------------------------------------------------
    # SAVE TO CSV
    # --------------------------------------------------------

    file_exists = os.path.exists(
        HISTORY_PATH
    )

    try:

        with open(
            HISTORY_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Date",
                    "Username",
                    "Image",
                    "Prediction",
                    "Confidence",
                    "Risk"
                ]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                record
            )

    except Exception as e:

        st.warning(
            f"History saved for this session, "
            f"but CSV could not be updated: {e}"
        )


# ============================================================
# READ HISTORY
# ============================================================

def read_history():

    all_records = []

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    if os.path.exists(
        HISTORY_PATH
    ):

        try:

            csv_df = pd.read_csv(
                HISTORY_PATH
            )

            if not csv_df.empty:

                all_records.extend(
                    csv_df.to_dict(
                        orient="records"
                    )
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # LOAD SESSION RECORDS
    # --------------------------------------------------------

    if st.session_state.prediction_history:

        all_records.extend(
            st.session_state.prediction_history
        )

    if not all_records:

        return pd.DataFrame()

    df = pd.DataFrame(
        all_records
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Username",
        "Image",
        "Prediction",
        "Confidence",
        "Risk"
    ]

    existing_columns = [
        col
        for col in required_columns
        if col in df.columns
    ]

    if existing_columns:

        df = df.drop_duplicates(
            subset=existing_columns,
            keep="last"
        )

    return df


# ============================================================
# CREATE EXCEL
# ============================================================

def create_excel(df):

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Prediction History"
        )

    buffer.seek(0)

    return buffer


# ============================================================
# CREATE PDF
# ============================================================

def create_pdf(df):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=8
    )

    story = []

    story.append(
        Paragraph(
            "Plant Disease Detection - Prediction History",
            title_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    if df.empty:

        story.append(
            Paragraph(
                "No prediction history available.",
                body_style
            )
        )

    else:

        table_data = []

        # Header

        table_data.append(
            [
                Paragraph(
                    str(column),
                    body_style
                )
                for column in df.columns
            ]
        )

        # Rows

        for _, row in df.iterrows():

            table_data.append(
                [
                    Paragraph(
                        str(value),
                        body_style
                    )
                    for value in row.tolist()
                ]
            )

        table = Table(
            table_data,
            repeatRows=1
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#243329")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#f2f2f2")
                    ]
                )
            ])
        )

        story.append(
            table
        )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    st.markdown(
        "### 👤 Profile"
    )

    profile = st.session_state.profile

    if profile["photo"]:

        try:

            profile_image = Image.open(
                BytesIO(
                    profile["photo"]
                )
            )

            st.image(
                profile_image,
                width=110
            )

        except Exception:

            st.write("👤")

    else:

        st.markdown(
            """
            <div style="
                width:90px;
                height:90px;
                border-radius:50%;
                background:#26342b;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:42px;
                margin-bottom:10px;
            ">
                👤
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="profile-name">{profile["name"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="profile-email">📧 {profile["email"]}</div>',
        unsafe_allow_html=True
    )

    phone_display = (
        profile["phone"]
        if profile["phone"]
        else "Phone not added"
    )

    st.markdown(
        f'<div class="profile-phone">📱 {phone_display}</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # EDIT PROFILE
    # --------------------------------------------------------

    with st.expander(
        "✏️ Edit Profile"
    ):

        profile_name = st.text_input(
            "Name",
            value=profile["name"]
        )

        profile_email = st.text_input(
            "Gmail",
            value=profile["email"]
        )

        profile_phone = st.text_input(
            "Phone Number",
            value=profile["phone"]
        )

        uploaded_profile_photo = st.file_uploader(
            "Profile Photo",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="profile_photo"
        )

        if st.button(
            "💾 Save Profile"
        ):

            st.session_state.profile[
                "name"
            ] = profile_name

            st.session_state.profile[
                "email"
            ] = profile_email

            st.session_state.profile[
                "phone"
            ] = profile_phone

            if uploaded_profile_photo:

                st.session_state.profile[
                    "photo"
                ] = uploaded_profile_photo.getvalue()

            st.success(
                "Profile updated!"
            )

            st.experimental_rerun()

    st.divider()

    # --------------------------------------------------------
    # ALL APPS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">All Apps</div>',
        unsafe_allow_html=True
    )

    pages = [
        "🏠 Home",
        "🌿 Disease Detection",
        "📊 Prediction History",
        "📈 Model Evaluation",
        "ℹ️ About"
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(
            st.session_state.page
        ),
        label_visibility="collapsed"
    )

    st.session_state.page = selected_page

    st.divider()

    st.caption(
        f"Logged in as: {st.session_state.username}"
    )

    if st.button(
        "🚪 Logout"
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.experimental_rerun()


# ============================================================
# CURRENT PAGE
# ============================================================

page = st.session_state.page


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🌿 Plant Disease Detection"
    )

    st.subheader(
        "Welcome to the AI Plant Disease Detection System"
    )

    st.write(
        """
        Upload an image of a plant leaf and the deep learning
        model will analyze the image and predict the most
        likely disease.
        """
    )

    st.write("")

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">
                    Model Accuracy
                </div>
                <div class="home-card-value">
                    63.98%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">
                    Plant Classes
                </div>
                <div class="home-card-value">
                    38
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">
                    Test Images
                </div>
                <div class="home-card-value">
                    8,266
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-title">
                    Architecture
                </div>
                <div class="home-card-value">
                    MobileNetV2
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.divider()

    # --------------------------------------------------------
    # QUICK START
    # --------------------------------------------------------

    st.header(
        "🚀 Quick Start"
    )

    quick_col1, quick_col2 = st.columns(2)

    with quick_col1:

        st.markdown(
            """
            ### 🔬 Detect a Disease

            Upload a plant leaf image and get:

            - Predicted disease
            - Confidence score
            - Top 3 predictions
            - Recommendation
            - Prediction history
            """
        )

        if st.button(
            "🌿 Start Disease Detection",
            use_container_width=True
        ):

            st.session_state.page = (
                "🌿 Disease Detection"
            )

            st.experimental_rerun()

    with quick_col2:

        st.markdown(
            """
            ### 📊 View Prediction History

            Review previous predictions and
            download them as:

            - CSV
            - Excel
            - PDF
            """
        )

        if st.button(
            "📊 Open Prediction History",
            use_container_width=True
        ):

            st.session_state.page = (
                "📊 Prediction History"
            )

            st.experimental_rerun()

    st.write("")

    st.divider()

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    st.header(
        "🧠 About the Model"
    )

    st.write(
        """
        The application uses **MobileNetV2 Transfer Learning**
        with a custom classification head.

        The model was trained using the PlantVillage dataset
        containing 38 plant disease and healthy classes.

        The final selected V1 model achieved **63.98% accuracy**
        on the held-out test dataset.
        """
    )


# ============================================================
# DISEASE DETECTION
# ============================================================

elif page == "🌿 Disease Detection":

    st.title(
        "🌿 Disease Detection"
    )

    st.write(
        "Upload a plant leaf image to identify the disease."
    )

    if model is None:

        st.error(
            "❌ Model file not found."
        )

        st.info(
            f"Expected model: {MODEL_PATH}"
        )

        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_column_width=True
            )

        with col2:

            st.subheader(
                "📋 Image Information"
            )

            st.write(
                f"**File:** {uploaded_file.name}"
            )

            st.write(
                f"**Image Size:** "
                f"{image.size[0]} × {image.size[1]}"
            )

            st.write(
                "**Model Input:** 128 × 128"
            )

            st.write(
                f"**Classes:** {len(class_names)}"
            )

        st.write("")

        if st.button(
            "🔬 Predict Disease",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing leaf image..."
            ):

                # --------------------------------------------
                # PREPROCESS
                # --------------------------------------------

                resized_image = image.resize(
                    (128, 128)
                )

                image_array = np.array(
                    resized_image
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                image_array = preprocess_input(
                    image_array
                )

                # --------------------------------------------
                # PREDICT
                # --------------------------------------------

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                top_indices = np.argsort(
                    predictions
                )[-3:][::-1]

                top_predictions = []

                for index in top_indices:

                    disease = class_names[index]

                    confidence = (
                        predictions[index] * 100
                    )

                    top_predictions.append(
                        (
                            disease,
                            confidence
                        )
                    )

                predicted_disease = (
                    top_predictions[0][0]
                )

                confidence = (
                    top_predictions[0][1]
                )

                risk = get_risk(
                    confidence
                )

                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                st.divider()

                st.header(
                    "🎯 Prediction Result"
                )

                result_col1, result_col2 = st.columns(
                    [2, 1]
                )

                with result_col1:

                    st.subheader(
                        clean_disease_name(
                            predicted_disease
                        )
                    )

                    st.write(
                        f"Confidence: "
                        f"**{confidence:.2f}%**"
                    )

                    if confidence >= 80:

                        st.success(
                            "🟢 High Confidence"
                        )

                    elif confidence >= 60:

                        st.warning(
                            "🟡 Medium Confidence"
                        )

                    else:

                        st.error(
                            "🔴 Low Confidence"
                        )

                with result_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                # --------------------------------------------
                # TOP 3
                # --------------------------------------------

                st.subheader(
                    "📊 Top 3 Predictions"
                )

                for rank, (
                    disease,
                    score
                ) in enumerate(
                    top_predictions,
                    start=1
                ):

                    st.write(
                        f"**{rank}. "
                        f"{clean_disease_name(disease)}** "
                        f"— {score:.2f}%"
                    )

                    st.progress(
                        min(
                            score / 100,
                            1.0
                        )
                    )

                # --------------------------------------------
                # RECOMMENDATION
                # --------------------------------------------

                st.subheader(
                    "💡 Recommendation"
                )

                if confidence >= 80:

                    st.success(
                        "The model has high confidence "
                        "in this prediction. Expert "
                        "confirmation is still recommended."
                    )

                elif confidence >= 60:

                    st.warning(
                        "The model has moderate confidence. "
                        "Consider checking another image "
                        "or consulting an expert."
                    )

                else:

                    st.info(
                        "The model has low confidence. "
                        "Try uploading a clearer image "
                        "with better lighting."
                    )

                # --------------------------------------------
                # SAVE HISTORY
                # --------------------------------------------

                save_prediction(
                    st.session_state.username,
                    uploaded_file.name,
                    clean_disease_name(
                        predicted_disease
                    ),
                    confidence,
                    risk
                )

                st.success(
                    "✅ Prediction saved to history."
                )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📊 Prediction History":

    st.title(
        "📊 Prediction History"
    )

    st.write(
        "View and download your previous disease predictions."
    )

    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------

    history_df = read_history()

    if history_df.empty:

        st.info(
            "No predictions have been recorded yet."
        )

        st.write(
            "Go to **Disease Detection**, upload an image, "
            "and click **Predict Disease**."
        )

    else:

        # ----------------------------------------------------
        # CURRENT USER HISTORY
        # ----------------------------------------------------

        if "Username" in history_df.columns:

            user_history = history_df[
                history_df["Username"].astype(str)
                == str(
                    st.session_state.username
                )
            ].copy()

        else:

            user_history = history_df.copy()

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if user_history.empty:

            st.info(
                "No predictions found for your account."
            )

            st.write(
                f"Current account: "
                f"**{st.session_state.username}**"
            )

        else:

            # ------------------------------------------------
            # SORT
            # ------------------------------------------------

            if "Date" in user_history.columns:

                user_history = user_history.sort_values(
                    by="Date",
                    ascending=False
                )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Summary"
            )

            total_predictions = len(
                user_history
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Predictions",
                    total_predictions
                )

            with col2:

                if "Prediction" in user_history.columns:

                    most_common = (
                        user_history[
                            "Prediction"
                        ]
                        .value_counts()
                        .index[0]
                    )

                    st.metric(
                        "Most Predicted",
                        most_common
                    )

                else:

                    st.metric(
                        "Most Predicted",
                        "-"
                    )

            with col3:

                if "Risk" in user_history.columns:

                    high_count = (
                        user_history[
                            "Risk"
                        ]
                        == "High Confidence"
                    ).sum()

                    st.metric(
                        "High Confidence",
                        high_count
                    )

                else:

                    st.metric(
                        "High Confidence",
                        0
                    )

            st.write("")

            st.divider()

            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            st.subheader(
                "📋 Your Prediction Records"
            )

            st.dataframe(
                user_history,
                use_container_width=True,
                hide_index=True
            )

            st.write("")

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "⬇️ Download Prediction History"
            )

            st.write(
                "Select the format you want to download:"
            )

            # =================================================
            # CSV
            # =================================================

            csv_data = user_history.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            # =================================================
            # EXCEL
            # =================================================

            excel_buffer = create_excel(
                user_history
            )

            # =================================================
            # PDF
            # =================================================

            pdf_buffer = create_pdf(
                user_history
            )

            # =================================================
            # BUTTONS
            # =================================================

            download_col1, download_col2, download_col3 = st.columns(
                3
            )

            with download_col1:

                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with download_col2:

                st.download_button(
                    label="📊 Download Excel",
                    data=excel_buffer,
                    file_name="prediction_history.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )

            with download_col3:

                st.download_button(
                    label="📕 Download PDF",
                    data=pdf_buffer,
                    file_name="prediction_history.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.write("")

            st.success(
                "✅ Prediction history is ready to download."
            )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.title(
        "📈 Model Evaluation"
    )

    st.write(
        "Performance of the final selected V1 MobileNetV2 model."
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    63.98%
                </div>
                <div class="metric-label">
                    Accuracy
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    71.58%
                </div>
                <div class="metric-label">
                    Weighted Precision
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    63.98%
                </div>
                <div class="metric-label">
                    Weighted Recall
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    61.09%
                </div>
                <div class="metric-label">
                    Weighted F1
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🧠 Model Information"
    )

    model_info = pd.DataFrame({
        "Parameter": [
            "Architecture",
            "Base Model",
            "Input Size",
            "Number of Classes",
            "Optimizer",
            "Training Strategy",
            "Test Samples"
        ],
        "Value": [
            "MobileNetV2 + Classification Head",
            "MobileNetV2 ImageNet",
            "128 × 128",
            "38",
            "Adam",
            "Transfer Learning + Fine Tuning",
            "8,266"
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "📊 Confusion Matrix"
    )

    if os.path.exists(
        CONFUSION_MATRIX_PATH
    ):

        st.image(
            CONFUSION_MATRIX_PATH,
            caption="Test Set Confusion Matrix",
            use_column_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## 🌿 Plant Disease Detection

        This project is a deep learning based application
        designed to identify plant diseases from leaf images.

        The system uses **MobileNetV2 Transfer Learning**
        and the PlantVillage dataset containing **38 classes**.
        """
    )

    st.divider()

    st.subheader(
        "⚙️ Technology Stack"
    )

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "TensorFlow",
            "Keras",
            "MobileNetV2",
            "Streamlit",
            "Pandas",
            "NumPy",
            "ReportLab",
            "OpenPyXL"
        ],
        "Purpose": [
            "Programming Language",
            "Deep Learning Framework",
            "Model Development",
            "Transfer Learning",
            "Web Application",
            "Data Processing",
            "Numerical Processing",
            "PDF Generation",
            "Excel Generation"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "🔄 System Workflow"
    )

    st.write(
        """
        **1. Upload Plant Leaf Image**

        ↓

        **2. Image Preprocessing**

        ↓

        **3. MobileNetV2 Feature Extraction**

        ↓

        **4. Classification**

        ↓

        **5. Disease Prediction**

        ↓

        **6. Confidence Score + Top-3 Predictions**

        ↓

        **7. Save Prediction History**
        """
    )

    st.divider()

    st.subheader(
        "📊 Actual Model Performance"
    )

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1 Score"
        ],
        "Score": [
            "63.98%",
            "71.58%",
            "63.98%",
            "61.09%"
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "⚠️ Disclaimer"
    )

    st.info(
        """
        This application is intended for educational and
        demonstration purposes. Predictions should not be
        treated as a substitute for professional agricultural
        diagnosis.
        """
    )

    st.write("")

    st.caption(
        "Plant Disease Detection | Deep Learning Mini Project"
    )