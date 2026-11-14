# 🌿 Plant Disease Detection using MobileNetV2

## 🚀 Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

To download PlantVillage into class folders that you can browse from the **Choose a plant image** uploader:

```bash
pip install -r requirements-dataset.txt
python download_dataset.py
```

### Run the tests

```bash
python -m unittest discover -s tests
```

The download is large (about 827 MB before export) and is saved under `data/plantvillage` by default. Dataset images and trained model files are intentionally ignored by Git and are not included in commits. To use the images, choose one of the exported files from `data/plantvillage/<class>/` in the app.

Use `python download_dataset.py --max-images 3` to verify the downloader on a small sample before exporting the full dataset.

### Deploy the Streamlit app

This repository is a Python/Streamlit application, not a static website. Netlify cannot run the Streamlit server, so its deploy-preview check is not a valid deployment target for this app. Use the included `render.yaml` blueprint to deploy the Dockerized app to Render, or connect the repository to Streamlit Community Cloud and set the main file to `app.py`.

For Render, create a new Blueprint from this repository and set `PLANT_APP_USERNAME` and `PLANT_APP_PASSWORD` in the service environment before accessing the app. The blueprint enables production mode, uses the committed model files, and checks Streamlit's health endpoint. Choose a service plan with enough memory for TensorFlow and the model.

### Production configuration

Set `APP_ENV=production`, `PLANT_APP_USERNAME`, and `PLANT_APP_PASSWORD` in the hosting provider's environment settings. Production mode disables the demo account and self-service registration. Set `PLANT_MODEL_PATH` if the model is stored outside `models/`, and set `PLANT_APP_DATA_DIR` to a persistent writable directory if prediction history must survive redeployments. Do not use the development demo credentials in production.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An AI-based plant disease detection system using MobileNetV2 Transfer Learning and the PlantVillage dataset.**

</div>

---

# 📌 Project Overview

Plant diseases can significantly affect crop production and agricultural productivity. Early identification of plant diseases can help farmers take appropriate action before the disease spreads.

This project presents a **Deep Learning based Plant Disease Detection System** that classifies plant leaf images into different disease and healthy categories.

The system uses **MobileNetV2**, a lightweight Convolutional Neural Network architecture, with **Transfer Learning**. A Streamlit-based web application provides an easy-to-use interface where users can upload a leaf image and receive a predicted disease along with its confidence score and top-3 predictions.

The final model supports **38 plant disease and healthy classes**.

---

# 🎯 Objectives

The main objectives of this project are:

- To develop an automated plant disease classification system.
- To use Deep Learning for image-based disease detection.
- To apply Transfer Learning using MobileNetV2.
- To classify plant leaves into 38 different classes.
- To provide prediction confidence and top-3 predictions.
- To develop a simple web interface using Streamlit.
- To maintain prediction history.
- To allow prediction history to be downloaded as CSV, Excel, and PDF.
- To evaluate the trained model using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

# ✨ Key Features

## 🌿 Disease Detection

Upload a plant leaf image and obtain the predicted disease or healthy class.

## 📊 Confidence Score

The application displays the confidence score of the predicted class.

## 🔝 Top-3 Predictions

The system displays the three most probable disease classes.

## 📋 Prediction History

Previous predictions are stored with:

- Date and time
- Username
- Image filename
- Predicted disease
- Confidence
- Confidence level

## 📥 Download Reports

Prediction history can be downloaded as:

- CSV
- Excel
- PDF

## 👤 User Login

The application provides a simple login and account creation system.
Development accounts created in the app require passwords of at least 12
characters; their passwords are stored as salted PBKDF2 hashes in session
memory. Production mode continues to use the configured environment account.

## 👤 User Profile

Users can maintain:

- Name
- Gmail
- Phone number
- Profile photo

## 📈 Model Evaluation

The application displays:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 🏗️ System Architecture

The overall workflow of the system is:

```text
                 Plant Leaf Image
                        │
                        ▼
               Image Preprocessing
                        │
                        ▼
                 Resize to 128×128
                        │
                        ▼
              MobileNetV2 Backbone
                  ImageNet Weights
                        │
                        ▼
             Global Average Pooling
                        │
                        ▼
                    Dropout
                     (0.30)
                        │
                        ▼
                  Dense Layer
                  256 Neurons
                     ReLU
                        │
                        ▼
                    Dropout
                     (0.30)
                        │
                        ▼
              Dense Layer - 38
                   Softmax
                        │
                        ▼
              Disease Prediction
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Confidence Score       Top-3 Predictions
             │
             ▼
       Save Prediction History

       🧠 Model Architecture

The project uses MobileNetV2 as the main Convolutional Neural Network (CNN) backbone.

MobileNetV2 was selected because it is a lightweight and efficient architecture suitable for image classification while requiring fewer computational resources compared with larger CNN architectures.

The final model architecture consists of:

Input Image
   │
   ▼
128 × 128 × 3
   │
   ▼
MobileNetV2
ImageNet Pre-trained Weights
   │
   ▼
GlobalAveragePooling2D
   │
   ▼
Dropout (0.30)
   │
   ▼
Dense Layer (256 neurons)
ReLU Activation
   │
   ▼
Dropout (0.30)
   │
   ▼
Dense Layer (38 neurons)
Softmax Activation
   │
   ▼
Predicted Plant Disease Class

Model Components

| Component           | Description            |
| ------------------- | ---------------------- |
| Input Size          | 128 × 128 × 3          |
| Backbone            | MobileNetV2            |
| Pre-trained Weights | ImageNet               |
| Pooling             | Global Average Pooling |
| Dropout             | 0.30                   |
| Dense Layer         | 256 neurons            |
| Activation          | ReLU                   |
| Output Layer        | 38 neurons             |
| Output Activation   | Softmax                |


🔄 Transfer Learning

Transfer Learning is used to utilize the knowledge learned by MobileNetV2 from the ImageNet dataset.

The model was trained in two phases.

Phase 1 — Feature Extraction

In the first phase, the pre-trained MobileNetV2 backbone was frozen.

Only the newly added classification layers were trained to learn features specific to plant disease classification.

Phase 2 — Fine-Tuning

In the second phase, selected deeper layers of MobileNetV2 were unfrozen and fine-tuned using a smaller learning rate.

This allowed the model to adapt the pre-trained ImageNet features to the PlantVillage plant disease dataset.

The use of Transfer Learning reduces training time and allows the model to utilize useful visual features learned from a large-scale dataset.

🖼️ Data Preprocessing

Before an image is passed to the model, it goes through the following preprocessing steps:

The uploaded image is loaded.
The image is converted into RGB format.
The image is resized to 128 × 128 pixels.
MobileNetV2 preprocessing is applied.
The processed image is passed to the trained model.
Data Augmentation

Data augmentation was applied during training to improve the model's ability to generalize to different leaf images.

The following augmentation techniques were used:

Random horizontal flipping
Random rotation
Random zoom
Random contrast adjustment

These transformations help the model handle variations in leaf orientation, position, and appearance

📚 Dataset

The project uses the PlantVillage dataset for training and evaluating the plant disease classification model.

The dataset contains images of healthy and diseased plant leaves belonging to 38 different classes.

Dataset Statistics

| Dataset        | Number of Images |
| -------------- | ---------------: |
| Training Set   |           37,847 |
| Validation Set |            8,192 |
| Test Set       |            8,266 |
| **Total**      |       **54,305** |


Dataset Classes

The dataset contains 38 plant disease and healthy categories covering multiple plant species, including:

Apple
Blueberry
Cherry
Corn
Grape
Orange
Peach
Pepper
Potato
Raspberry
Soybean
Squash
Strawberry
Tomato

Both healthy and diseased leaf categories are included.

🔀 Dataset Split

The complete dataset was divided into three subsets:

Training Set

37,847 images

Used to train the MobileNetV2-based model.

Validation Set

8,192 images

Used during training to monitor model performance and select the best model.

Test Set

8,266 images

Used for the final evaluation of the trained model on unseen images.

Dataset Distribution

                 PlantVillage Dataset
                       54,305
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
        Training      Validation       Test
         37,847          8,192         8,266

The dataset split was performed before model training so that the test set remained separate from the training process.


⚙️ Model Training

The MobileNetV2-based model was trained using a two-phase transfer learning strategy.

Phase 1 — Training the Classification Head

In the first phase, the MobileNetV2 base model was kept frozen.

The newly added classification layers were trained using:

Optimizer: Adam
Learning Rate: 0.001
Maximum Epochs: 15
Loss Function: Categorical Cross-Entropy
Output Activation: Softmax
Class Weighting: Balanced Class Weights

The best validation accuracy achieved during Phase 1 was approximately 54.64%.

Phase 2 — Fine-Tuning

In the second phase, selected deeper layers of MobileNetV2 were unfrozen.

A smaller learning rate was used to make gradual adjustments to the pre-trained weights.

The configuration included:

Optimizer: Adam
Learning Rate: 0.00001
Maximum Epochs: 15
Selected deeper MobileNetV2 layers fine-tuned
Batch Normalization layers kept frozen

The best validation accuracy achieved during Phase 2 was approximately 62.49%.

🛠️ Training Callbacks

The following callbacks were used during training:

Model Checkpoint

Saves the model with the best validation performance.

Early Stopping

Stops training when validation performance stops improving for several epochs.

Reduce Learning Rate on Plateau

Reduces the learning rate when validation performance stops improving.

These techniques help improve training stability and prevent unnecessary training.

📊 Training Performance

The training process showed improvement after fine-tuning the MobileNetV2 backbone.

| Training Phase | Best Validation Accuracy |
| -------------- | -----------------------: |
| Phase 1        |                   54.64% |
| Phase 2        |                   62.49% |

The model showed improved validation performance after fine-tuning compared with the initial feature-extraction phase.

📊 Model Evaluation & Results

The final trained model was evaluated using the separate test dataset containing 8,266 images.

The following evaluation metrics were used:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

📈 Overall Performance

| Metric             |      Score |
| ------------------ | ---------: |
| Test Accuracy      | **63.98%** |
| Weighted Precision | **71.58%** |
| Weighted Recall    | **63.98%** |
| Weighted F1-score  | **61.09%** |
| Macro Precision    | **66.39%** |
| Macro Recall       | **61.51%** |
| Macro F1-score     | **56.80%** |

The final model achieved an overall test accuracy of 63.98% across 38 plant disease and healthy classes.

🔲 Confusion Matrix

A confusion matrix was generated to analyze the classification performance across all 38 classes.

It helps identify which plant disease classes are correctly classified and which classes are frequently confused with one another.

🔍 Class-wise Performance

The model performed particularly well on several classes.

Some of the stronger-performing classes include:

| Plant Disease / Class   | F1-score |
| ----------------------- | -------: |
| Corn - Healthy          |   96.57% |
| Corn - Common Rust      |   91.81% |
| Peach - Bacterial Spot  |   88.08% |
| Grape - Leaf Blight     |   87.06% |
| Squash - Powdery Mildew |   85.42% |
| Soybean - Healthy       |   81.18% |

Performance varied across different disease categories. Some visually similar classes were more difficult for the model to distinguish.

⚠️ Model Limitations

The evaluation showed weaker performance for some visually similar disease categories.

Examples include:

Potato - Healthy
Potato - Late Blight
Tomato - Leaf Mold
Tomato - Bacterial Spot
Tomato - Spider Mites
Tomato - Early Blight

These results indicate that visually similar disease patterns can be challenging for the model to distinguish.

The model should therefore be considered a prototype plant disease classification system and not a replacement for expert agricultural diagnosis.

🌐 Streamlit Web Application

A web-based interface was developed using Streamlit to make the plant disease detection system easy to use.

The application allows users to interact with the trained Deep Learning model without requiring them to write Python code.

🏠 Home

The Home page provides an overview of the system and displays important model information such as:

Number of supported classes
Test dataset size
Model accuracy
Deep Learning architecture used
🔐 User Login & Registration

The application provides a simple user authentication interface.

Users can:

Login using an existing account.
Create a new account.
Logout from the application.
👤 User Profile

Users can maintain their basic profile information, including:

Name
Gmail
Phone number
Profile photo
🌿 Disease Detection

The Disease Detection page allows users to upload a plant leaf image.

The uploaded image is:

Resized to 128 × 128 pixels.
Preprocessed using MobileNetV2 preprocessing.
Passed to the trained MobileNetV2 model.
Classified into one of the 38 supported classes.

The application displays:

Uploaded leaf image
Predicted disease
Confidence score
Confidence level
Top-3 predictions
🔝 Top-3 Predictions

Instead of displaying only the highest prediction, the application also shows the three most probable classes.

This provides additional information when multiple classes have similar prediction probabilities.

📋 Prediction History

Every prediction made through the application can be recorded in the prediction history.
The history contains:

| Field            | Description                  |
| ---------------- | ---------------------------- |
| Date             | Date and time of prediction  |
| Username         | User who made the prediction |
| Image            | Uploaded image filename      |
| Prediction       | Predicted disease/class      |
| Confidence       | Prediction probability       |
| Confidence Level | High, Medium, or Low         |


Low
📥 Report Downloads

Prediction history can be downloaded in multiple formats:

CSV

Useful for storing and analyzing prediction records.

Excel

Useful for viewing and organizing prediction history in spreadsheet format.

PDF

Provides a formatted report of prediction records.

📈 Model Evaluation

The application contains a Model Evaluation page where users can view:

Test Accuracy
Weighted Precision
Weighted Recall
Weighted F1-score
Macro Precision
Macro Recall
Macro F1-score
Confusion Matrix

🖥️ Application Workflow

              User Login
                  │
                  ▼
             Home Page
                  │
                  ▼
          Disease Detection
                  │
                  ▼
          Upload Leaf Image
                  │
                  ▼
         Image Preprocessing
                  │
                  ▼
          MobileNetV2 Model
                  │
                  ▼
          Disease Prediction
                  │
          ┌───────┴────────┐
          ▼                ▼
   Confidence Score    Top-3 Results
          │                │
          └───────┬────────┘
                  ▼
          Prediction History
                  │
          ┌───────┼────────┐
          ▼       ▼        ▼
         CSV    Excel     PDF

📁 Project Structure

Plant-Disease-Detection/
│
├── app.py
│
├── models/
│   └── plant_disease_model.keras
│
├── output/
│   ├── class_names.json
│   └── confusion_matrix.png
│
├── data/
│   ├── reports/
│   │   └── PlantVillage dataset reports
│   │
│   ├── splits/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   └── class_names.txt
│   │
│   └── plantvillage_leaf_mapping.csv
│
├── prediction_history.csv
│
├── requirements.txt
│
└── README.md

📌 Important Files

| File / Folder               | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| `app.py`                    | Main Streamlit web application                   |
| `models/`                   | Stores the trained MobileNetV2 model             |
| `plant_disease_model.keras` | Final trained model                              |
| `output/`                   | Stores evaluation outputs and class information  |
| `class_names.json`          | Contains the 38 class names                      |
| `confusion_matrix.png`      | Model confusion matrix                           |
| `data/`                     | Dataset-related files                            |
| `data/splits/`              | Training, validation, and test split information |
| `prediction_history.csv`    | Stores prediction history                        |
| `requirements.txt`          | Python dependencies                              |
| `README.md`                 | Project documentation                            |


🚀 Installation & Usage
1️⃣ Clone the Repository

Clone the project repository using Git:
git clone https://github.com/Sajjadde/Plant-Disease-Detection.git
cd Plant-Disease-Detection

2️⃣ Create a Virtual Environment

It is recommended to use a Python virtual environment.
python -m venv venv

Windows

Activate the virtual environment using:
venv\Scripts\activate

3️⃣ Install Dependencies

Install the required Python packages:
pip install -r requirements.txt

The main technologies used in the project include:

Python 3.11
TensorFlow 2.16.1
Keras 3.1.1
Streamlit 1.29.0
NumPy
Pandas
Pillow
OpenCV
ReportLab
OpenPyXL
4️⃣ Run the Streamlit Application

Start the application using:
streamlit run app.py

After starting the application, Streamlit will provide a local URL.

Open the displayed URL in a web browser to access the Plant Disease Detection System.


.

🧪 Using the Application
Step 1 — Login

Login using an existing account or create a new account.

Step 2 — Open Disease Detection

Navigate to Disease Detection from the sidebar.

Step 3 — Upload an Image

Upload an image of a plant leaf.

Step 4 — Generate Prediction

The trained MobileNetV2 model processes the image and predicts the most likely plant disease or healthy class.

Step 5 — View Results

The application displays:

Predicted class
Confidence score
Confidence level
Top-3 predictions
Step 6 — View History

Previous predictions can be viewed in Prediction History.

Step 7 — Download Reports

Prediction history can be downloaded as:

CSV
Excel
PDF
Step 8 — Evaluate the Model

The Model Evaluation section displays the model performance metrics and confusion matrix.

⚠️ Limitations

The current system has several limitations:

The model achieves 63.98% test accuracy, so incorrect predictions can occur.
Some visually similar diseases are difficult to distinguish.
The model was trained using the PlantVillage dataset, which may not represent all real-world field conditions.
Image quality, lighting, background, leaf orientation, and camera quality may affect prediction performance.
The system should not be considered a substitute for professional agricultural diagnosis.
The current application is designed primarily as a prototype and academic project.

🔮 Future Scope

The system can be improved further through:

Training with larger and more diverse real-world datasets.
Increasing image resolution.
Experimenting with stronger CNN architectures such as EfficientNet.
Using ensemble learning to improve classification performance.
Improving performance on visually similar disease classes.
Adding real-time camera-based disease detection.
Deploying the application on a cloud platform.
Adding multilingual support.
Adding explainable AI techniques such as Grad-CAM.
Developing a mobile application for field-level disease detection.
Integrating additional agricultural information such as disease descriptions and prevention guidelines


🛠️ Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming language    |
| TensorFlow   | Deep Learning framework |
| Keras        | Model development       |
| MobileNetV2  | CNN architecture        |
| NumPy        | Numerical computation   |
| Pandas       | Data processing         |
| Pillow       | Image processing        |
| OpenCV       | Computer vision         |
| Streamlit    | Web application         |
| ReportLab    | PDF report generation   |
| OpenPyXL     | Excel report generation |
| PlantVillage | Dataset                 |


📚 References
PlantVillage Dataset
https://github.com/spMohanty/PlantVillage-Dataset
MobileNetV2
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C.
"MobileNetV2: Inverted Residuals and Linear Bottlenecks."
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
TensorFlow Documentation
https://www.tensorflow.org/
Keras Documentation
https://keras.io/
Streamlit Documentation
https://docs.streamlit.io/


📄 License

This project is released under the MIT License.


👨‍💻 Project Summary

Plant Disease Detection using MobileNetV2 is a Deep Learning-based image classification system designed to identify plant diseases from leaf images.

The project combines:

PlantVillage Dataset + MobileNetV2 + Transfer Learning + Streamlit

to provide an interactive plant disease detection application with prediction confidence, top-3 predictions, prediction history, downloadable reports, and model evaluation.

The final trained model achieved:

63.98% Test Accuracy

across 38 plant disease and healthy classes.
