🫁 Pneumonia Prediction from Chest X-Ray Images

An end-to-end Deep Learning image classification project that analyzes chest X-ray images and predicts whether an image belongs to the Normal or Pneumonia class. The project covers the complete machine learning lifecycle—from data preprocessing and exploratory analysis to transfer learning, model evaluation, and Flask-based deployment.

📌 Project Overview

Pneumonia is a respiratory infection that can cause visible abnormalities in chest X-ray images. Manual examination of large volumes of medical images can be time-consuming, motivating the development of computer-aided image classification systems.

This project uses Deep Learning and Transfer Learning to automatically classify chest X-ray images into:

🟢 NORMAL
🔴 PNEUMONIA

The project follows an end-to-end workflow involving:

Dataset Analysis → Image Preprocessing → Data Augmentation → Transfer Learning → Fine-Tuning → Model Evaluation → Model Saving → Flask Deployment → Prediction

The trained model is integrated into a Flask web application, allowing users to upload a chest X-ray image and receive a model-generated prediction.

🎯 Project Objectives

The primary objectives of this project are:

🩻 Develop an automated chest X-ray image classification system.
🔍 Perform Exploratory Data Analysis (EDA) to understand the dataset and class distribution.
🧹 Apply image preprocessing and normalization techniques.
🔄 Use data augmentation to improve model generalization.
⚖️ Analyze and address class imbalance using appropriate techniques such as class weighting.
🧠 Experiment with multiple Convolutional Neural Network (CNN) and Transfer Learning architectures.
🔧 Apply fine-tuning to pretrained models.
📊 Evaluate models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
🏆 Compare multiple architectures and select a suitable model for deployment.
🌐 Integrate the trained model into a Flask web application.
📤 Enable users to upload X-ray images through a web interface.
⚡ Generate real-time Normal/Pneumonia predictions.
📦 Manage large trained model files using Git LFS.
🔄 Deep Learning Workflow
                 ┌──────────────────────────┐
                 │   Chest X-Ray Dataset     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Exploratory Data Analysis│
                 │   & Class Distribution   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Image Preprocessing     │
                 │ Resize + Normalize        │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Data Augmentation      │
                 │ Flip / Rotation / Shift   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Train / Validation / Test│
                 │       Dataset Split      │
                 └────────────┬─────────────┘
                              │
                              ▼
            ┌────────────────────────────────────┐
            │     Transfer Learning Models       │
            │                                    │
            │ VGG19 | ResNet50 | MobileNetV2    │
            │ EfficientNetB0 | DenseNet121       │
            └────────────────┬───────────────────┘
                             │
                             ▼
                 ┌──────────────────────────┐
                 │    Model Training        │
                 │    & Fine-Tuning         │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Model Evaluation      │
                 │ Accuracy / Precision     │
                 │ Recall / F1 / Confusion  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Best Model Selection │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Save Trained Model   │
                 │          .h5             │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Flask Web Application  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Upload X-Ray Image   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Image Preprocessing   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Trained CNN Prediction │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   NORMAL / PNEUMONIA    │
                 └──────────────────────────┘
🧠 Model Architecture & Transfer Learning
Transfer Learning

Instead of training a Deep Learning model completely from scratch, this project leverages pretrained CNN architectures.

Pretrained networks have already learned useful visual features from large-scale image datasets. These learned features can then be adapted to the chest X-ray classification problem.

🏗️ Models Experimented With

The project explores multiple architectures, including:

VGG19
ResNet50
MobileNetV2
EfficientNetB0
DenseNet121

The architectures are evaluated based on their ability to distinguish between Normal and Pneumonia X-ray images.

🔧 Fine-Tuning Strategy

The transfer learning process follows two major stages:

Stage 1 — Feature Extraction

The pretrained convolutional layers are initially frozen.

Pretrained CNN
      ↓
Frozen Convolutional Layers
      ↓
Feature Extraction
      ↓
Custom Classification Layers
      ↓
NORMAL / PNEUMONIA
Stage 2 — Fine-Tuning

Selected pretrained layers are unfrozen and trained with a lower learning rate.

Pretrained CNN
      ↓
Selected Layers Unfrozen
      ↓
Fine-Tuning
      ↓
Custom Classification Head
      ↓
NORMAL / PNEUMONIA

Fine-tuning allows the model to learn features that are more specific to chest X-ray images.

🔬 Data Preprocessing

Before training, the X-ray images undergo preprocessing to ensure consistency.

Preprocessing Steps
Load image.
Resize to the required input dimensions.
Normalize pixel values.
Convert images into model-compatible tensors.
Create batches for training.
Assign class labels.

Typical pipeline:

Raw X-Ray
    ↓
Image Loading
    ↓
Resize
    ↓
Normalization
    ↓
Tensor Conversion
    ↓
Model Input
🔄 Data Augmentation

To improve model generalization and reduce overfitting, augmentation techniques are applied to training images.

Possible transformations include:

Horizontal Flip
Rotation
Width Shift
Height Shift
Shearing
Zooming
Original X-Ray
      │
      ├── Rotation
      ├── Horizontal Flip
      ├── Width Shift
      ├── Height Shift
      └── Shearing
             │
             ▼
      Augmented Images

Data augmentation increases the diversity of training samples without requiring additional real-world images.

⚖️ Class Imbalance Handling

Class imbalance can cause a model to favor the majority class.

The project analyzes the distribution of Normal and Pneumonia images and can use techniques such as class weighting during model training.

Example:

class_weights = {
    0: 1.05,
    1: 0.95
}

The exact weights depend on the class distribution of the training dataset.

📊 Model Evaluation

The models are evaluated using multiple classification metrics.

Metrics Used
Metric	Purpose
Accuracy	Overall percentage of correct predictions
Precision	Proportion of predicted pneumonia cases that are actually pneumonia
Recall	Proportion of actual pneumonia cases correctly identified
F1-Score	Harmonic mean of Precision and Recall
Confusion Matrix	Detailed breakdown of classification errors
🚨 Why Recall Is Important

For this medical image classification use case, Recall is particularly important.

Recall answers:

"Of all patients/images that actually contain pneumonia, how many did the model correctly identify?"

A false negative occurs when:

Actual:     PNEUMONIA
Predicted:  NORMAL

Reducing false negatives is important in a pneumonia-screening context.

Therefore, model selection should not rely solely on accuracy. Recall, precision, F1-score, and the confusion matrix should also be considered.

📈 Model Performance

Enter the final experimental results in the table below:

Model	Accuracy	Precision	Recall	F1-Score
VGG19	XX%	XX%	XX%	XX%
ResNet50	XX%	XX%	XX%	XX%
MobileNetV2	XX%	XX%	XX%	XX%
EfficientNetB0	XX%	XX%	XX%	XX%
DenseNet121	XX%	XX%	XX%	XX%
Final Selected Model	XX%	XX%	XX%	XX%

Note: Replace the placeholder values with your actual test-set results.

🛠️ Technology Stack

Category	Technologies
Programming Language	Python
Deep Learning	TensorFlow, Keras
CNN Architectures	VGG19, ResNet50, MobileNetV2, EfficientNetB0, DenseNet121
Image Processing	OpenCV, Pillow
Data Processing	NumPy, Pandas
Model Evaluation	Scikit-Learn
Visualization	Matplotlib, Seaborn
Web Framework	Flask
Frontend	HTML, CSS, JavaScript
Development	Jupyter Notebook, VS Code
Version Control	Git, GitHub
Large Model Files	Git LFS

📂 Project Structure
Pneumonia-Prediction/
│
├── app.py
│
├── model.h5
│
├── model_weights/
│   ├── vgg19_model_01.h5
│   ├── vgg19_model_02.h5
│   └── vgg_unfrozen.h5
│
├── static/
│   └── uploads/
│       ├── NORMAL2-IM-0338-0001.jpeg
│       └── person1660_virus_2869.jpeg
│
├── templates/
│   └── ...
│
├── pneumonia_classifier.ipynb
│
├── requirements.txt
│
├── .gitignore
│
├── .gitattributes
│
└── README.md
⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/Ruthu543/Pneumonia_Prediction.git

Navigate into the project:

cd Pneumonia_Prediction
2. Create a Virtual Environment
Windows
python -m venv venv

Activate the environment:

venv\Scripts\activate
macOS / Linux
python3 -m venv venv

Activate:

source venv/bin/activate
3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt
4. Git LFS Setup

Because the trained .h5 model files are large, this project uses Git LFS.

Install Git LFS if it is not already installed:

git lfs install

Pull the large model files:

git lfs pull
▶️ Running the Application

Start the Flask application:

python app.py

The application will typically be available at:

http://127.0.0.1:5000

Open the URL in your browser.

🌐 Application Usage

The web application provides a simple workflow for performing predictions.

Step 1 — Open the Application

Launch the Flask application in your browser.

Step 2 — Upload an X-Ray

Select a chest X-ray image from your computer.

Step 3 — Submit the Image

The image is sent to the Flask backend.

Step 4 — Image Preprocessing

The application performs the same preprocessing used during model training.

Step 5 — Model Prediction

The processed image is passed to the trained Deep Learning model.

Step 6 — Display Result

The application displays the predicted class.

Example:

Prediction: PNEUMONIA

or:

Prediction: NORMAL
🔄 Prediction Pipeline
User
  │
  ▼
Upload X-Ray
  │
  ▼
Flask Backend
  │
  ▼
Image Validation
  │
  ▼
Resize & Normalize
  │
  ▼
Trained Deep Learning Model
  │
  ▼
Prediction Probability
  │
  ▼
Classification
  │
  ├───────────────┐
  ▼               ▼
NORMAL       PNEUMONIA
🧪 Training Workflow

The training notebook contains the model development process:

1. Import Libraries
        ↓
2. Load Dataset
        ↓
3. Explore Dataset
        ↓
4. Analyze Class Distribution
        ↓
5. Visualize Sample Images
        ↓
6. Preprocess Images
        ↓
7. Apply Data Augmentation
        ↓
8. Prepare Data Generators
        ↓
9. Build Transfer Learning Model
        ↓
10. Train Classification Head
        ↓
11. Fine-Tune Selected Layers
        ↓
12. Evaluate Model
        ↓
13. Generate Confusion Matrix
        ↓
14. Compare Models
        ↓
15. Save Best Model
