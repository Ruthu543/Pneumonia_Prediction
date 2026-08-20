🫁 Pneumonia Prediction from Chest X-Ray Images

An end-to-end Deep Learning image classification project that analyzes chest X-ray images and predicts whether an image belongs to the Normal or Pneumonia class.

The project covers the complete machine learning lifecycle: dataset analysis, image preprocessing, data augmentation, transfer learning, fine-tuning, model evaluation, model saving, and Flask deployment.

📌 Project Overview

Pneumonia is a respiratory infection that can cause visible abnormalities in chest X-ray images. This project explores how Deep Learning and Transfer Learning can be used to build a computer-aided image classification system.

The trained model classifies chest X-ray images into:

🟢 NORMAL

🔴 PNEUMONIA

The final model is integrated into a Flask web application, where users can upload a chest X-ray image and receive a model-generated prediction.

End-to-End Workflow

Chest X-Ray Dataset
        ↓
Exploratory Data Analysis
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
Transfer Learning
        ↓
Model Training
        ↓
Fine-Tuning
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Save Trained Model
        ↓
Flask Web Application
        ↓
Upload X-Ray
        ↓
Prediction

🎯 Project Objectives

🩻 Build an automated chest X-ray image classification system.

🔍 Perform Exploratory Data Analysis and analyze class distribution.

🧹 Apply image preprocessing and normalization.

🔄 Use data augmentation to improve model generalization.

⚖️ Analyze class imbalance and apply class weighting where appropriate.

🧠 Experiment with multiple CNN and Transfer Learning architectures.

🔧 Fine-tune selected pretrained layers.

📊 Evaluate models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

🏆 Compare different architectures and select a suitable model for deployment.

🌐 Integrate the trained model into a Flask web application.

📤 Allow users to upload chest X-ray images through a web interface.

⚡ Generate Normal/Pneumonia predictions.

📦 Manage large trained model files using Git LFS.

🧠 Models Experimented With

The project explores multiple pretrained Convolutional Neural Network architectures:

Model

Approach

VGG19

Transfer Learning + Fine-Tuning

ResNet50

Transfer Learning

MobileNetV2

Transfer Learning

EfficientNetB0

Transfer Learning

DenseNet121

Transfer Learning

The models are compared based on their ability to distinguish between Normal and Pneumonia chest X-ray images.

🔧 Transfer Learning Strategy

Instead of training a Deep Learning model completely from scratch, pretrained CNN architectures are used as the starting point.

Stage 1 — Feature Extraction

The pretrained convolutional layers are initially frozen while custom classification layers are trained.

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

Data augmentation is applied to training images to improve generalization and reduce overfitting.

Techniques used/explored include:

Horizontal Flip

Rotation

Width Shift

Height Shift

Shearing

Zooming

             Original X-Ray
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
   Rotation    Flip/Shift    Shearing
       │           │           │
       └───────────┼───────────┘
                   ↓
            Augmented Images

Data augmentation increases training-data diversity without requiring additional real-world images.

⚖️ Class Imbalance Handling

Class imbalance can cause a model to favor the majority class.

The project analyzes the distribution of Normal and Pneumonia images and can use class weighting during model training.

Example:

class_weights = {
    0: 1.05,
    1: 0.95
}

The exact weights depend on the class distribution of the training dataset.

📊 Model Evaluation

The models are evaluated using multiple classification metrics.

Metric

Purpose

Accuracy

Overall percentage of correct predictions

Precision

Proportion of predicted Pneumonia cases that are actually Pneumonia

Recall

Proportion of actual Pneumonia cases correctly identified

F1-Score

Harmonic mean of Precision and Recall

Confusion Matrix

Detailed breakdown of classification errors

🚨 Why Recall Matters

For a pneumonia-screening use case, recall is particularly important because it measures how many actual Pneumonia cases are correctly identified.

A false negative occurs when:

Actual:    PNEUMONIA
Predicted: NORMAL

Therefore, model selection should not rely on accuracy alone. Recall, Precision, F1-Score, and the Confusion Matrix should also be considered.

📈 Model Performance

Add the final test-set results from your experiments to the table below.

Model

Accuracy

Precision

Recall

F1-Score

VGG19

—

—

—

—

ResNet50

—

—

—

—

MobileNetV2

—

—

—

—

EfficientNetB0

—

—

—

—

DenseNet121

—

—

—

—

Final Selected Model

—

—

—

—

Note: Replace the dashes with your actual test-set results. Do not publish placeholder percentages such as XX%.

🌐 Flask Web Application

The trained model is integrated into a Flask-based web application.

Prediction Workflow

User
  ↓
Upload Chest X-Ray
  ↓
Flask Backend
  ↓
Image Validation
  ↓
Resize & Normalize
  ↓
Trained Deep Learning Model
  ↓
Prediction Probability
  ↓
Classification
  ↓
NORMAL / PNEUMONIA

Application Steps

1. Open the application

Launch the Flask application in your browser.

2. Upload an X-ray

Select a chest X-ray image from your computer.

3. Submit the image

The image is sent to the Flask backend.

4. Preprocess the image

The application applies the preprocessing used during model development.

5. Generate prediction

The processed image is passed to the trained Deep Learning model.

6. Display result

The application displays the predicted class.

Example:

Prediction: PNEUMONIA

or

Prediction: NORMAL

🛠️ Technology Stack

Category

Technologies

Programming Language

Python

Deep Learning

TensorFlow, Keras

CNN Architectures

VGG19, ResNet50, MobileNetV2, EfficientNetB0, DenseNet121

Image Processing

OpenCV, Pillow

Data Processing

NumPy, Pandas

Model Evaluation

Scikit-learn

Visualization

Matplotlib, Seaborn

Web Framework

Flask

Frontend

HTML, CSS, JavaScript

Development

Jupyter Notebook, VS Code

Version Control

Git, GitHub

Large Model Files

Git LFS

📂 Project Structure

Pneumonia_Prediction/
│
├── app.py
│
├── pneumonia_classifier.ipynb
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
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md

Large .h5 files are managed using Git LFS.

⚙️ Installation & Setup

1. Clone the Repository

git clone https://github.com/Ruthu543/Pneumonia_Prediction.git

2. Navigate to the Project

cd Pneumonia_Prediction

3. Create a Virtual Environment

Windows

python -m venv venv

Activate it:

venv\Scripts\activate

macOS / Linux

python3 -m venv venv

Activate it:

source venv/bin/activate

4. Install Dependencies

pip install -r requirements.txt

📦 Git LFS Setup

The trained .h5 model files can be large, so this project uses Git Large File Storage (Git LFS).

Install and initialize Git LFS:

git lfs install

Pull the tracked model files:

git lfs pull

To verify tracked LFS files:

git lfs ls-files

▶️ Run the Application

Start the Flask application:

python app.py

The application will typically be available at:

http://127.0.0.1:5000

Open the address in your browser and upload a chest X-ray image.

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

📁 Dataset

The project works with chest X-ray images categorized into:

NORMAL
PNEUMONIA

The dataset is divided into training, validation, and testing sets.

Expected Dataset Structure

dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

The dataset itself is not included in this repository.

🚀 Future Improvements

Improve model performance through additional fine-tuning.

Compare advanced architectures and hyperparameters.

Improve handling of class imbalance.

Add Grad-CAM for model explainability.

Display prediction confidence.

Add Docker support.

Deploy the Flask application to a cloud platform.

Add automated model evaluation.

Improve the web application's UI/UX.

Extend the system to support additional chest X-ray conditions.

💡 Key Learnings

This project provided practical experience in:

Deep Learning

CNN-based image classification

Transfer Learning

Fine-Tuning

Medical image preprocessing

Data augmentation

Class imbalance handling

TensorFlow and Keras

Model evaluation

Flask application development

Git and GitHub

Git LFS

End-to-end ML project development

🔮 Project Highlights

✔ End-to-End Deep Learning Pipeline
✔ Transfer Learning
✔ Multiple CNN Architectures
✔ Fine-Tuning
✔ Data Augmentation
✔ Class Imbalance Analysis
✔ Multiple Evaluation Metrics
✔ Flask Web Application
✔ Git LFS for Large Models
✔ Real-Time Image Prediction

👨‍💻 Author

Ruthu Madhavi Kola

Python Developer | Data Analyst | Machine Learning Enthusiast

GitHub:
https://github.com/Ruthu543

⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

📄 Disclaimer

This project is developed for educational and research purposes. Predictions generated by the model should not be considered medical advice, diagnosis, or treatment recommendations. Always consult a qualified healthcare professional for medical evaluation.
