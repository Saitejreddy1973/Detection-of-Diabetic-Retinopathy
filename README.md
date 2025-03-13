**Diabetic Retinopathy Detection Using Deep Learning**

**Overview**

This project is an AI-powered Diabetic Retinopathy (DR) detection system that classifies retinal fundus images into two categories:

✅ No DR (Healthy Retina)

✅ DR (Diabetic Retinopathy Detected)


It uses a Convolutional Neural Network (CNN) built with PyTorch to analyze fundus images and predict DR severity. A Flask/Django-based web application enables users to upload images and receive real-time classification results.

**Features**

🔹 Deep learning-based Retinal Image Classification (DR vs. No DR)

🔹 Preprocessing using OpenCV (CLAHE, resizing, normalization)

🔹 CNN model using PyTorch (Trained on APTOS 2019 Dataset)

🔹 Web Interface (Flask/Django) for real-time DR detection

🔹 Visualization of Model Confidence (Prediction Graph)



**Installation**
1. Clone the Repository

2. Create a Virtual Environment (Recommended)

3. Install Dependencies

   pip install -r requirements.txt

4. Run the Web Application

Then, open your browser and go to:
http://127.0.0.1:8000/


**Model Architecture**

The CNN model consists of:

✔ 4 Convolutional Layers (Feature Extraction)

✔ 2 Fully Connected Layers (Classification)

✔ Activation: ReLU + Softmax

✔ Loss Function: CrossEntropyLoss

✔ Optimizer: Adam

