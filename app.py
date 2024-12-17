import streamlit as st
import pickle
import requests
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image

# Function to load the model from GitHub
def load_model_from_github(model_url, encoder_url):
    model_response = requests.get(model_url)
    encoder_response = requests.get(encoder_url)

    with open('svm_model.pkl', 'wb') as f:
        f.write(model_response.content)
    with open('label_encoder.pkl', 'wb') as f:
        f.write(encoder_response.content)

    with open('svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)

    return model, label_encoder

# Extract HOG features
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (64, 128))
    fd, _ = hog(gray_image, block_norm='L2-Hys', visualize=True)
    return fd

# Function to predict the class based on the uploaded image
def predict_image(image_path, model, label_encoder):
    image = cv2.imread(image_path)
    feature_vector = extract_hog_features(image)
    feature_vector = feature_vector.reshape(1, -1)

    prediction = model.predict(feature_vector)
    predicted_label = label_encoder.inverse_transform(prediction)

    return predicted_label[0]

# Streamlit app layout
st.title("Yawn Detection")

# GitHub URLs for the model and label encoder files
model_url = 'https://github.com/your-username/your-repository/raw/main/svm_model.pkl'
encoder_url = 'https://github.com/your-username/your-repository/raw/main/label_encoder.pkl'

# Load model and label encoder
model, label_encoder = load_model_from_github(model_url, encoder_url)

# File uploader to upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Predict the class of the uploaded image
    predicted_label = predict_image(image_path, model, label_encoder)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {predicted_label}")
