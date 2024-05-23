import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('/content/drive/MyDrive/Datasets/rice_model.h5')

# Define the label names
label_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy']

# Create a Streamlit app
st.title("Rice Leaf Disease Classification")
st.write("Upload an image of a rice leaf to classify its disease")

# Create a file uploader
uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

# Create a button to classify the image
if st.button("Classify"):
    if uploaded_file is not None:
        # Load the uploaded image
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make a prediction on the image
        prediction = model.predict(img)

        # Get the predicted class label
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Display the predicted label
        st.write(f"Predicted label: {label_names[predicted_label]}")
    else:
        st.write("Please upload an image")
