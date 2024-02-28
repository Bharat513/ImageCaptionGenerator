import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import json
from keras.preprocessing.text import Tokenizer
# Load pre-trained model and tokenizer
model_path = "model.h5"
tokenizer_path = "tokenizer.pkl"

model = load_model(model_path)
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    loaded_tokenizer_config = json.load(f)
    tokenizer = Tokenizer.from_config(loaded_tokenizer_config)

import pickle

# Saving the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Loading the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
# Function to preprocess the input image
def preprocess_image(image):
    img = load_img(image, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to generate caption
def generate_caption(image_path, model, tokenizer):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    caption = decode_caption(prediction, tokenizer)
    return caption

# Function to decode the predicted caption
def decode_caption(prediction, tokenizer):
    predicted_ids = np.argmax(prediction, axis=2)
    predicted_caption = tokenizer.sequences_to_texts(predicted_ids)[0]
    return predicted_caption

# Streamlit app
st.title("Image to Caption Generator")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to generate caption
    if st.button("Generate Caption"):
        # Generate and display the caption
        try:
            caption = generate_caption(uploaded_file, model, tokenizer)
            st.success(f"Generated Caption: {caption}")
        except Exception as e:
            st.error(f"Error: {e}")