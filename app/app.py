import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('../models/mobilenet.h5')

class_names = ['Bass', 'Salmon', 'Tuna', ...]  # Update based on dataset

def predict(img):
    img = img.resize((224,224))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return class_names[np.argmax(preds)], preds

st.title("Fish Classifier App")
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')
    label, conf = predict(img)
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {np.max(conf)*100:.2f}%")
