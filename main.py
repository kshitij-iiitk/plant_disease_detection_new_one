import streamlit as st
import tensorflow as tf
import numpy as np
import re

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # batch dimension
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Clean class name
def clean_class_name(name):
    parts = name.split("___")
    if len(parts) == 2:
        main = parts[0].replace("_", " ").strip()
        sub = parts[1].replace("_", " ").strip()
        return f"{main}: {sub}"
    return name.replace("_", " ").strip()

# Title and Intro
st.title("🌿 Plant Disease Recognition System")
st.image("home_page.jpeg", use_column_width=True)
st.markdown("""
Upload a plant image, and the system will detect possible diseases using a trained deep learning model.

---

### How It Works
1. **Upload Image** below.
2. **Model Processes** the image.
3. **Result** is displayed with the disease name.

---
""")

# About Dataset
st.subheader("📊 About Dataset")
st.markdown("""
- ~87K RGB images of healthy and diseased crop leaves (38 classes).  
- Split into **Train: 80%**, **Validation: 20%**, plus 33 test images.  
- Images collected and augmented from open datasets.  

---
""")

# Prediction Section
st.subheader("🔍 Disease Recognition")

col1, col2 = st.columns(2)

with col1:
    test_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

with col2:
    if test_image and st.button("Predict"):
        st.snow()
        result_index = model_prediction(test_image)

        raw_class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        cleaned_class_names = [clean_class_name(name) for name in raw_class_names]
        disease_name = cleaned_class_names[result_index]
        st.success(f"🌱 Predicted Disease: **{disease_name}**")
    elif not test_image:
        st.info("Please upload an image to predict.")
