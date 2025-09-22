import streamlit as st
import tensorflow as tf
import numpy as np
import re
from pathlib import Path

# ---- LOAD TFLITE MODEL ----
@st.cache_resource
def load_tflite_interpreter(model_path="trained_plant_disease_model_plantvillage.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_interpreter()

def model_prediction(test_image):
    # Load first keras model
    model1 = tf.keras.models.load_model("trained_plant_disease_model.keras")

    # Preprocess input
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr], dtype=np.float32)

    # Prediction from model1 (Keras)
    prediction1 = model1.predict(input_arr)

    # Prediction from model2 (TFLite)
    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    prediction2 = interpreter.get_tensor(output_details[0]['index'])

    # Combine predictions
    combined_predictions = prediction1 + prediction2
    return np.argmax(combined_predictions)

def clean_class_name(name):
    parts = name.split("___")
    if len(parts) == 2:
        main = parts[0].replace("_", " ").strip()
        sub = parts[1].replace("_", " ").strip()
        return f"{main}: {sub}"
    return name.replace("_", " ").strip()

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
    'Tomato___healthy','Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]
cleaned_class_names = [clean_class_name(name) for name in raw_class_names]

# ---- SIDEBAR ----
st.sidebar.title("🌱 Supported Plants")
plant_groups = {}
for name in cleaned_class_names:
    plant = name.split(":")[0] if ":" in name else name
    plant_groups.setdefault(plant, []).append(name)

for plant, diseases in plant_groups.items():
    with st.sidebar.expander(plant):
        for disease in diseases:
            st.sidebar.write(f"- {disease}")

st.sidebar.markdown("---")
st.sidebar.subheader("🖼 Try Sample Images")
sample_images_path = Path("sample_images")
if sample_images_path.exists():
    sample_files = list(sample_images_path.glob("*.jpg")) + list(sample_images_path.glob("*.png"))
    selected_sample = st.sidebar.selectbox(
        "Choose a sample image:",
        ["None"] + [file.name for file in sample_files]
    )
    if selected_sample != "None":
        test_image = sample_images_path / selected_sample
        st.sidebar.image(str(test_image), caption="Sample Image", use_column_width=True)
else:
    st.sidebar.info("No sample images found. Add images to `sample_images/` folder.")

# ---- MAIN PAGE ----
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

st.subheader("🔍 Disease Recognition")
col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

final_image = uploaded_image if uploaded_image else (test_image if 'test_image' in locals() else None)

with col2:
    if final_image and st.button("Predict"):
        st.snow()
        result_index = model_prediction(final_image)
        disease_name = cleaned_class_names[result_index]
        st.success(f"🌱 Predicted Disease: **{disease_name}**")
    elif not final_image:
        st.info("Please upload or select an image to predict.")
