import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---- CUSTOM CSS FOR GLASSMORPHISM ----
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e8f5e9, #f1f8e9);
}
.reportview-container .main .block-container{
    padding-top: 2rem;
}
.glass-box {
    background: rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}
</style>
""", unsafe_allow_html=True)

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
    model1 = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr], dtype=np.float32)
    prediction1 = model1.predict(input_arr)
    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    prediction2 = interpreter.get_tensor(output_details[0]['index'])
    combined_predictions = prediction1 + prediction2
    return np.argmax(combined_predictions)

def clean_class_name(name):
    parts = name.split("___")
    if len(parts) == 2:
        return f"{parts[0].replace('_',' ').strip()}: {parts[1].replace('_',' ').strip()}"
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

# ---- MAIN FUNCTIONALITY (TOP) ----
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.subheader("🔍 Upload Image & Predict Disease")
col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Upload plant leaf image:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_column_width=True)

# Sample images from folder
sample_images_path = Path("sample_images")
if sample_images_path.exists():
    sample_files = list(sample_images_path.glob("*.jpg")) + list(sample_images_path.glob("*.png"))
    selected_sample = st.selectbox("Or choose a sample image:", ["None"] + [file.name for file in sample_files])
    if selected_sample != "None":
        test_image = sample_images_path / selected_sample
        st.image(str(test_image), caption="Sample Image", use_column_width=True)
else:
    st.info("No sample images found. Add some to `sample_images/` folder.")

final_image = uploaded_image if uploaded_image else (test_image if 'test_image' in locals() else None)

with col2:
    if final_image and st.button("Predict", use_container_width=True):
        st.snow()
        result_index = model_prediction(final_image)
        disease_name = cleaned_class_names[result_index]
        st.success(f"🌱 Predicted Disease: **{disease_name}**")
    elif not final_image:
        st.info("Upload or select an image to predict.")

st.markdown("</div>", unsafe_allow_html=True)

# ---- PROJECT INFO (BOTTOM) ----
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    This project uses **Deep Learning (CNNs)** to classify plant diseases from leaf images.
    The model is trained on the **PlantVillage dataset** and deployed using **Streamlit**.
    
    **How it works:**
    - Upload a plant image.
    - Model processes it using a TensorFlow + TFLite hybrid pipeline.
    - Displays predicted disease name with confidence.
    
    This helps farmers & researchers quickly identify plant diseases for better crop management.
    """)
