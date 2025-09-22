import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd

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

# ---- LOAD TFLITE MODELS ----
@st.cache_resource
def load_tflite_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter1, input_details1, output_details1 = load_tflite_interpreter("trained_plant_disease_model_plantvillage.tflite")
interpreter2, input_details2, output_details2 = load_tflite_interpreter("trained_plant_disease_model.tflite")

# ---- LOAD KERAS MODEL ----
@st.cache_resource
def load_keras_model(model_path="trained_plant_disease_model.keras"):
    return tf.keras.models.load_model(model_path)

keras_model = load_keras_model()

# ---- CLASS NAMES ----
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

def clean_class_name(name):
    parts = name.split("___")
    if len(parts) == 2:
        return f"{parts[0].replace('_',' ').strip()}: {parts[1].replace('_',' ').strip()}"
    return name.replace("_", " ").strip()

cleaned_class_names = [clean_class_name(c) for c in raw_class_names]

# ---- PREDICTION FUNCTION ----
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr], dtype=np.float32)
    
    # Keras model prediction
    pred1 = keras_model.predict(input_arr)
    
    # TFLite model predictions
    interpreter1.set_tensor(input_details1[0]['index'], input_arr)
    interpreter1.invoke()
    pred2_1 = interpreter1.get_tensor(output_details1[0]['index'])
    
    interpreter2.set_tensor(input_details2[0]['index'], input_arr)
    interpreter2.invoke()
    pred2_2 = interpreter2.get_tensor(output_details2[0]['index'])
    
    # Ensure same shape
    min_len = min(pred1.shape[1], pred2_1.shape[1], pred2_2.shape[1])
    combined = pred1[0, :min_len] + pred2_1[0, :min_len] + pred2_2[0, :min_len]
    
    return np.argmax(combined), combined

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

# ---- MAIN FUNCTIONALITY ----
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.subheader("🔍 Upload Image & Predict Disease")
col1, col2 = st.columns([1,1.2])

with col1:
    uploaded_image = st.file_uploader("Upload plant leaf image:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_column_width=True)

# Sample images
sample_images_path = Path(__file__).parent / "sample_images"
sample_files = []
if sample_images_path.exists():
    sample_files = list(sample_images_path.glob("*.jpg")) + list(sample_images_path.glob("*.jpeg")) + \
                   list(sample_images_path.glob("*.png")) + list(sample_images_path.glob("*.JPG")) + \
                   list(sample_images_path.glob("*.JPEG")) + list(sample_images_path.glob("*.PNG"))

if sample_files:
    selected_sample = st.selectbox("Or choose a sample image:", ["None"] + [f.name for f in sample_files])
    if selected_sample != "None":
        test_image = sample_images_path / selected_sample
        st.image(str(test_image), caption="Sample Image", use_column_width=True)

final_image = uploaded_image if uploaded_image else (test_image if 'test_image' in locals() else None)

with col2:
    if final_image and st.button("Predict", use_container_width=True):
        st.snow()
        result_index, probabilities = model_prediction(final_image)
        disease_name = cleaned_class_names[result_index]
        st.success(f"🌱 Predicted Disease: **{disease_name}**")
        
        # Display probabilities as a horizontal bar chart
        df = pd.DataFrame({
            "Disease": cleaned_class_names[:len(probabilities)],
            "Probability": probabilities
        })
        st.subheader("Prediction Probabilities")
        st.bar_chart(df.set_index("Disease"))

    elif not final_image:
        st.info("Upload or select an image to predict.")

st.markdown("</div>", unsafe_allow_html=True)

# ---- PROJECT INFO ----
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    This project uses **Deep Learning (CNNs)** to classify plant diseases from leaf images.
    The model is trained on the **PlantVillage dataset** and deployed using **Streamlit**.
    
    **How it works:**
    - Upload a plant image.
    - Model processes it using TensorFlow + TFLite hybrid pipeline.
    - Displays predicted disease name with confidence.
    
    This helps farmers & researchers quickly identify plant diseases for better crop management.
    """)
