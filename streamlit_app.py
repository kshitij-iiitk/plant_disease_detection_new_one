import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---- GLASSMORPHISM CSS ----
st.markdown("""
<style>
body { background: linear-gradient(to right, #e8f5e9, #f1f8e9); }
.reportview-container .main .block-container{ padding-top: 2rem; }
.glass-box {
    background: rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---- LOAD KERAS MODELS ----
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("trained_plant_disease_model_plantvillage.keras")
    model2 = tf.keras.models.load_model("trained_plant_disease_model.keras")
    return model1, model2

model1, model2 = load_models()

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

class_names = [clean_class_name(c) for c in raw_class_names]

# ---- PREDICTION FUNCTION ----
def predict_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    arr = arr.astype(np.float32)

    pred1 = model1.predict(arr)[0]
    pred2 = model2.predict(arr)[0]

    # Trim to smallest shape to combine
    min_len = min(len(pred1), len(pred2))
    combined = pred1[:min_len] + pred2[:min_len]

    predicted_idx = np.argmax(combined)
    return predicted_idx, combined

# ---- SIDEBAR ----
st.sidebar.title("🌱 Supported Plants")
plants = {}
for name in class_names:
    plant = name.split(":")[0] if ":" in name else name
    plants.setdefault(plant, []).append(name)

for plant, diseases in plants.items():
    with st.sidebar.expander(plant):
        for disease in diseases:
            st.sidebar.write(f"- {disease}")

# ---- MAIN UI ----
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.subheader("🔍 Upload Image & Predict Disease")
col1, col2 = st.columns([1,1.2])

with col1:
    uploaded_image = st.file_uploader("Upload plant leaf image:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_column_width=True)

# Sample images
sample_images_path = Path("sample_images")
sample_files = []
if sample_images_path.exists():
    sample_files = list(sample_images_path.glob("*.[jJ][pP][gG]")) + list(sample_images_path.glob("*.[pP][nN][gG]"))

if sample_files:
    selected_sample = st.selectbox("Or choose a sample image:", ["None"] + [f.name for f in sample_files])
    if selected_sample != "None":
        test_image = sample_images_path / selected_sample
        st.image(str(test_image), caption="Sample Image", use_column_width=True)

final_image = uploaded_image if uploaded_image else (test_image if 'test_image' in locals() else None)

with col2:
    if final_image and st.button("Predict", use_container_width=True):
        st.snow()
        idx, probs = predict_image(final_image)
        st.success(f"🌱 Predicted Disease: **{class_names[idx]}**")

        # Top 5 probabilities
        top_idx = np.argsort(probs)[-5:][::-1]
        top_classes = [class_names[i] for i in top_idx]
        top_probs = [probs[i] for i in top_idx]
        df = pd.DataFrame({"Disease": top_classes, "Probability": top_probs})
        st.subheader("Top 5 Predictions")
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
- Model processes it using two Keras models.
- Displays predicted disease and top 5 predictions.

This helps farmers & researchers quickly identify plant diseases.
""")
