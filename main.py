import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---- GLASSMORPHISM CSS ----
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
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---- LOAD PYTORCH MODEL ----
@st.cache_resource
def load_model():
    model = torch.load("plant_disease_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# ---- CLASS NAMES ----
raw_class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy'
]

def clean_class_name(name):
    parts = name.split("___")
    if len(parts) == 2:
        return f"{parts[0].replace('_',' ').strip()}: {parts[1].replace('_',' ').strip()}"
    return name.replace("_", " ").strip()

class_names = [clean_class_name(c) for c in raw_class_names]

# ---- PREDICTION FUNCTION ----
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    predicted_idx = torch.argmax(probs).item()
    return predicted_idx, probs.numpy()

# ---- BASIC LEAF CHECK ----
def check_if_leaf(image_path):
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    img_np = np.array(img)
    green_channel = img_np[:, :, 1]  # take green channel
    variance = np.var(green_channel)
    avg_green = np.mean(green_channel)

    # Heuristic: low variance + low green intensity → suspicious
    if variance < 200 and avg_green < 80:
        return False
    return True

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
col1, col2 = st.columns([1, 1.2])

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
        # Warn if image likely not a leaf
        if not check_if_leaf(final_image):
            st.warning("⚠️ This image might not be a plant leaf. Prediction may be inaccurate.")

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
    The model is trained on a **custom dataset** and deployed using **Streamlit**.

    **How it works:**
    - Upload a plant image.
    - Model processes it using a PyTorch model.
    - Displays predicted disease and top 5 predictions.

    This helps farmers & researchers quickly identify plant diseases.
    """)
