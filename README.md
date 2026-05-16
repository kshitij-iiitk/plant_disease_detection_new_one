# Plant Disease Detection using Deep Learning

## Overview

This project is a web-based plant disease detection system built using:

- TensorFlow / Keras
- Streamlit
- CNN-based deep learning models

The application predicts diseases from plant leaf images and displays the most probable disease classes along with confidence scores.

The models are trained using the PlantVillage dataset and combined for improved prediction performance.

Live demo available at: https://plant-disease-detection-krp.streamlit.app

---

## Features

- Upload plant leaf images
- Predict plant diseases using deep learning
- Ensemble prediction using two trained models
- Display top 5 predictions
- Interactive Streamlit UI
- Sidebar showing supported plants and diseases
- Sample image testing support

---

## Supported Plants

The project supports multiple crops including:

- Apple
- Blueberry
- Cherry
- Corn (Maize)
- Grape
- Orange
- Peach
- Pepper Bell
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

Along with multiple disease categories and healthy leaf detection.

---

## Tech Stack

- Python
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Librosa
- PyTorch

---

## Project Structure

```text
plant_disease_detection_new_one/
│
├── main.py
├── requirements.txt
├── runtime.txt
├── Train_plant_disease_2.ipynb
├── Test_plant_disease.ipynb
├── trained_plant_disease_model.keras
├── trained_plant_disease_model_plantvillage.keras
├── sample_images/
├── screenshots/
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kshitij-iiitk/plant_disease_detection_new_one.git
cd plant_disease_detection_new_one
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Recommended Requirements

```txt
tensorflow==2.9.3
scikit-learn==1.1.3
numpy==1.23.5
matplotlib==3.6.3
seaborn==0.12.2
pandas==1.5.3
streamlit==1.32.2
librosa==0.10.1
torch==2.2.2
```

Create a `runtime.txt` file:

```txt
python-3.9
```

---

## Running the Application

Start the Streamlit app:

```bash
streamlit run main.py
```

---

## Model Details

The application loads two Keras models:

- `trained_plant_disease_model.keras`
- `trained_plant_disease_model_plantvillage.keras`

Predictions from both models are combined to improve classification accuracy.

---

## Screenshots

### Home Page

![Home Page](screenshots\Homepage.png)

---

### Disease Prediction

![Prediction](screenshots\Prediction.png)

---

### Confusion Matrix


![Confusion Matrix](screenshots/confusion_matrix.png)


---

## Dataset

Dataset used:

- PlantVillage Dataset

The dataset contains healthy and diseased plant leaf images across multiple crop categories.

---

## Future Improvements

- Add more plant species
- Improve model accuracy
- Mobile-friendly UI
- Real-time camera detection
- Disease treatment recommendations
- Multi-language support

---

## Deployment

This project can be deployed on:

- Streamlit Community Cloud
- Render
- Hugging Face Spaces

---

## Repository

Repository Link:

https://github.com/kshitij-iiitk/plant_disease_detection_new_one

---

## Author

Developed by Kshitij.

