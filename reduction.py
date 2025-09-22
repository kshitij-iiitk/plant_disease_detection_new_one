import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("trained_plant_disease_model_plantvillage.keras")

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enables weight quantization
tflite_model = converter.convert()

# Save compressed model
with open("trained_plant_disease_model_plantvillage.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite and saved.")
