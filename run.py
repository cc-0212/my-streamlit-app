import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# --- CLASS LABELS ---
CLASSES = [
    "Broccoli Stage 1", "Broccoli Stage 2", "Broccoli Stage 3",
    "Cilantro Stage 1", "Cilantro Stage 2", "Cilantro Stage 3",
    "Purple Kale Stage 1", "Purple Kale Stage 2", "Purple Kale Stage 3",
    "WheatGrass Stage 1", "WheatGrass Stage 2", "WheatGrass Stage 3"
]

IMG_SIZE = (160, 160)  # Modify this to match your model input

# --- LOAD MODEL ---
@st.cache_resource
def load_model_from_file(model_path='microgreens_model.h5'):
    return load_model(model_path)

model = load_model_from_file()

# --- STREAMLIT UI ---
st.title("📷 Real-Time Microgreen Classifier")
st.markdown("Capture an image using your webcam to classify the microgreen type and its growth stage.")

# Capture webcam input
camera_image = st.camera_input("Take a photo using your webcam")

if camera_image:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASSES[predicted_index]

    # Parse and display
    parts = predicted_label.split(" Stage ")
    microgreen_type = parts[0]
    growth_stage = f"Stage {parts[1]}"

    st.subheader("🔍 Prediction Result")
    st.write(f"**Microgreen Type:** {microgreen_type}")
    st.write(f"**Growth Stage:** {growth_stage}")
