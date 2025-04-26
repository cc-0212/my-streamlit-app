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

IMG_SIZE = (160, 160)  # Adjust to match your model's input shape

# --- LOAD MODEL ---
@st.cache_resource
def load_model_from_file(model_path='microgreens_model.h5'):
    return load_model(model_path)

model = load_model_from_file()

# --- STREAMLIT UI ---
st.title("🌱 Microgreen Classifier")
st.markdown("Choose how you want to upload the microgreen image:")

# --- Choose input method ---
input_method = st.radio("Select input method:", ("Upload Image", "Use Webcam"))

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif input_method == "Use Webcam":
    camera_image = st.camera_input("Take a photo using your webcam")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

# --- Perform Prediction ---
if image is not None:
    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASSES[predicted_index]

    # Parse
    microgreen_type, stage_number = predicted_label.split(" Stage ")
    growth_stage = f"Stage {stage_number}"

    # Display result
    st.subheader("🔍 Prediction Result")
    st.write(f"**Microgreen Type:** {microgreen_type}")
    st.write(f"**Growth Stage:** {growth_stage}")
