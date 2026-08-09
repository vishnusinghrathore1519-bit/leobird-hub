import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🎯 Leobird AI Detector")

# Model load
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Camera Input
img_file_buffer = st.camera_input("Camera kholo aur photo lo")

if img_file_buffer is not None:
    # Image open karo PIL se
    image = Image.open(img_file_buffer)
    
    # Predict
    results = model(image)
    res_plotted = results[0].plot()
    
    # Show output
    st.image(res_plotted, caption="Detection Result", use_container_width=True)
    
