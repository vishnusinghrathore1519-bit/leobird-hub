import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🎯 Leobird AI Fast Detector")

# Model Load (Cached)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Fast Camera Input
img_file_buffer = st.camera_input("Camera on karo")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    
    # Prediction (conf=0.6 se faltu detections khatam ho jayenge)
    results = model(image, conf=0.6)
    res_plotted = results[0].plot()
    
    # Image show
    st.image(res_plotted, caption="Detection Result", use_container_width=True)
    
