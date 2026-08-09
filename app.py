import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.title("🎯 Leobird AI Detector")

# Model load
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Camera Input
img_file_buffer = st.camera_input("Camera kholo aur photo lo")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Predict
    results = model(cv2_img)
    res_plotted = results[0].plot()
    
    # Show output
    st.image(res_plotted, caption="Detection Result", use_container_width=True)
  
