import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("🎯 Leobird AI Live Detector")

# Model load (Cached)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Class for live video
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO Prediction
        results = model(img)
        res_plotted = results[0].plot()
        
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

# Streamer
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

    
