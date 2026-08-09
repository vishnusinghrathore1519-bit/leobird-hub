import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

st.title("🎯 Leobird AI Fast Live")

# Model load
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Sirf har 5th frame process karenge (Speed badhane ke liye)
        if self.frame_count % 5 == 0:
            # Image resize (Fast inference)
            img_small = cv2.resize(img, (640, 480))
            results = model(img_small, conf=0.55, verbose=False)
            self.res_plotted = results[0].plot()
            # wapas resize to original
            self.res_plotted = cv2.resize(self.res_plotted, (img.shape[1], img.shape[0]))
            
        # Agar frame processed nahi hai, to original dikhao
        if hasattr(self, 'res_plotted'):
            return av.VideoFrame.from_ndarray(self.res_plotted, format="bgr24")
        else:
            return frame

webrtc_streamer(
    key="leobird",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
