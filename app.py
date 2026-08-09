import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

st.title("🎯 Leobird Turbo Live")

@st.cache_resource
def load_model():
    # Agar model slow hai toh 'yolov8n.pt' (Nano) use karna zyada fast hoga
    return YOLO('best.pt')

model = load_model()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # SUPER FAST: Sirf har 10th frame process karo (10x fast)
        if self.frame_count % 10 == 0:
            # Resizing to 320x320 (Model inference bahut fast ho jayega)
            img_small = cv2.resize(img, (320, 320))
            results = model(img_small, conf=0.6, verbose=False)
            
            # Plotting result on resized image
            res_plotted = results[0].plot()
            # wapas resize karo original screen size par
            self.latest_frame = cv2.resize(res_plotted, (img.shape[1], img.shape[0]))
            
        if self.latest_frame is not None:
            return av.VideoFrame.from_ndarray(self.latest_frame, format="bgr24")
        else:
            return frame

webrtc_streamer(
    key="turbo-live",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
