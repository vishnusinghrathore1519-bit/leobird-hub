import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

st.title("🎯 Leobird Turbo Live")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Frame skipping for speed
        if self.frame_count % 10 == 0:
            img_small = cv2.resize(img, (320, 320))
            results = model(img_small, conf=0.6, verbose=False)
            res_plotted = results[0].plot()
            self.latest_frame = cv2.resize(res_plotted, (img.shape[1], img.shape[0]))
            
        if self.latest_frame is not None:
            return av.VideoFrame.from_ndarray(self.latest_frame, format="bgr24")
        else:
            return frame

# Multi STUN servers for iPhone / Safari Compatibility
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
    ]
}

webrtc_streamer(
    key="turbo-live-iphone",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
