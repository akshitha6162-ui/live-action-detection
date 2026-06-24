import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Page Configuration
st.set_page_config(
    page_title="Road Object Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Real-Time Road Object Detection")
st.write(
    "Detect vehicles, pedestrians, traffic lights, and other road objects using YOLOv8."
)

# Load Model Once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

ROAD_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign"
]


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=img,
            conf=0.4,
            verbose=False
        )

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[cls_id]

                if class_name in ROAD_CLASSES:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="road-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
