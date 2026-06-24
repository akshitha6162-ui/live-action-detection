import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Road Object Detection", layout="wide")

st.title("🚗 Real-Time Road Object Detection")
st.write("Detect vehicles, pedestrians, traffic lights, and more using YOLOv8.")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Road-related classes
ROAD_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign"
]

class VideoProcessor(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLO Detection
        results = model(img)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = model.names[cls_id]

                if class_name in ROAD_OBJECTS:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = f"{class_name} {conf:.2f}"

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        return img

st.subheader("📷 Live Camera Feed")

webrtc_streamer(
    key="road-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)