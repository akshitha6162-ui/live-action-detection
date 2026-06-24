import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
from PIL import Image
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Road Object Detection",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🚦 Smart Road Object Detection Dashboard")
st.markdown("Real-Time Road Object Detection using YOLOv8")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.4,
    0.05
)

source = st.sidebar.radio(
    "Select Input Source",
    ["Image", "Video"]
)

# --------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------
if source == "Image":

    uploaded_image = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image)
        image_np = np.array(image)

        results = model(
            image_np,
            conf=confidence
        )

        annotated = results[0].plot()

        names = model.names

        vehicles = 0
        people = 0
        bikes = 0

        if len(results[0].boxes) > 0:

            classes = results[0].boxes.cls.cpu().numpy()

            for cls in classes:

                label = names[int(cls)]

                if label in ["car", "truck", "bus"]:
                    vehicles += 1

                elif label == "person":
                    people += 1

                elif label in ["motorcycle", "bicycle"]:
                    bikes += 1

        col1, col2, col3 = st.columns(3)

        col1.metric("🚗 Vehicles", vehicles)
        col2.metric("🚶 People", people)
        col3.metric("🏍 Bikes", bikes)

        st.image(
            annotated,
            caption="Detected Objects",
            use_container_width=True
        )

# --------------------------------------------------
# VIDEO DETECTION
# --------------------------------------------------
else:

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        vehicle_metric = st.empty()
        people_metric = st.empty()
        bike_metric = st.empty()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = model(
                frame,
                conf=confidence,
                verbose=False
            )

            annotated = results[0].plot()

            vehicles = 0
            people = 0
            bikes = 0

            if len(results[0].boxes) > 0:

                classes = results[0].boxes.cls.cpu().numpy()

                for cls in classes:

                    label = model.names[int(cls)]

                    if label in ["car", "truck", "bus"]:
                        vehicles += 1

                    elif label == "person":
                        people += 1

                    elif label in ["motorcycle", "bicycle"]:
                        bikes += 1

            col1, col2, col3 = st.columns(3)

            col1.metric("🚗 Vehicles", vehicles)
            col2.metric("🚶 People", people)
            col3.metric("🏍 Bikes", bikes)

            frame_placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, OpenCV and YOLOv8"
)
