# 🚗 Real-Time Road Object Detection Using YOLOv8 and Streamlit

## 📌 Project Overview

This project is a real-time road object detection system that uses a live webcam feed to identify and classify common road objects such as cars, buses, trucks, motorcycles, bicycles, pedestrians, traffic lights, and stop signs.

The application is built using **Streamlit**, **OpenCV**, and **YOLOv8 (You Only Look Once)**, providing a simple and interactive interface for real-time computer vision applications.

---

## 🎯 Objectives

* Detect road objects in real time using a webcam.
* Improve road safety awareness through object recognition.
* Demonstrate the application of deep learning in intelligent transportation systems.
* Provide an easy-to-use dashboard for monitoring detected objects.

---

## 🛠️ Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Programming Language      |
| Streamlit        | Web Application Framework |
| OpenCV           | Image Processing          |
| YOLOv8           | Object Detection Model    |
| streamlit-webrtc | Live Webcam Streaming     |
| NumPy            | Numerical Computations    |

---

## 📂 Project Structure

```text
Road-Object-Detection/
│
├── app.py
├── requirements.txt
├── README.md
└── yolov8n.pt
```

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/road-object-detection.git
cd road-object-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit ultralytics opencv-python av streamlit-webrtc
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🔍 Detected Objects

The model can detect:

* Person
* Bicycle
* Car
* Motorcycle
* Bus
* Truck
* Traffic Light
* Stop Sign

---

## 🧠 Model Information

This project uses the YOLOv8 Nano model:

```python
model = YOLO("yolov8n.pt")
```

Advantages:

* Fast inference speed
* Lightweight model
* Suitable for real-time applications
* High accuracy for common road objects

---

## 📸 Features

✅ Real-time webcam object detection

✅ Bounding box visualization

✅ Confidence score display

✅ Road object filtering

✅ Streamlit interactive dashboard

✅ Lightweight and easy deployment

---

## 🚀 Future Enhancements

* Vehicle counting system
* Speed estimation
* Lane detection
* Traffic density monitoring
* Accident detection
* Real-time alert notifications
* Data logging and reporting
* Integration with CCTV cameras
* Smart traffic management dashboard

---

## 📊 Applications

* Smart Cities
* Traffic Monitoring
* Autonomous Vehicles
* Driver Assistance Systems
* Road Safety Analysis
* Intelligent Transportation Systems

---

## 📈 Expected Outcomes

The system successfully identifies road objects from a live camera feed and displays them with bounding boxes and confidence scores. The project demonstrates how modern deep learning models can be applied to real-time transportation and traffic management scenarios.

---

## 👨‍💻 Author

**Aishu**

Road Object Detection using YOLOv8 and Streamlit

Feel free to use, modify, and distribute this project for educational and research purposes.
