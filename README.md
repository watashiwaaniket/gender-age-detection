# 👤 Gender and Age Detection Web App

A simple Streamlit web application that detects gender and age from uploaded images using deep learning and OpenCV's pre-trained Caffe models.

---

## ⚙️ Setting Up the Project

Clone this repository and then install the required dependencies
```
pip install streamlit opencv-python numpy
```
After this run the project using this command
```
streamlit run app.py
```
---

## 🚀 Features

- Upload an image
- Automatically detects faces in the image
- Predicts gender and age range for each face
- Displays the annotated image with predictions

---

## 🧠 Models Used

- **Face Detector**: `opencv_face_detector_uint8.pb` & `opencv_face_detector.pbtxt`
- **Age Detector**: `age_net.caffemodel` & `age_deploy.prototxt`
- **Gender Detector**: `gender_net.caffemodel` & `gender_deploy.prototxt`

These models are based on pre-trained Caffe networks and can detect human faces, then estimate age and gender.

---

## 📁 Folder Structure

gender_age_detector/
│
├── models/
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
│
├── app.py
└── requirements.txt