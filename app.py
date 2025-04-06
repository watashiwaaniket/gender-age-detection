import streamlit as st
import cv2
import numpy as np

# Paths
MODEL_DIR = "models/"
FACE_PROTO = MODEL_DIR + "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR + "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR + "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR + "age_net.caffemodel"
GENDER_PROTO = MODEL_DIR + "gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR + "gender_net.caffemodel"

# Labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load networks
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

def detect_faces(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    h, w = image.shape[:2]
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype(int)
            boxes.append(box)
    return boxes

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    genderNet.setInput(blob)
    gender = GENDER_LIST[genderNet.forward()[0].argmax()]

    ageNet.setInput(blob)
    age = AGE_LIST[ageNet.forward()[0].argmax()]

    return gender, age

# Streamlit UI
st.set_page_config(page_title="Gender & Age Detector", layout="centered")
st.title("🧑🏻‍💻 Gender and Age Detection")
st.write("Upload an image to detect faces, gender and age.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    #st.write("Image uploaded successfully!")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    original_image = image.copy()
    #boxes = detect_faces(image)

    boxes = detect_faces(image)
    st.write(f"Detected {len(boxes)} face(s)")
    for box in boxes:
        x1, y1, x2, y2 = box
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue
        gender, age = predict_age_gender(face)
        label = f"{gender}, {age}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Result")

