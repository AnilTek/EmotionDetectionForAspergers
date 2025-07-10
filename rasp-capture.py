import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import sys
import select
from PIL import Image, ImageDraw, ImageFont
from oled_spi_test import write_to_oled

model = load_model('best_emotion_model.h5')
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5,
    model_selection=0
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def preprocess_for_model(face_gray):
    face_input = np.expand_dims(face_gray, axis=-1)
    face_input = np.expand_dims(face_input, axis=0)
    return face_input

def get_face_roi(frame, detection):
    bboxC = detection.location_data.relative_bounding_box
    h, w, _ = frame.shape
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    bw = int(bboxC.width * w)
    bh = int(bboxC.height * h)

    crop_factor = 0.85
    new_bw = int(bw * crop_factor)
    new_bh = int(bh * crop_factor)
    x = x + (bw - new_bw) // 2
    y = y + (bh - new_bh) // 2

    return frame[y:y+new_bh, x:x+new_bw]

try:
    print("Ready. Press 'p' then Enter to predict, 'q' then Enter to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            write_to_oled("Camera error")
            time.sleep(0.1)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        i, _, _ = select.select([sys.stdin], [], [], 0.1)
        if i:
            key = sys.stdin.readline().strip().lower()
            if key == 'q':
                break
            elif key == 'p':
                faces = []
                for _ in range(5):
                    ret, f = cap.read()
                    if not ret:
                        continue
                    rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    res = face_detection.process(rgb_f)
                    if res.detections:
                        face_roi = get_face_roi(f, res.detections[0])
                        if face_roi.size > 0:
                            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            faces.append(face_gray)
                    time.sleep(0.05)

                if not faces:
                    write_to_oled("No face found")
                    continue

                mean_probs = np.mean(
                    [model.predict(preprocess_for_model(fg), verbose=0)[0] for fg in faces],
                    axis=0
                )
                top2_indices = mean_probs.argsort()[-2:][::-1]
                top1_conf = mean_probs[top2_indices[0]]

                if top1_conf < 0.70:
                    result = f"{EMOTIONS[top2_indices[0]]} - {EMOTIONS[top2_indices[1]]}"
                else:
                    result = f"{EMOTIONS[top2_indices[0]]}"

                print("Prediction:", result)
                write_to_oled(result)

except KeyboardInterrupt:
    print("Program terminated by user")

finally:
    cap.release()
    write_to_oled("Exiting...")
    time.sleep(1)
    write_to_oled("")