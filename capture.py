import cv2
import mediapipe as mp
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
import time

# Load your trained model
model = load_model('saved_models/best_emotion_model.h5')
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Create directory for captured faces if it doesn't exist
if not os.path.exists('captured_faces'):
    os.makedirs('captured_faces')

cap = cv2.VideoCapture(0)
photo_taken = False
photo_timer = 0
last_prediction = None
last_pred_timer = 0

# Helper to preprocess for model (grayscale, no resize, no normalization)
def preprocess_for_model(face_gray):
    # Model expects (batch, h, w, 1) and will resize/normalize internally
    face_input = np.expand_dims(face_gray, axis=-1)  # (h, w, 1)
    face_input = np.expand_dims(face_input, axis=0)  # (1, h, w, 1)
    return face_input

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect faces
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            # Get the bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)
            
            # Tighter cropping: shrink the box horizontally and vertically
            crop_factor = 0.85  # Keep only 85% of the width and height, centered
            new_bw = int(bw * crop_factor)
            new_bh = int(bh * crop_factor)
            x = x + (bw - new_bw) // 2
            y = y + (bh - new_bh) // 2
            bw = new_bw
            bh = new_bh
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            # Extract and display the face region
            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size > 0:
                # Convert face to grayscale for saving
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                # Display the grayscale face in the top-right corner (resized for preview only)
                face_preview = cv2.resize(face_gray, (48, 48))
                frame[10:58, 10:58] = cv2.cvtColor(face_preview, cv2.COLOR_GRAY2BGR)
                # Draw a border around the small face preview
                cv2.rectangle(frame, (10, 10), (58, 58), (0, 255, 0), 1)
                
                # Save photo and predict when 'p' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p') and not photo_taken:
                    # Take 5 images in quick succession
                    faces = []
                    for i in range(5):
                        ret, f = cap.read()
                        if not ret:
                            continue
                        rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        res = face_detection.process(rgb_f)
                        if res.detections:
                            det = res.detections[0]
                            bboxC2 = det.location_data.relative_bounding_box
                            h2, w2, _ = f.shape
                            x2 = int(bboxC2.xmin * w2)
                            y2 = int(bboxC2.ymin * h2)
                            bw2 = int(bboxC2.width * w2)
                            bh2 = int(bboxC2.height * h2)
                            # Same tight crop
                            new_bw2 = int(bw2 * crop_factor)
                            new_bh2 = int(bh2 * crop_factor)
                            x2 = x2 + (bw2 - new_bw2) // 2
                            y2 = y2 + (bh2 - new_bh2) // 2
                            bw2 = new_bw2
                            bh2 = new_bh2
                            face_roi2 = f[y2:y2+bh2, x2:x2+bw2]
                            if face_roi2.size > 0:
                                face_gray2 = cv2.cvtColor(face_roi2, cv2.COLOR_BGR2GRAY)
                                faces.append(face_gray2)
                        time.sleep(0.05)  # Small delay between captures
                    # Calculate mean probabilities across all samples
                    mean_probs = np.mean([model.predict(preprocess_for_model(fg), verbose=0)[0] for fg in faces], axis=0)
                    top2_indices = mean_probs.argsort()[-2:][::-1]
                    top1_conf = mean_probs[top2_indices[0]]
                    
                    # Only show top-2 if confidence is low (below 70%)
                    if top1_conf < 0.7:
                        print(f"\nTop-2 Predictions:")
                        print(f"1. {EMOTIONS[top2_indices[0]]} ({top1_conf*100:.1f}%)")
                        print(f"2. {EMOTIONS[top2_indices[1]]} ({mean_probs[top2_indices[1]]*100:.1f}%)")
                        print(f"Full probabilities: {[f'{p*100:.1f}%' for p in mean_probs]}")
                    else:
                        print(f"\nPrediction: {EMOTIONS[top2_indices[0]]} ({top1_conf*100:.1f}%)")
                        print(f"Full probabilities: {[f'{p*100:.1f}%' for p in mean_probs]}")
                    
                    # Save the first face with the top prediction
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    label = EMOTIONS[top2_indices[0]]
                    label_dir = os.path.join('captured_faces', label)
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)
                    filename = os.path.join(label_dir, f'face_{timestamp}_{top1_conf:.2f}.png')
                    cv2.imwrite(filename, faces[0])
                    photo_taken = True
                    photo_timer = 30  # Show indicator for 30 frames
                
                # Show "Photo Taken!" indicator
                if photo_timer > 0:
                    cv2.putText(frame, "Photo Taken!", (10, 80), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    photo_timer -= 1
                else:
                    photo_taken = False
    # Show last prediction
    if last_pred_timer > 0 and last_prediction is not None:
        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        last_pred_timer -= 1
    # Display the frame
    cv2.imshow('Face Detection', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()

        
