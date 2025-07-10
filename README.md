# Real-Time Emotion Recognition on Raspberry Pi

## 📌 Project Summary

This project presents a portable, Raspberry Pi-based device designed to classify and display facial emotions in real time. The system uses a camera to capture facial expressions, processes the image through a trained CNN model, and displays the recognized **emotion label(s)** on a small OLED screen.

Designed with accessibility in mind, the device helps individuals (particularly those with social communication challenges) interpret emotional cues visually and clearly.

## 🧠 System Overview

- A **Raspberry Pi 5** and **NoIR Camera** are used to capture facial expressions.
- A **CNN model**, trained on the **FER2013 dataset** (with adjusted labels), classifies the expression into one of six categories:
  - Angry
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- The **top prediction** is displayed on the **2.23” OLED screen**.
- If model confidence is **below a certain threshold**, the system uses a **Top-K strategy** to display the top 2 predicted emotions, reducing false negatives and increasing interpretability.

## 💡 Example Display Logic

- High-confidence result → OLED shows:  
  `Happy`

- Low-confidence result → OLED shows:  
  `Happy / Neutral` TOP-K (k=2)

## 🛠 Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- Raspberry Pi 5  
- Logitech Webcam 
- OLED Display Module (128×32 px, I2C)  
- FER2013 Dataset

## 📷 Device Preview
<img width="648" height="671" alt="Ekran görüntüsü 2025-05-07 200400" src="https://github.com/user-attachments/assets/50769b59-22ab-4b60-84c8-28a61fddbf30" />

<img width="641" height="506" alt="Ekran görüntüsü 2025-05-07 144622" src="https://github.com/user-attachments/assets/4827bdc5-873f-4197-bd03-1f424bba21a3" />

<img width="650" height="395" alt="Ekran görüntüsü 2025-05-07 153019" src="https://github.com/user-attachments/assets/b6bfe44e-b85c-4f12-8046-4ddaddce42bf" />

## 📄 Thesis Document

<img width="784" height="1116" alt="image" src="https://github.com/user-attachments/assets/8e50b4eb-6c11-471f-81d1-637c7c775024" />

[Fer-Tez.pdf](https://github.com/user-attachments/files/21164580/Fer-Tez.pdf)

