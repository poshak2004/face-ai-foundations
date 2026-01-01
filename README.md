# Face AI Foundations 🧠🎥

An end-to-end computer vision and deep learning project focused on **face detection, emotion recognition, and gender classification**, built from scratch to understand how face-based AI systems work in practice.

This repository documents my **learning journey**, **experiments**, and **implementations**, progressing from basic image processing to real-time inference using deep learning and transfer learning.

---

## 📌 Project Overview

This project was developed incrementally to learn and apply:

- Classical image processing techniques
- Convolutional Neural Networks (CNNs)
- Transfer learning with pretrained models
- Dataset preparation and preprocessing
- Real-time webcam inference pipelines
- Practical issues in modern ML tooling (Keras 3, model formats, deployment)

The focus is on **understanding why things work**, not just making them run.

---

## 🔍 What This Project Covers

### 1️⃣ Face Detection
- OpenCV Haar Cascades
- MediaPipe BlazeFace (real-time and efficient)
- Bounding box extraction
- Face cropping and preprocessing

### 2️⃣ Emotion Recognition (Custom CNN)
- Dataset: FER-2013
- Grayscale facial images
- Custom CNN architecture
- Image normalization and resizing
- Train / validation split
- Real-time emotion prediction using webcam

**Emotions detected**
- Angry  
- Happy  
- Sad  
- Surprise  
- Neutral  

### 3️⃣ Gender Classification (Transfer Learning)
- Pretrained MobileNetV2
- Binary classification: Male / Female
- Data augmentation for robustness
- Partial fine-tuning of backbone layers
- Real-time webcam inference with confidence scores
- Temporal smoothing to reduce flickering predictions

---

## 🧠 Key Concepts Learned

- Image representation (pixels, channels, normalization)
- OpenCV image processing pipelines
- CNN design and training from scratch
- Transfer learning and fine-tuning strategies
- Data augmentation and overfitting control
- Real-time ML inference pipelines
- Keras 3 model format changes (.keras)
- Debugging ML systems end-to-end
- Ethical considerations in face-based AI systems

---

## 🗂 Repository Structure

face-ai-foundations  
├── src        (training and real-time inference scripts)  
├── docs       (learning notes and architecture explanations)  
├── datasets   (ignored – large datasets not tracked)  
│   └── README.md  
├── models     (ignored – trained models not tracked)  
│   └── README.md  
├── .gitignore  
└── README.md  

Datasets and trained models are intentionally excluded from version control to keep the repository lightweight, clean, and reproducible.

---

## 🚀 How to Run (High-Level)

Assumes a Python environment with required dependencies installed.

**Real-Time Emotion Detection**

python src/realtime_emotion_detection.py

**Real-Time Gender Detection**

python src/realtime_gender_detection.py

Press **q** to exit the webcam window.

---

## ⚠️ Ethics & Limitations

- Gender classification is binary due to dataset limitations
- Emotion prediction is probabilistic and sensitive to:
  - Lighting
  - Pose
  - Facial occlusion
  - Dataset bias
- These models are built for learning and demonstration purposes only
- Not intended for real-world decision-making systems

Acknowledging these limitations is an important part of responsible AI development.

---

## 📈 Results (Indicative)

- Emotion CNN learns meaningful facial expression patterns from FER-2013
- Gender classifier:
  - High training accuracy using transfer learning
  - Reasonable validation performance given dataset size
  - Improved stability with preprocessing alignment and smoothing

Exact performance may vary depending on data quality and environment.

---

## 🎯 Motivation

This project was built to move beyond tutorials and gain a deeper understanding of:

- How datasets shape model behavior
- Why preprocessing choices matter
- What breaks in real-time ML systems
- How modern ML tooling behaves in practice

The goal was **learning depth**, not just output accuracy.

---

## 🔮 Future Work

- Age prediction (regression)
- Multi-head models (Age + Gender)
- Bias analysis and robustness testing
- Model optimization for edge devices
- Improved dataset balancing and evaluation

---

## 📚 Acknowledgements

- OpenCV  
- MediaPipe  
- TensorFlow / Keras  
- FER-2013 Dataset  
- MobileNet Architecture  

---

## 👤 Author

Poshak K  
Undergraduate Student — Information Science & Engineering  
Exploring Computer Vision and Applied Deep Learning
