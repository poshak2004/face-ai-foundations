# Face AI Foundations 🧠🎥

An end-to-end computer vision and deep learning project focused on **face detection, emotion recognition, and gender classification**, built from scratch to deeply understand how face-based AI systems work in practice.

This repository documents my **learning journey**, **experiments**, and **implementations**, moving from basic image processing to real-time inference using deep learning and transfer learning.

---

## 📌 Project Overview

This project was developed incrementally to learn and apply:

- Classical image processing
- Convolutional Neural Networks (CNNs)
- Transfer Learning with pretrained models
- Dataset preparation and preprocessing
- Real-time webcam inference pipelines
- Practical issues in modern ML tooling (Keras 3, model formats, deployment)

The focus is on **understanding why things work**, not just making them run.

---

## 🔍 What This Project Covers

### 1️⃣ Face Detection
- OpenCV Haar Cascades
- MediaPipe BlazeFace (real-time, efficient)
- Bounding box extraction
- Face cropping and preprocessing

### 2️⃣ Emotion Recognition (Custom CNN)
- Dataset: **FER-2013**
- Grayscale facial images
- Custom CNN architecture
- Image normalization and resizing
- Train/validation split
- Real-time emotion prediction using webcam

**Emotions detected:**
- Angry
- Happy
- Sad
- Surprise
- Neutral

---

### 3️⃣ Gender Classification (Transfer Learning)
- Pretrained **MobileNetV2**
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
- Real-time ML inference
- Keras 3 model format changes (`.keras`)
- Debugging ML pipelines end-to-end
- Ethical considerations in face-based AI systems

---

## 🗂 Repository Structure

face-ai-foundations/
├── src/ # Training and real-time inference scripts
├── docs/ # Learning notes and architecture explanations
├── datasets/ # Ignored (large datasets not tracked)
│ └── README.md
├── models/ # Ignored (trained models not tracked)
│ └── README.md
├── .gitignore
└── README.md

yaml
Copy code

> Datasets and trained models are intentionally excluded from version control to keep the repository lightweight, clean, and reproducible.

---

## 🚀 How to Run (High-Level)

> Assumes Python environment with required dependencies installed.

### Real-Time Emotion Detection
```bash
python src/realtime_emotion_detection.py
Real-Time Gender Detection
bash
Copy code
python src/realtime_gender_detection.py
Press q to exit the webcam window.

⚠️ Ethics & Limitations
Gender classification is binary due to dataset limitations

Emotion prediction is probabilistic and sensitive to:

Lighting

Pose

Facial occlusion

Dataset bias

These models are built for learning and demonstration purposes only

Not intended for real-world decision-making systems

Acknowledging these limitations is an important part of responsible AI development.

📈 Results (Indicative)
Emotion CNN: Learns meaningful facial expression patterns from FER-2013

Gender classifier:

High training accuracy using transfer learning

Reasonable validation performance given dataset size

Improved stability with preprocessing alignment and smoothing

Exact performance varies based on data quality and environment.

🎯 Motivation
This project was built to move beyond tutorials and understand:

How datasets shape model behavior

Why preprocessing matters

What breaks in real-time systems

How modern ML tooling actually behaves

The goal was learning depth, not just output accuracy.

🔮 Future Work
Age prediction (regression)

Multi-head model (Age + Gender)

Bias analysis and robustness testing

Model optimization for edge devices

Better dataset balancing and evaluation

📚 Acknowledgements
OpenCV

MediaPipe

TensorFlow / Keras

FER-2013 dataset

MobileNet architecture

👤 Author
Poshak K
Undergraduate student — Information Science & Engineering
Exploring computer vision and applied deep learning

markdown
Copy code

---

### ✅ What this README does right
- Shows **learning progression**
- Explains **why**, not just **what**
- Looks professional to:
  - Recruiters
  - Professors
  - Interviewers
- Matches real ML repo standards
- Doesn’t expose datasets or private images
