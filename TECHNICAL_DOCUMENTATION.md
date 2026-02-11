# Technical Documentation: Face Detection and Emotion Recognition System

## 1. Overview

This document provides a formal technical specification of the face detection and facial emotion recognition pipeline implemented in this project. The system performs real-time emotion classification from webcam feed through a multi-stage pipeline comprising video capture, face detection, region extraction, preprocessing, and emotion classification.

---

## 2. Pipeline Architecture

The inference pipeline consists of the following sequential components:

1. **Video Capture** — Acquisition of frames from the default webcam device  
2. **Frame Preprocessing** — Optional horizontal flip (mirror mode in `app2.py`) and color space conversion  
3. **Face Detection** — Localization of face regions within each frame  
4. **Region of Interest (ROI) Extraction** — Cropping and resizing of detected face regions  
5. **Emotion Classification** — CNN-based prediction of emotion labels and confidence scores  
6. **Output Rendering** — Overlay of bounding boxes and labels on the display frame  

---

## 3. Face Detection Model

### 3.1 Technology

**Model:** Haar Cascade Classifier (Viola-Jones detector)  
**Implementation:** OpenCV (`cv2.CascadeClassifier`)  
**Classifier File:** `haarcascade_frontalface_default.xml`  
**Source:** OpenCV bundled data (`cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'`)

### 3.2 Technical Specification

| Parameter | Value |
|-----------|-------|
| Cascade Type | BOOST with Haar-like features |
| Detection Window | 24×24 pixels (minimum scale) |
| Stages | 211 cascaded classifier stages |
| Face Orientation | Frontal, upright faces |

### 3.3 Working Principle

The Haar cascade uses a series of Haar-like rectangular features computed at multiple scales. Each stage applies thresholds to filter out non-face regions; only regions passing all stages are returned as face detections. This cascaded structure enables efficient rejection of background regions early in the pipeline.

### 3.4 API Usage

```python
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `scaleFactor` | 1.3 | Scale reduction between pyramid levels (image downscaling for multi-scale search) |
| `minNeighbors` | 5 | Minimum neighbors required to retain a detection (higher values reduce false positives) |

**Input:** Grayscale image (`cv2.COLOR_BGR2GRAY`)  
**Output:** List of bounding boxes `(x, y, w, h)` for each detected face  

---

## 4. CNN Architecture (Emotion Classification Model)

### 4.1 Framework and Format

**Framework:** TensorFlow / Keras  
**Model File:** `emotion_model.h5` (HDF5/Keras legacy format)  
**Model Type:** `tf.keras.models.Sequential`  

### 4.2 Layer-by-Layer Specification

| Layer | Type | Configuration | Output Shape (approx.) |
|-------|------|---------------|------------------------|
| 1 | Conv2D | 64 filters, 3×3 kernel, ReLU, input (48, 48, 1) | (46, 46, 64) |
| 2 | BatchNormalization | — | (46, 46, 64) |
| 3 | MaxPooling2D | 2×2 pool size | (23, 23, 64) |
| 4 | Dropout | 0.25 | (23, 23, 64) |
| 5 | Conv2D | 128 filters, 3×3 kernel, ReLU | (21, 21, 128) |
| 6 | BatchNormalization | — | (21, 21, 128) |
| 7 | MaxPooling2D | 2×2 pool size | (10, 10, 128) |
| 8 | Dropout | 0.25 | (10, 10, 128) |
| 9 | Conv2D | 256 filters, 3×3 kernel, ReLU | (8, 8, 256) |
| 10 | BatchNormalization | — | (8, 8, 256) |
| 11 | MaxPooling2D | 2×2 pool size | (4, 4, 256) |
| 12 | Dropout | 0.25 | (4, 4, 256) |
| 13 | Flatten | — | (4096,) |
| 14 | Dense | 256 units, ReLU | (256,) |
| 15 | Dropout | 0.5 | (256,) |
| 16 | Dense | 7 units, Softmax | (7,) |

### 4.3 Architecture Summary

- **Convolutional blocks:** 3 blocks, each with Conv2D → BatchNormalization → MaxPooling2D → Dropout  
- **Filters:** 64 → 128 → 256  
- **Kernel size:** 3×3 for all Conv2D layers  
- **Activation:** ReLU in convolutional and first dense layer; Softmax in output layer  
- **Regularization:** BatchNormalization, Dropout (0.25 after conv blocks, 0.5 before output)  

### 4.4 Input/Output Specification

- **Input shape:** `(48, 48, 1)` — 48×48 grayscale image, single channel  
- **Output:** 7-dimensional probability vector (one per emotion class)  
- **Output interpretation:** `argmax` yields predicted class; `max` yields confidence  

---

## 5. Emotion Detection Model (Training)

### 5.1 Dataset

**Source:** `train.csv` (FER2013-style format)  
**Columns:** `emotion`, `pixels`  
**Image format:** Space-separated grayscale pixel values; 2304 values per image (48×48)  
**Class encoding:** Integer labels 0–6  

### 5.2 Emotion Classes

| Index | Label |
|-------|-------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

### 5.3 Data Preprocessing

- **Pixel parsing:** `np.fromstring(pixels, sep=' ')`  
- **Reshape:** `(-1, 48, 48, 1)`  
- **Normalization:** Pixel values scaled to [0, 1] via division by 255.0  
- **Labels:** One-hot encoded via `to_categorical(emotion, num_classes=7)`  

### 5.4 Train/Validation Split

- **Split:** 90% training, 10% validation  
- **Method:** `sklearn.model_selection.train_test_split`  
- **Random seed:** 42  

### 5.5 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss | categorical_crossentropy |
| Metrics | accuracy |
| Epochs | 25 |
| Batch size | 64 |

---

## 6. Inference Pipeline (Application)

### 6.1 ROI Preprocessing for Emotion Model

For each detected face region `(x, y, w, h)`:

1. Extract grayscale ROI: `gray[y:y+h, x:x+w]`  
2. Resize to 48×48: `cv2.resize(roi_gray, (48, 48))`  
3. Normalize: `roi.astype('float') / 255.0`  
4. Add batch and channel dimensions: `(1, 48, 48, 1)`  

### 6.2 Prediction and Display

- **Prediction:** `model.predict(roi)[0]`  
- **Label:** `emotion_labels[np.argmax(prediction)]`  
- **Confidence:** `np.max(prediction)`  
- **Overlay:** Bounding box (`cv2.rectangle`) and label text (`cv2.putText`)  

---

## 7. Software Stack and Dependencies

| Library | Purpose | Version (from reqs.txt) |
|---------|---------|--------------------------|
| OpenCV (`opencv-python`) | Video capture, image processing, Haar cascade | — |
| TensorFlow | Model loading and inference | — |
| NumPy | Array operations | — |
| Streamlit | Web UI (`app.py`) | — |

Additional training dependencies: `pandas`, `scikit-learn`, `matplotlib`

---

## 8. Application Variants

| File | Interface | Notes |
|------|-----------|-------|
| `app.py` | Streamlit web app | Uses `st.image` for frame display; camera index 0 |
| `app2.py` | OpenCV window | Horizontal flip enabled for mirror view; `cv2.imshow` |

---

## 9. Supplementary Components

### 9.1 Stress Detection (stressDetection.ipynb)

A separate pipeline uses **dlib** for face detection and **68-point facial landmark prediction** (`shape_predictor_68_face_landmarks.dat`). Stress is inferred from geometric features (e.g., eyebrow-to-eye distance, eyebrow symmetry). This is independent of the main emotion recognition pipeline.

### 9.2 Asset Files

| File | Purpose |
|------|---------|
| `haarcascade_frontalface_default.xml` | Optional local Haar cascade (OpenCV bundle preferred) |
| `shape_predictor_68_face_landmarks.dat` | dlib 68-point face landmark model (stress pipeline) |
| `emotion_model.h5` | Trained emotion classification CNN |

---

## 10. Summary

The system combines a classical Haar cascade for face detection with a custom CNN for emotion classification. The CNN is a 3-block ConvNet with BatchNormalization and Dropout, trained on FER-style 48×48 grayscale faces for seven emotion classes. Inference runs in real time via OpenCV video capture and TensorFlow/Keras model loading.
