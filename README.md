# Face Detection and Emotion Recognition

Real-time facial emotion recognition from webcam using OpenCV for face detection and a Keras/TensorFlow CNN for emotion classification. Supports seven classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Requirements

- Python 3.x
- Dependencies in `reqs.txt`: numpy, opencv-python, tensorflow; streamlit only if using the web UI.

## Setup

```bash
pip install -r reqs.txt
```

Ensure `emotion_model.h5` is present in the project root. The face detector uses OpenCV’s bundled Haar cascade (`haarcascade_frontalface_default.xml`).

## Usage

**Local camera (recommended)**  
Runs from the command line and displays an OpenCV window. Press `q` to quit.

```bash
python app2.py
```

**Web UI (Streamlit)**  
Starts a browser-based interface with a “Start Webcam” button.

```bash
streamlit run app.py
```

## Project layout

| File | Purpose |
|------|--------|
| `app2.py` | Local camera emotion detection (OpenCV window) |
| `app.py` | Streamlit web app for the same pipeline |
| `emotion_model.h5` | Pre-trained Keras emotion model (48×48 grayscale input) |
| `training.ipynb`, `testing.ipynb` | Model training and evaluation |
| `stressDetection.ipynb` | Stress-related analysis |

## License

Not specified.
