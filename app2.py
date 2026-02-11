import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror/invert camera horizontally

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)[0]
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
