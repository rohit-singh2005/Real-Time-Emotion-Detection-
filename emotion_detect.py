import os
import cv2
from deepface import DeepFace

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]

        try:
            # Emotion detection
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)[0]
            emotion = result.get('dominant_emotion', 'No Emotion')

            if emotion in ['neutral', 'sad']:
                emotion += ' (Improve lighting/position)'

            # Face recognition
            recognition = DeepFace.find(img_path=face_region, db_path="face_db", enforce_detection=False)
            identity = "Unknown"

            if len(recognition) > 0 and not recognition[0].empty:
                identity_path = recognition[0].iloc[0]['identity']
                identity = os.path.splitext(os.path.basename(identity_path))[0]

            # Display emotion and name
            cv2.putText(frame, f'{emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'User: {identity}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        except Exception as e:
            print("Analysis/Recognition error:", e)
            pass

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Emotion & Face Recognition', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()