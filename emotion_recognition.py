import cv2
from deepface import DeepFace
import os

# Path to your known faces folder
known_faces_dir = "known_faces"

# Initialize webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the current frame for faces + emotions
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Handle case when multiple faces are detected
        if not isinstance(results, list):
            results = [results]

        for result in results:
            # Get coordinates of detected face
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            x, y = max(0, x), max(0, y)

            # Crop the face for identification
            face_crop = frame[y:y+h, x:x+w]
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_crop)

            # Try identifying the face
            try:
                df = DeepFace.find(img_path=temp_path, db_path=known_faces_dir, enforce_detection=False, detector_backend='opencv')
                if len(df) > 0 and len(df[0]) > 0:
                    name = os.path.splitext(os.path.basename(df[0].iloc[0]['identity']))[0]
                else:
                    name = "Unknown"
            except:
                name = "Unknown"

            # Get emotion
            emotion = result['dominant_emotion']

            # Draw bounding box and label
            label = f"{name} - {emotion}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    except Exception as e:
        print("[ERROR]", e)

    cv2.imshow("Face & Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()