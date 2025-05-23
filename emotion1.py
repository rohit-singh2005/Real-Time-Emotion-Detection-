import os
import cv2
from deepface import DeepFace
import time
import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create directory for saving snapshots
snapshots_dir = "emotion_snapshots"
if not os.path.exists(snapshots_dir):
    os.makedirs(snapshots_dir)

# Emotions to automatically capture
emotions_to_capture = ['happy', 'surprise', 'angry', 'fear', 'sad']

# Cooldown time between captures (in seconds)
capture_cooldown = 2.0
last_capture_time = {}  # Track last capture time for each emotion

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Debug mode
debug_mode = True

# Auto-capture mode
auto_capture = True

print("Starting emotion detection...")
print("Press 'q' to quit, 's' to take a snapshot, 'c' to toggle auto-capture")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break
    
    # Make a clean copy for saving
    clean_frame = frame.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using OpenCV's Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    current_time = time.time()
    
    # Find the largest face (assuming it's the main subject)
    main_face = None
    largest_area = 0
    
    for (x, y, w, h) in faces:
        # Calculate face area
        area = w * h
        
        # Skip if face is too small
        if w < 30 or h < 30:
            continue
            
        # Update if this face is larger than previous ones
        if area > largest_area:
            largest_area = area
            main_face = (x, y, w, h)
    
    # Process only the largest face if found
    if main_face:
        x, y, w, h = main_face
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # Emotion analysis
            result = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=False
            )
            
            # Extract emotion
            if result and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                
                # Display emotion text
                cv2.putText(frame, emotion, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                if debug_mode:
                    print(f"Detected emotion: {emotion}")
                
                # Auto-capture specific emotions
                if auto_capture and emotion in emotions_to_capture:
                    # Check if cooldown period has passed for this emotion
                    if (emotion not in last_capture_time or 
                        (current_time - last_capture_time[emotion]) >= capture_cooldown):
                        
                        # Create timestamp for filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{emotion}_{timestamp}.jpg"
                        filepath = os.path.join(snapshots_dir, filename)
                        
                        # Save the image
                        cv2.imwrite(filepath, clean_frame)
                        
                        # Update last capture time for this emotion
                        last_capture_time[emotion] = current_time
                        
                        # Display capture notification
                        cv2.putText(frame, f"Captured: {emotion}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"Auto-captured {emotion} emotion")
                
        except Exception as e:
            if debug_mode:
                print(f"Error: {str(e)}")
    elif debug_mode and len(faces) > 0:
        print("Skipped processing faces that were too small or no suitable face found")
    
    # Display instructions
    cv2.putText(frame, "Press 'q' to quit, 's' for snapshot, 'c' for auto-capture", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display auto-capture status
    status = "ON" if auto_capture else "OFF"
    cv2.putText(frame, f"Auto-capture: {status}", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display which emotions are being captured
    if auto_capture:
        emotions_text = ", ".join(emotions_to_capture)
        cv2.putText(frame, f"Capturing: {emotions_text}", (10, frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Quit on 'q'
    if key == ord('q'):
        break
    
    # Take snapshot on 's'
    elif key == ord('s'):
        if main_face:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_{timestamp}.jpg"
            filepath = os.path.join(snapshots_dir, filename)
            cv2.imwrite(filepath, clean_frame)
            print(f"Manual snapshot saved: {filepath}")
        else:
            print("No face detected for snapshot")
    
    # Toggle auto-capture on 'c'
    elif key == ord('c'):
        auto_capture = not auto_capture
        print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
    
    # Toggle debug mode on 'd'
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Emotion detection stopped")