# Real-Time Emotion Detection System

A Python-based real-time emotion detection system that uses computer vision and deep learning to detect and analyze facial emotions through a webcam feed.

## Features

- Real-time emotion detection using webcam
- Automatic capture of specific emotions (happy, surprise, angry, fear, sad)
- Manual snapshot capture
- Face detection and tracking
- Debug mode for detailed logging
- Configurable auto-capture settings

## Requirements

- Python 3.x
- OpenCV
- DeepFace
- TensorFlow (automatically installed with DeepFace)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/rohit-singh2005/Real-Time-Emotion-Detection-.git
cd Real-Time-Emotion-Detection-
```

2. Install the required packages:
```bash
pip install opencv-python deepface
```

## Usage

Run the main script:
```bash
python prj2/emotion1.py
```

### Controls
- Press 'q' to quit the application
- Press 's' to take a manual snapshot
- Press 'c' to toggle auto-capture mode
- Press 'd' to toggle debug mode

## Project Structure

- `prj2/emotion1.py` - Main application file
- `prj2/emotion_snapshots/` - Directory where captured images are stored
- `prj2/known_faces/` - Directory for known face references

## How It Works

The system uses OpenCV for face detection and DeepFace for emotion analysis. It processes the webcam feed in real-time, detecting faces and analyzing their emotional expressions. When specific emotions are detected, the system can automatically capture snapshots or allow manual captures.

## License

[Your chosen license]

## Contributing

Feel free to submit issues and enhancement requests! 