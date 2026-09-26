# DrawSpace

DrawSpace is a real-time virtual drawing application built with Python, OpenCV, MediaPipe, and NumPy.

The application uses hand tracking through a webcam to allow users to draw on a digital canvas using simple hand gestures.

## Features

- Real-time hand tracking using MediaPipe
- Webcam input using OpenCV
- Gesture-based drawing
- Color selection
- Eraser mode
- Digital canvas
- Adjustable brush and eraser thickness
- Real-time rendering of the camera feed and canvas

## How It Works

The application follows this workflow:

-text
Webcam
   ?
OpenCV captures video
   ?
MediaPipe detects hand landmarks
   ?
Index and middle fingers are tracked
   ?
Gesture is detected
   ?
Color selection / Drawing / Erasing
   ?
Canvas is rendered with the camera feed