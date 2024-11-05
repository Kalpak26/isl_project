import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os

def record_video(duration=10, frame_width=640, frame_height=480, fps=20):
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.avi"
    
    print(f"Starting video recording... Saving to {output_filename}")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' codec for .avi files
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    # Record video for the specified duration
    num_frames = int(duration * fps)
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        out.write(frame)
        cv2.imshow('Recording...', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break
        
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)
    print("MediaPipe version:", mp.__version__)
    print("NumPy version:", np.__version__)
    
    record_video()
