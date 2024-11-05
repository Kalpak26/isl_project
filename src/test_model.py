# src/test_model.py
import cv2
import numpy as np
import pickle
import os
from hand_detector import HandDetector

class SignPredictor:
    def __init__(self):
        self.detector = HandDetector()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load the trained model
        model_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_sign(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Detect hands and get landmarks
            frame = self.detector.find_hands(frame)
            landmarks = self.detector.get_landmarks_array()

            if landmarks is not None:
                # Reshape landmarks to match training data format
                landmarks = landmarks.reshape(1, -1)
                
                # Make prediction
                prediction = self.model.predict(landmarks)[0]
                
                # Display prediction
                cv2.putText(frame, f"Sign: {prediction}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Sign Language Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    predictor = SignPredictor()
    predictor.predict_sign()

if __name__ == "__main__":
    main()