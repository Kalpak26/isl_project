# src/test_model.py
import cv2
import numpy as np
import pickle
import os
from hand_detector import HandDetector
import time

class SignPredictor:
    def __init__(self):
        try:
            self.detector = HandDetector()
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Load the trained model
            model_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.detected_signs = []
            self.current_sign = None
            self.last_prediction_time = time.time()
            self.prediction_cooldown = 0.5
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def pad_landmarks(self, landmarks):
        """Pad landmarks to 126 features if only one hand is detected"""
        if len(landmarks) == 63:  # One hand detected
            # Pad with zeros for the second hand
            return np.concatenate([landmarks, np.zeros(63)])
        return landmarks

    def predict_sign(self):
        try:
            print("Starting video capture...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            print("Camera opened successfully")
            
            while True:
                success, frame = cap.read()
                if not success:
                    print("Error: Failed to read frame")
                    break

                try:
                    frame = self.detector.find_hands(frame)
                    landmarks = self.detector.get_landmarks_array()

                    current_time = time.time()

                    if landmarks is not None:
                        # Pad landmarks if needed and reshape
                        landmarks = self.pad_landmarks(landmarks)
                        landmarks = landmarks.reshape(1, -1)
                        
                        prediction = self.model.predict(landmarks)[0]
                        
                        if current_time - self.last_prediction_time >= self.prediction_cooldown:
                            self.current_sign = prediction
                            if not self.detected_signs or self.detected_signs[-1] != prediction:
                                self.detected_signs.append(prediction)
                                self.last_prediction_time = current_time

                    # Display current sign
                    cv2.putText(frame, f"Current: {self.current_sign}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Display detected signs string
                    detected_text = " ".join(self.detected_signs)
                    y_pos = 90
                    while detected_text:
                        if len(detected_text) > 40:
                            space_index = detected_text[:40].rfind(' ')
                            if space_index == -1:
                                space_index = 40
                            line = detected_text[:space_index]
                            detected_text = detected_text[space_index:].lstrip()
                        else:
                            line = detected_text
                            detected_text = ""
                        
                        cv2.putText(frame, line, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        y_pos += 30

                    # Instructions
                    cv2.putText(frame, "Press 'c' to clear | 'q' to quit", 
                               (10, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Sign Language Detection", frame)

                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit command received")
                    break
                elif key == ord('c'):
                    print("Clearing detected signs")
                    self.detected_signs = []

        except Exception as e:
            print(f"Error in prediction loop: {e}")
        
        finally:
            print("Releasing camera...")
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    try:
        predictor = SignPredictor()
        predictor.predict_sign()
    except Exception as e:
        print(f"Main error: {e}")

if __name__ == "__main__":
    main()