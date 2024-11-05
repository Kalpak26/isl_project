# src/test_model.py
import cv2
import numpy as np
import pickle
import os
from hand_detector import HandDetector
import time
from googletrans import Translator

class SignPredictor:
    def __init__(self):
        try:
            self.detector = HandDetector()
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            model_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.detected_signs = []
            self.current_sign = None
            self.last_prediction = None  # Store last prediction
            self.last_prediction_time = time.time()
            self.prediction_cooldown = 2.0
            self.translator = Translator()
            self.consecutive_same = 0  # Counter for consecutive same predictions
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def pad_landmarks(self, landmarks):
        if len(landmarks) == 63:
            return np.concatenate([landmarks, np.zeros(63)])
        return landmarks

    def translate_text(self, text):
        try:
            hindi = self.translator.translate(text, dest='hi').text
            marathi = self.translator.translate(text, dest='mr').text
            return hindi, marathi
        except Exception as e:
            print(f"Translation error: {e}")
            return text, text

    def predict_sign(self):
        try:
            print("Starting video capture...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            print("Camera opened successfully")
            ready_to_capture = True
            
            while True:
                success, frame = cap.read()
                if not success:
                    break

                try:
                    frame = self.detector.find_hands(frame)
                    landmarks = self.detector.get_landmarks_array()

                    current_time = time.time()
                    time_since_last = current_time - self.last_prediction_time

                    if time_since_last >= self.prediction_cooldown:
                        status_color = (0, 255, 0)
                        status_text = "Ready to capture"
                        ready_to_capture = True
                    else:
                        status_color = (0, 0, 255)
                        status_text = f"Wait {self.prediction_cooldown - time_since_last:.1f}s"
                        ready_to_capture = False

                    if landmarks is not None and ready_to_capture:
                        landmarks = self.pad_landmarks(landmarks)
                        landmarks = landmarks.reshape(1, -1)
                        prediction = self.model.predict(landmarks)[0]
                        
                        if time_since_last >= self.prediction_cooldown:
                            self.current_sign = prediction
                            
                            # Handle repeated letters
                            if prediction == self.last_prediction:
                                self.consecutive_same += 1
                                if self.consecutive_same >= 2:  # Need to see same letter twice to confirm repeat
                                    self.detected_signs.append(prediction)
                                    self.consecutive_same = 0
                            else:
                                if not self.detected_signs or prediction != self.detected_signs[-1]:
                                    self.detected_signs.append(prediction)
                                self.consecutive_same = 0
                            
                            self.last_prediction = prediction
                            self.last_prediction_time = current_time
                            ready_to_capture = False

                    # Display current sign and status
                    cv2.putText(frame, f"Current: {self.current_sign}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, status_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Show repetition counter when same sign is held
                    if self.consecutive_same > 0:
                        cv2.putText(frame, f"Hold for repeat ({self.consecutive_same}/2)", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                    # Display sentence
                    if self.detected_signs:
                        eng_text = "".join(self.detected_signs)
                        hindi_text, marathi_text = self.translate_text(eng_text)

                        y_pos = 150
                        for text, label in [(eng_text, "Word:"), 
                                          (hindi_text, "Hindi:"), 
                                          (marathi_text, "Marathi:")]:
                            cv2.putText(frame, f"{label} {text}", (10, y_pos), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            y_pos += 40

                    # Instructions
                    cv2.putText(frame, "Press 'c' to clear | 'q' to quit | Space for word break", 
                              (10, frame.shape[0] - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Sign Language Translation", frame)

                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.detected_signs = []
                    self.consecutive_same = 0
                    self.last_prediction = None
                elif key == ord(' '):
                    self.detected_signs.append(' ')
                    self.consecutive_same = 0
                    self.last_prediction = None

        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    predictor = SignPredictor()
    predictor.predict_sign()

if __name__ == "__main__":
    main()