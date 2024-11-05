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

            self.translations = {
                'hello': {'hindi': 'नमस्ते', 'marathi': 'नमस्कार'},
                'thank_you': {'hindi': 'धन्यवाद', 'marathi': 'धन्यवाद'},
                'please': {'hindi': 'कृपया', 'marathi': 'कृपया'},
                'water': {'hindi': 'पानी', 'marathi': 'पाणी'},
            }
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def pad_landmarks(self, landmarks):
        if len(landmarks) == 63:
            return np.concatenate([landmarks, np.zeros(63)])
        return landmarks

    def get_translation(self, sign, language='hindi'):
        if sign in self.translations:
            return self.translations[sign][language]
        return sign

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
                    break

                try:
                    frame = self.detector.find_hands(frame)
                    landmarks = self.detector.get_landmarks_array()

                    current_time = time.time()

                    if landmarks is not None:
                        landmarks = self.pad_landmarks(landmarks)
                        landmarks = landmarks.reshape(1, -1)
                        
                        prediction = self.model.predict(landmarks)[0]
                        
                        if current_time - self.last_prediction_time >= self.prediction_cooldown:
                            self.current_sign = prediction
                            if not self.detected_signs or self.detected_signs[-1] != prediction:
                                self.detected_signs.append(prediction)
                                self.last_prediction_time = current_time

                    # Display current sign with translations
                    if self.current_sign:
                        hindi_trans = self.get_translation(self.current_sign, 'hindi')
                        marathi_trans = self.get_translation(self.current_sign, 'marathi')
                        
                        cv2.putText(frame, f"Sign: {self.current_sign}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Hindi: {hindi_trans}", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Marathi: {marathi_trans}", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display sentence
                    if self.detected_signs:
                        # English sentence
                        eng_text = " ".join(self.detected_signs)
                        # Hindi sentence
                        hindi_text = " ".join([self.get_translation(sign, 'hindi') 
                                             for sign in self.detected_signs])
                        # Marathi sentence
                        marathi_text = " ".join([self.get_translation(sign, 'marathi') 
                                               for sign in self.detected_signs])

                        y_pos = 150
                        # Display all sentences
                        for text in [eng_text, hindi_text, marathi_text]:
                            while text:
                                if len(text) > 40:
                                    space_index = text[:40].rfind(' ')
                                    if space_index == -1:
                                        space_index = 40
                                    line = text[:space_index]
                                    text = text[space_index:].lstrip()
                                else:
                                    line = text
                                    text = ""
                                
                                cv2.putText(frame, line, (10, y_pos), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                y_pos += 30

                    # Instructions
                    cv2.putText(frame, "Press 'c' to clear | 'q' to quit", 
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

        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    predictor = SignPredictor()
    predictor.predict_sign()

if __name__ == "__main__":
    main()