# src/test_model.py
import cv2
import numpy as np
import pickle
import os
from hand_detector import HandDetector
import time
from googletrans import Translator
from datetime import datetime

class SignPredictor:
    def __init__(self):
        self.detector = HandDetector()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load model
        model_path = os.path.join(self.project_root, 'models', 'random_forest_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.detected_signs = []
        self.current_sign = None
        self.last_prediction = None
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 6.0
        self.hold_count = 0
        self.hold_threshold = 2
        
        # Initialize translator
        self.translator = Translator()

        # Setup output directory and file
        self.output_dir = os.path.join(self.project_root, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f'translations_{timestamp}.txt')
        
        # Create file with header
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("ISL Translation Log\n")
            f.write("=" * 50 + "\n\n")

    def save_translation(self, word):
        try:
            # Basic translations
            basic_translations = {
                'hello': ('नमस्ते', 'नमस्कार'),
                'thank you': ('धन्यवाद', 'धन्यवाद'),
                'thanks': ('धन्यवाद', 'धन्यवाद'),
                'please': ('कृपया', 'कृपया'),
                'water': ('पानी', 'पाणी'),
                'yes': ('हाँ', 'होय'),
                'no': ('नहीं', 'नाही'),
                'help': ('मदद', 'मदत'),
                'goodbye': ('अलविदा', 'निरोप'),
                'bye': ('अलविदा', 'निरोप'),
                'food': ('खाना', 'जेवण'),
            }

            word = word.lower().strip()
            
            # Get translations
            if word in basic_translations:
                hindi, marathi = basic_translations[word]
            else:
                hindi = self.translator.translate(word, src='en', dest='hi').text
                marathi = self.translator.translate(word, src='en', dest='mr').text

            # Print to terminal
            print("\nNew Translation:")
            print(f"English: {word}")
            print(f"Hindi: {hindi}")
            print(f"Marathi: {marathi}")
            print("-" * 30)

            # Save to file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\nTime: {datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"English: {word}\n")
                f.write(f"Hindi: {hindi}\n")
                f.write(f"Marathi: {marathi}\n")
                f.write("-" * 30 + "\n")

            return hindi, marathi
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return word, word

    def predict_sign(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\nStarting Sign Language Detection...")
        print("Translations will appear here in the terminal")
        print("=" * 50)
        print(f"Saving translations to: {self.output_file}")

        last_word = ""
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = self.detector.find_hands(frame)
            landmarks = self.detector.get_landmarks_array()
            current_time = time.time()
            time_since_last = current_time - self.last_prediction_time

            is_ready = time_since_last >= self.prediction_cooldown
            status_color = (0, 255, 0) if is_ready else (0, 0, 255)
            status_text = "Ready" if is_ready else f"Wait {self.prediction_cooldown - time_since_last:.1f}s"

            if landmarks is not None and is_ready:
                if len(landmarks) == 63:
                    landmarks = np.concatenate([landmarks, np.zeros(63)])
                landmarks = landmarks.reshape(1, -1)
                prediction = self.model.predict(landmarks)[0]
                
                if prediction == self.last_prediction:
                    self.hold_count += 1
                    if self.hold_count >= self.hold_threshold:
                        self.detected_signs.append(prediction)
                        self.last_prediction_time = current_time
                        self.hold_count = 0
                else:
                    if not self.detected_signs or prediction != self.last_prediction:
                        self.detected_signs.append(prediction)
                        self.last_prediction_time = current_time
                    self.hold_count = 0
                
                self.last_prediction = prediction
                self.current_sign = prediction

            # Display section
            cv2.putText(frame, f"Current: {self.current_sign}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            if self.hold_count > 0:
                cv2.putText(frame, f"Hold for repeat: {self.hold_count}/{self.hold_threshold}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            if self.detected_signs:
                current_word = "".join(self.detected_signs)
                cv2.putText(frame, f"Word: {current_word}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Translate only when word changes
                if current_word != last_word:
                    self.save_translation(current_word)
                    last_word = current_word

            cv2.putText(frame, "Press: c-clear | q-quit | space-word break", 
                      (10, frame.shape[0] - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Sign Language Translation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.detected_signs = []
                self.last_prediction = None
                self.hold_count = 0
                last_word = ""
                print("\nCleared current word")
            elif key == ord(' '):
                self.detected_signs.append(' ')
                self.last_prediction = None
                self.hold_count = 0
                print("\nAdded space")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. Translations saved to: {self.output_file}")

if __name__ == "__main__":
    predictor = SignPredictor()
    predictor.predict_sign()