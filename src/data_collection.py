# src/data_collecton.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime
from hand_detector import HandDetector

class SignDataCollector:
    def __init__(self):
        self.detector = HandDetector()
        # Get the project root directory (parent of src)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def collect_sign_data(self, sign_name, num_samples=30):
        # Create path using project root
        data_path = os.path.join(self.project_root, 'data', 'raw', sign_name)
        os.makedirs(data_path, exist_ok=True)

        print(f"Saving data to: {data_path}")

        cap = cv2.VideoCapture(0)
        collected_samples = 0

        print(f"\n=== Collecting data for sign: {sign_name} ===")
        print("Instructions:")
        print("1. Press SPACE to capture a sample")
        print("2. Press Q to quit")
        print(f"Target: {num_samples} samples\n")

        while collected_samples < num_samples:
            success, frame = cap.read()
            if not success:
                break

            # Process frame
            frame = self.detector.find_hands(frame)
            landmarks = self.detector.get_landmarks_array()

            # Display info
            cv2.putText(frame, f"Sign: {sign_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {collected_samples}/{num_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if self.detector.results.multi_hand_landmarks:
                num_hands = len(self.detector.results.multi_hand_landmarks)
                cv2.putText(frame, f"Hands detected: {num_hands}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(1)

            if key == 32:  # SPACE key
                if landmarks is not None:
                    # Save landmarks
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(data_path, f'{sign_name}_{timestamp}.csv')
                    
                    # Convert landmarks to DataFrame
                    df = pd.DataFrame([landmarks])
                    df['sign'] = sign_name
                    df.to_csv(filename, index=False)
                    
                    collected_samples += 1
                    print(f"Collected sample {collected_samples}/{num_samples}")
                else:
                    print("No hands detected! Please show hands clearly.")
            
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCollection complete for {sign_name}!")
        return collected_samples

def main():
    # List of signs to collect
    signs = [
        'hello',
        'thank_you',
        'water',
        'no'
    ]

    collector = SignDataCollector()

    for sign in signs:
        input(f"\nPress Enter to start collecting data for '{sign}'...")
        samples = collector.collect_sign_data(sign)
        print(f"Collected {samples} samples for {sign}")
        print(f"Data saved in data/raw/{sign}/")

if __name__ == "__main__":
    main()