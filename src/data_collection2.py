# src/data_collection2.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from hand_detector import HandDetector

class SignDataCollector:
    def __init__(self):
        self.detector = HandDetector()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.columns = []
        for hand in ['left', 'right']:
            for point in range(21):
                for coord in ['x', 'y', 'z']:
                    self.columns.append(f'{hand}_point{point}_{coord}')

    def collect_sign_data(self, sign_name, delay=0.5):
        data_path = os.path.join(self.project_root, 'data', 'raw', sign_name)
        os.makedirs(data_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        collected_samples = 0
        last_capture_time = time.time()

        print(f"\nCollecting data for: {sign_name}")
        print("Press 'q' to stop collecting")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = self.detector.find_hands(frame)
            landmarks = self.detector.get_landmarks_array()
            current_time = time.time()

            cv2.putText(frame, f"Sign: {sign_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {collected_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.detector.results.multi_hand_landmarks:
                num_hands = len(self.detector.results.multi_hand_landmarks)
                cv2.putText(frame, f"Hands detected: {num_hands}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if current_time - last_capture_time >= delay and landmarks is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(data_path, f'{sign_name}_{timestamp}.csv')
                    
                    df = pd.DataFrame([landmarks], columns=self.columns)
                    df['sign'] = sign_name
                    df.to_csv(filename, index=False)
                    
                    collected_samples += 1
                    last_capture_time = current_time
                    print(f"Samples collected: {collected_samples}", end='\r')
            else:
                cv2.putText(frame, "No hands detected", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCollection complete. Total samples: {collected_samples}")
        return collected_samples

def main():
    sign_name = input("Enter the sign to collect: ")
    collector = SignDataCollector()
    collector.collect_sign_data(sign_name)

if __name__ == "__main__":
    main()