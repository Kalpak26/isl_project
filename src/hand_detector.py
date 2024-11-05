# src/hand_detector.py
import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # detect two hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """Detect both hands in the image"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img
    
    def get_landmarks_array(self):
        """Get landmarks for both hands"""
        if not self.results.multi_hand_landmarks:
            return None
            
        # Initialize arrays for both hands
        landmarks_combined = []
        
        # Get handedness (left/right) information
        handedness = self.results.multi_handedness if self.results.multi_handedness else []
        
        # If only one hand is detected, pad with zeros for the other hand
        if len(self.results.multi_hand_landmarks) == 1:
            hand = self.results.multi_hand_landmarks[0]
            # Get the detected hand type (left/right)
            hand_type = handedness[0].classification[0].label
            
            # Extract landmarks for the detected hand
            hand_landmarks = []
            for landmark in hand.landmark:
                hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
            # Add zeros for the missing hand (63 values = 21 landmarks × 3 coordinates)
            empty_hand = [0.0] * 63
            
            # Order the hands correctly (left hand first, then right hand)
            if hand_type == 'Left':
                landmarks_combined = hand_landmarks + empty_hand
            else:
                landmarks_combined = empty_hand + hand_landmarks
                
        # If both hands are detected
        elif len(self.results.multi_hand_landmarks) == 2:
            # Get landmarks for both hands
            hand1_landmarks = []
            hand2_landmarks = []
            
            # Extract landmarks for both hands
            for idx, hand in enumerate(self.results.multi_hand_landmarks):
                current_hand = []
                for landmark in hand.landmark:
                    current_hand.extend([landmark.x, landmark.y, landmark.z])
                
                # Check handedness and assign to correct position
                if handedness[idx].classification[0].label == 'Left':
                    hand1_landmarks = current_hand
                else:
                    hand2_landmarks = current_hand
                    
            landmarks_combined = hand1_landmarks + hand2_landmarks
            
        return np.array(landmarks_combined) if landmarks_combined else None

# Test class for visualization
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.find_hands(img)
        landmarks = detector.get_landmarks_array()
        
        if detector.results.multi_hand_landmarks:
            # Directly count the number of hands detected
            num_hands = len(detector.results.multi_hand_landmarks)
            cv2.putText(img, f"Hands detected: {num_hands}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display handedness information
            if detector.results.multi_handedness:
                y_pos = 70  # Starting y position for hand labels
                for idx, hand_info in enumerate(detector.results.multi_handedness):
                    hand_type = hand_info.classification[0].label
                    cv2.putText(img, f"Hand {idx+1}: {hand_type}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 0, 0), 2)
                    y_pos += 40  # Increment y position for next label
        else:
            cv2.putText(img, "No hands detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ISL Hand Detection Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()