import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.sequence = []
        self.is_recording = False
        self.current_gesture = None
        self.sequence_length = 30
        self.collected_count = 0
        
    def start_recording(self, gesture_id):
        """Start recording a gesture sequence"""
        self.current_gesture = gesture_id
        self.sequence = []
        self.is_recording = True
        print(f"\n🎥 Recording gesture {gesture_id}...")
        print(f"   Perform the gesture slowly and hold steady")
        print(f"   Need {self.sequence_length} frames...")
        
    def process_frame(self, frame):
        """Process frame and add to sequence if recording"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if self.is_recording:
                self.sequence.append(landmarks)
                
                if len(self.sequence) == self.sequence_length:
                    self.save_sequence()
                    self.is_recording = False
                    self.collected_count += 1
                    print(f"✅ Saved sequence {self.collected_count} for gesture {self.current_gesture}")
                    return True
        return False
    
    def save_sequence(self):
        """Save sequence to file"""
        os.makedirs('training_data', exist_ok=True)
        filename = f'training_data/gesture_{self.current_gesture}_{datetime.now().timestamp()}.json'
        
        data = {
            'sequence': self.sequence,
            'gesture_id': self.current_gesture,
            'timestamp': datetime.now().isoformat(),
            'frames': len(self.sequence)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)

def main():
    collector = DataCollector()
    
    # Self-introduction gestures
    gestures = {
        '0': 'Hello / Introduction',
        '1': 'From Tamil Nadu',
        '2': 'Bachelor\'s degree',
        '3': 'Programming skills',
        '4': 'Quick learner',
        '5': 'Team player',
        '6': 'Problem solving',
        '7': 'Completed projects',
        '8': 'Passionate about tech',
        '9': 'Growth opportunities',
        '10': 'Contribute to team',
        '11': 'Dedicated worker',
        '12': 'Adapt quickly',
        '13': 'Thank you',
        '14': 'Looking forward'
    }
    
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("  LSTM Training Data Collector - Self Introduction")
    print("=" * 60)
    print("\n📝 Instructions:")
    print("  1. Press a number key (0-14) to start recording that gesture")
    print("  2. Perform the gesture slowly for about 2-3 seconds")
    print("  3. Keep your hand steady - system will capture 30 frames")
    print("  4. Repeat each gesture 20-30 times for good results")
    print("  5. Press 'q' to quit")
    print("\n🎯 Gesture mapping:")
    for key, name in gestures.items():
        print(f"   {key}: {name}")
    print("\n" + "=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Process frame
        completed = collector.process_frame(frame)
        
        # Display status
        if collector.is_recording:
            progress = len(collector.sequence) / collector.sequence_length
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            cv2.putText(frame, f"Recording Gesture {collector.current_gesture}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"[{bar}] {int(progress*100)}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Data Collection - SignBridge LSTM', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key).isdigit():
            gesture_id = int(chr(key))
            if 0 <= gesture_id <= 14:
                collector.start_recording(gesture_id)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Collection complete! Total sequences: {collector.collected_count}")
    print("🚀 Now run: python train_lstm.py")

if __name__ == "__main__":
    main()