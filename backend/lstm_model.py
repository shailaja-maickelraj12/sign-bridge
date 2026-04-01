import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os

class LSTMGestureRecognizer:
    def __init__(self, model_path='models/lstm_gesture.h5'):
        self.model = None
        self.sequence_length = 30  # frames per gesture
        self.num_landmarks = 63    # 21 landmarks × 3 coordinates
        self.sequence_buffer = []
        self.model_path = model_path
        self.gesture_mapping = {}
        self.load_model()
    
    def load_model(self):
        """Load pre-trained LSTM model if exists"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ LSTM model loaded from {self.model_path}")
            return True
        else:
            print(f"⚠️ LSTM model not found at {self.model_path}")
            print("   Training mode active - collect data first")
            return False
    
    def create_model(self, num_classes):
        """Create new LSTM model architecture"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequence_length, self.num_landmarks)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def add_frame(self, landmarks):
        """Add a frame to sequence buffer"""
        # Flatten landmarks: 21 points × (x,y,z) = 63 features
        flattened = []
        for point in landmarks:
            flattened.extend([point[0], point[1], point[2]])
        
        self.sequence_buffer.append(flattened)
        
        # Keep only last sequence_length frames
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
        
        # Return prediction if we have full sequence
        if len(self.sequence_buffer) == self.sequence_length and self.model:
            return self.predict()
        return None
    
    def predict(self):
        """Predict gesture from current buffer"""
        input_data = np.array(self.sequence_buffer).reshape(
            1, self.sequence_length, self.num_landmarks
        )
        predictions = self.model.predict(input_data, verbose=0)
        gesture_id = np.argmax(predictions[0])
        confidence = float(predictions[0][gesture_id])
        
        return gesture_id, confidence
    
    def reset(self):
        """Clear sequence buffer"""
        self.sequence_buffer = []
    
    def save_model(self, path=None):
        """Save trained model"""
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load_gesture_mapping(self, mapping_path='models/gesture_mapping.json'):
        """Load gesture ID to phrase mapping"""
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.gesture_mapping = json.load(f)
            return True
        return False