import json
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from lstm_model import LSTMGestureRecognizer

def load_training_data(data_dir='training_data'):
    """Load all collected sequences"""
    sequences = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Training data directory not found: {data_dir}")
        return np.array([]), np.array([])
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                sequences.append(data['sequence'])
                labels.append(data['gesture_id'])
    
    if not sequences:
        print("⚠️ No training data found. Please collect sequences first.")
        return np.array([]), np.array([])
    
    return np.array(sequences), np.array(labels)

def train():
    print("=" * 50)
    print("  Training LSTM Model for Self-Introduction")
    print("=" * 50)
    
    # Load data
    X, y = load_training_data()
    
    if len(X) == 0:
        print("\n📝 No training data found!")
        print("\nTo collect training data:")
        print("1. Run: python collect_data.py")
        print("2. Perform each gesture slowly for 30 frames")
        print("3. Save at least 30-50 sequences per gesture")
        return
    
    print(f"\n✅ Loaded {len(X)} sequences")
    print(f"📊 Gesture IDs: {np.unique(y)}")
    
    # One-hot encode labels
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📈 Training samples: {len(X_train)}")
    print(f"🧪 Test samples: {len(X_test)}")
    print(f"🎯 Number of gestures: {num_classes}")
    
    # Create and train model
    recognizer = LSTMGestureRecognizer()
    recognizer.create_model(num_classes)
    
    # Train
    history = recognizer.model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = recognizer.model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Test accuracy: {accuracy:.2%}")
    
    # Save model
    recognizer.save_model()
    
    # Save gesture mapping
    os.makedirs('models', exist_ok=True)
    with open('models/gesture_mapping.json', 'w') as f:
        mapping = {str(i): f"gesture_{i}" for i in range(num_classes)}
        json.dump(mapping, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: models/lstm_gesture.h5")
    print(f"📄 Mapping saved to: models/gesture_mapping.json")

if __name__ == "__main__":
    train()