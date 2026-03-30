from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
from lstm_model import LSTMGestureRecognizer

app = Flask(__name__)
CORS(app)

# ── GESTURE DEFINITIONS ──
# Maps gesture name → English + Tamil phrases
GESTURE_PHRASES = {
    "thumbs_up":    {"en": "Yes, I understand.",                          "ta": "ஆம், புரிகிறது."},
    "thumbs_down":  {"en": "No, that is not correct.",                    "ta": "இல்லை, அது சரியில்லை."},
    "open_hand":    {"en": "Please give me a moment to think.",           "ta": "சிறிது நேரம் சிந்திக்க அனுமதியுங்கள்."},
    "fist":         {"en": "I understand the question.",                  "ta": "கேள்வி புரிகிறது."},
    "pointing":     {"en": "I have one important point to make.",         "ta": "ஒரு முக்கியமான கருத்து சொல்ல வேண்டும்."},
    "peace":        {"en": "Thank you very much for this opportunity.",   "ta": "இந்த வாய்ப்பிற்கு மிக்க நன்றி."},
    "ok_sign":      {"en": "That sounds perfect to me.",                  "ta": "அது எனக்கு சரியாக இருக்கிறது."},
    "call_me":      {"en": "Please feel free to contact me anytime.",     "ta": "எந்த நேரத்திலும் தொடர்பு கொள்ளலாம்."},
    "wave":         {"en": "Hello! Thank you for having me today.",       "ta": "வணக்கம்! இன்று என்னை அழைத்தமைக்கு நன்றி."},
    "three_fingers":{"en": "I have 3 years of experience in this field.", "ta": "இந்த துறையில் எனக்கு 3 வருட அனுபவம் உள்ளது."},
    "pinch":        {"en": "That is an excellent point.",                 "ta": "அது சிறந்த கருத்து."},
    "crossed":      {"en": "I respectfully disagree with that point.",    "ta": "அந்த கருத்தை மரியாதையுடன் மறுக்கிறேன்."},
}

# Initialize LSTM recognizer
lstm_recognizer = LSTMGestureRecognizer()

# Flag to use LSTM for specific gestures (self-introduction)
USE_LSTM_FOR_SELF_INTRO = True

# Self-introduction script gestures (15 gestures)
SELF_INTRO_GESTURES = {
    0: {"en": "Hello, my name is...", "ta": "வணக்கம், என் பெயர்..."},
    1: {"en": "I am from Tamil Nadu", "ta": "நான் தமிழ்நாட்டைச் சேர்ந்தவன்"},
    2: {"en": "I completed my Bachelor's degree", "ta": "நான் இளங்கலை பட்டம் முடித்துள்ளேன்"},
    3: {"en": "I have skills in programming", "ta": "எனக்கு நிரலாக்கத்தில் திறமை உள்ளது"},
    4: {"en": "I am a quick learner", "ta": "நான் விரைவாக கற்றுக்கொள்பவன்"},
    5: {"en": "I work well in teams", "ta": "நான் குழுவில் நன்றாக வேலை செய்வேன்"},
    6: {"en": "My strength is problem-solving", "ta": "எனது பலம் சிக்கல் தீர்வு"},
    7: {"en": "I have completed several projects", "ta": "நான் பல திட்டங்களை முடித்துள்ளேன்"},
    8: {"en": "I am passionate about technology", "ta": "தொழில்நுட்பத்தில் எனக்கு ஆர்வம்"},
    9: {"en": "I am looking for growth opportunities", "ta": "நான் வளர்ச்சி வாய்ப்புகளை தேடுகிறேன்"},
    10: {"en": "I can contribute to your team", "ta": "உங்கள் குழுவிற்கு பங்களிக்க முடியும்"},
    11: {"en": "I am dedicated and hardworking", "ta": "நான் அர்ப்பணிப்பு மற்றும் கடின உழைப்பாளி"},
    12: {"en": "I adapt quickly to new environments", "ta": "புதிய சூழல்களுக்கு விரைவாக மாற்றியமைப்பேன்"},
    13: {"en": "Thank you for this opportunity", "ta": "இந்த வாய்ப்பிற்கு நன்றி"},
    14: {"en": "I look forward to working with you", "ta": "உங்களுடன் பணியாற்ற ஆவலாக உள்ளேன்"}
}

# ── LANDMARK-BASED GESTURE RECOGNITION ──
def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def finger_extended(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def recognize_gesture(landmarks):
    """
    landmarks: list of 21 points, each [x, y, z]
    Returns gesture name string or None
    """
    lm = landmarks

    thumb_up   = lm[4][1] < lm[3][1] < lm[2][1]
    thumb_down = lm[4][1] > lm[3][1] and lm[4][1] > lm[2][1]

    idx  = finger_extended(lm, 8,  6)
    mid  = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pink = finger_extended(lm, 20, 18)

    all_ext  = idx and mid and ring and pink
    all_curl = not idx and not mid and not ring and not pink

    # Thumbs up
    if thumb_up and all_curl:
        return "thumbs_up"

    # Thumbs down
    if thumb_down and all_curl:
        return "thumbs_down"

    # Open hand / wave
    if all_ext:
        if lm[0][1] > lm[9][1]:
            return "wave"
        return "open_hand"

    # Fist
    if all_curl and not thumb_up and not thumb_down:
        return "fist"

    # Pointing (index only)
    if idx and not mid and not ring and not pink:
        return "pointing"

    # Peace / V sign
    if idx and mid and not ring and not pink:
        if lm[8][0] > lm[12][0]:
            return "crossed"
        return "peace"

    # Three fingers
    if idx and mid and ring and not pink:
        return "three_fingers"

    # OK sign (thumb + index circle)
    thumb_tip = lm[4]
    index_tip = lm[8]
    if dist(thumb_tip, index_tip) < 0.07 and mid and ring and pink:
        return "ok_sign"

    # Pinch
    if dist(thumb_tip, index_tip) < 0.05 and not mid and not ring:
        return "pinch"

    # Call me (thumb + pinky extended)
    thumb_ext = lm[4][0] < lm[3][0]
    if thumb_ext and pink and not idx and not mid and not ring:
        return "call_me"

    return None


# ── API ROUTES ──

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "SignBridge backend running", "version": "1.0"})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """
    Receives hand landmarks from frontend MediaPipe,
    recognizes gesture, returns English + Tamil text.

    Body: { "landmarks": [[x,y,z], ...21 points] }
    """
    data = request.get_json()
    if not data or "landmarks" not in data:
        return jsonify({"error": "No landmarks provided"}), 400

    landmarks = data["landmarks"]
    if len(landmarks) != 21:
        return jsonify({"error": "Expected 21 landmarks"}), 400

    # Try LSTM first for self-introduction gestures
    if USE_LSTM_FOR_SELF_INTRO and lstm_recognizer.model:
        prediction = lstm_recognizer.add_frame(landmarks)
        if prediction:
            gesture_id, confidence = prediction
            if confidence > 0.7 and gesture_id in SELF_INTRO_GESTURES:
                phrases = SELF_INTRO_GESTURES[gesture_id]
                return jsonify({
                    "gesture": f"self_intro_{gesture_id}",
                    "en": phrases["en"],
                    "ta": phrases["ta"],
                    "confidence": confidence,
                    "method": "lstm"
                })
    
    # Fallback to rule-based recognition for other gestures
    gesture = recognize_gesture(landmarks)
    
    if gesture is None:
        # Reset LSTM buffer if no gesture detected
        lstm_recognizer.reset()
        return jsonify({"gesture": None, "en": None, "ta": None})
    
    phrases = GESTURE_PHRASES.get(gesture, {"en": gesture, "ta": gesture})
    
    return jsonify({
        "gesture": gesture,
        "en": phrases["en"],
        "ta": phrases["ta"],
        "confidence": 0.95,
        "method": "rule_based"
    })


@app.route("/api/gestures", methods=["GET"])
def list_gestures():
    """Returns all supported gestures and their phrases."""
    return jsonify(GESTURE_PHRASES)


@app.route("/api/translate", methods=["POST"])
def translate():
    """
    Translates English text to requested language.
    Body: { "text": "...", "target": "ta" | "hi" | "te" }
    Uses simple dictionary for offline hackathon use.
    """
    data = request.get_json()
    text = data.get("text", "")
    target = data.get("target", "ta")

    # Find matching phrase across all gestures
    for gesture_id, phrases in GESTURE_PHRASES.items():
        if phrases["en"] == text:
            return jsonify({"translated": phrases.get(target, text)})

    # Fallback: return as-is if not found
    return jsonify({"translated": text, "note": "exact translation not found"})

@app.route("/api/collect_sequence", methods=["POST"])
def collect_sequence():
    """Collect gesture sequence for LSTM training"""
    try:
        data = request.get_json()
        sequence = data.get("sequence", [])
        gesture_id = data.get("gesture_id")
        
        # Save sequence to file
        from datetime import datetime
        
        os.makedirs('training_data', exist_ok=True)
        filename = f'training_data/gesture_{gesture_id}_{datetime.now().timestamp()}.json'
        
        with open(filename, 'w') as f:
            json.dump({
                'sequence': sequence,
                'gesture_id': gesture_id,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        return jsonify({"status": "success", "frames": len(sequence)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("  SignBridge Backend Server")
    print("  Running at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
