from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os

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

    gesture = recognize_gesture(landmarks)

    if gesture is None:
        return jsonify({"gesture": None, "en": None, "ta": None})

    phrases = GESTURE_PHRASES.get(gesture, {"en": gesture, "ta": gesture})

    return jsonify({
        "gesture": gesture,
        "en": phrases["en"],
        "ta": phrases["ta"],
        "confidence": 0.95
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


if __name__ == "__main__":
    print("=" * 50)
    print("  SignBridge Backend Server")
    print("  Running at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
