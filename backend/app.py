from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room, emit
import numpy as np
import json
import os
import random
import string
from lstm_model import LSTMGestureRecognizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── GESTURE DEFINITIONS ──
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

# ── LIVE ROOM REGISTRY ──
rooms = {}

def generate_room_id():
    return "SB-" + "".join(random.choices(string.digits, k=4))

# ── LSTM ──
lstm_recognizer = LSTMGestureRecognizer()
USE_LSTM_FOR_SELF_INTRO = True

# ── GESTURE RECOGNITION ──
def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def finger_extended(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def recognize_gesture(landmarks):
    lm = landmarks
    thumb_up   = lm[4][1] < lm[3][1] < lm[2][1]
    thumb_down = lm[4][1] > lm[3][1] and lm[4][1] > lm[2][1]
    idx  = finger_extended(lm, 8,  6)
    mid  = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pink = finger_extended(lm, 20, 18)
    all_ext  = idx and mid and ring and pink
    all_curl = not idx and not mid and not ring and not pink

    if thumb_up and all_curl:    return "thumbs_up"
    if thumb_down and all_curl:  return "thumbs_down"
    if all_ext:
        return "wave" if lm[0][1] > lm[9][1] else "open_hand"
    if all_curl and not thumb_up and not thumb_down: return "fist"
    if idx and not mid and not ring and not pink:    return "pointing"
    if idx and mid and not ring and not pink:
        return "crossed" if lm[8][0] > lm[12][0] else "peace"
    if idx and mid and ring and not pink:            return "three_fingers"
    thumb_tip = lm[4]; index_tip = lm[8]
    if dist(thumb_tip, index_tip) < 0.07 and mid and ring and pink: return "ok_sign"
    if dist(thumb_tip, index_tip) < 0.05 and not mid and not ring:  return "pinch"
    if lm[4][0] < lm[3][0] and pink and not idx and not mid and not ring: return "call_me"
    return None


# ══════════════════════════════════════════════════════════
#  REST API ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "SignBridge backend running", "version": "2.1", "rooms": len(rooms)})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    if not data or "landmarks" not in data:
        return jsonify({"error": "No landmarks provided"}), 400
    landmarks = data["landmarks"]
    if len(landmarks) != 21:
        return jsonify({"error": "Expected 21 landmarks"}), 400

    if USE_LSTM_FOR_SELF_INTRO and lstm_recognizer.model:
        prediction = lstm_recognizer.add_frame(landmarks)
        if prediction:
            gesture_id, confidence = prediction
            if confidence > 0.7 and gesture_id in SELF_INTRO_GESTURES:
                phrases = SELF_INTRO_GESTURES[gesture_id]
                return jsonify({
                    "gesture": f"self_intro_{gesture_id}",
                    "en": phrases["en"], "ta": phrases["ta"],
                    "confidence": confidence, "method": "lstm"
                })

    gesture = recognize_gesture(landmarks)
    if gesture is None:
        lstm_recognizer.reset()
        return jsonify({"gesture": None, "en": None, "ta": None})

    phrases = GESTURE_PHRASES.get(gesture, {"en": gesture, "ta": gesture})
    return jsonify({
        "gesture": gesture,
        "en": phrases["en"], "ta": phrases["ta"],
        "confidence": 0.95, "method": "rule_based"
    })


@app.route("/api/gestures", methods=["GET"])
def list_gestures():
    return jsonify(GESTURE_PHRASES)


@app.route("/api/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text   = data.get("text", "")
    target = data.get("target", "ta")
    for gesture_id, phrases in GESTURE_PHRASES.items():
        if phrases["en"] == text:
            return jsonify({"translated": phrases.get(target, text)})
    return jsonify({"translated": text, "note": "exact translation not found"})


@app.route("/api/collect_sequence", methods=["POST"])
def collect_sequence():
    try:
        data = request.get_json()
        sequence   = data.get("sequence", [])
        gesture_id = data.get("gesture_id")
        from datetime import datetime
        os.makedirs("training_data", exist_ok=True)
        filename = f"training_data/gesture_{gesture_id}_{datetime.now().timestamp()}.json"
        with open(filename, "w") as f:
            json.dump({"sequence": sequence, "gesture_id": gesture_id,
                       "timestamp": datetime.now().isoformat()}, f)
        return jsonify({"status": "success", "frames": len(sequence)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── LIVE ROOM REST ENDPOINTS ──

@app.route("/api/room/create", methods=["POST"])
def create_room():
    data = request.get_json() or {}
    name = data.get("name", "Candidate")
    for _ in range(20):
        room_id = generate_room_id()
        if room_id not in rooms:
            break
    rooms[room_id] = {
        "candidate": None,
        "interviewer": None,
        "candidate_name": name,
        "created_at": __import__("time").time()
    }
    return jsonify({"room_id": room_id, "status": "created"})


@app.route("/api/room/<room_id>", methods=["GET"])
def room_info(room_id):
    if room_id not in rooms:
        return jsonify({"exists": False}), 404
    r = rooms[room_id]
    return jsonify({
        "exists": True,
        "room_id": room_id,
        "candidate_connected": r["candidate"] is not None,
        "interviewer_connected": r["interviewer"] is not None,
        "candidate_name": r.get("candidate_name", "Candidate")
    })


# ══════════════════════════════════════════════════════════
#  SOCKET.IO — LIVE ROOM SIGNALING
# ══════════════════════════════════════════════════════════

@socketio.on("join_room")
def on_join(data):
    room_id = data.get("room_id", "").upper()
    role    = data.get("role", "candidate")
    name    = data.get("name", role.capitalize())

    if room_id not in rooms:
        emit("error", {"msg": f"Room {room_id} does not exist."})
        return

    join_room(room_id)
    rooms[room_id][role] = request.sid

    emit("room_joined", {
        "room_id": room_id, "role": role, "name": name,
        "candidate_name": rooms[room_id].get("candidate_name", "Candidate")
    })

    emit("peer_joined", {"role": role, "name": name}, to=room_id, skip_sid=request.sid)


@socketio.on("leave_room_custom")
def on_leave(data):
    room_id = data.get("room_id", "").upper()
    role    = data.get("role", "candidate")
    if room_id in rooms:
        rooms[room_id][role] = None
        emit("peer_left", {"role": role}, to=room_id)
    leave_room(room_id)


# ── WebRTC signaling ──

@socketio.on("webrtc_offer")
def on_offer(data):
    room_id = data.get("room_id", "").upper()
    emit("webrtc_offer", {"sdp": data["sdp"]}, to=room_id, skip_sid=request.sid)


@socketio.on("webrtc_answer")
def on_answer(data):
    room_id = data.get("room_id", "").upper()
    emit("webrtc_answer", {"sdp": data["sdp"]}, to=room_id, skip_sid=request.sid)


@socketio.on("webrtc_ice")
def on_ice(data):
    room_id = data.get("room_id", "").upper()
    emit("webrtc_ice", {"candidate": data["candidate"]}, to=room_id, skip_sid=request.sid)


# ── Candidate gesture broadcast ──

@socketio.on("gesture_detected")
def on_gesture(data):
    """
    Candidate detected a gesture → broadcast text+Tamil to the room.
    Interviewer receives 'gesture_event' and displays + speaks it.
    """
    room_id = data.get("room_id", "").upper()
    emit("gesture_event", {
        "gesture":    data.get("gesture"),
        "en":         data.get("en"),
        "ta":         data.get("ta"),
        "confidence": data.get("confidence", 0.95)
    }, to=room_id, skip_sid=request.sid)


# ── NEW: Interviewer speak-back → candidate hears it ──

@socketio.on("interviewer_speech")
def on_interviewer_speech(data):
    """
    Interviewer types/speaks a message → relay to candidate.
    Candidate receives 'interviewer_speech_event' and:
      - displays the text on screen
      - if not textOnly: speaks it aloud via Web Speech API
    data = { room_id, text, name, textOnly? }
    """
    room_id = data.get("room_id", "").upper()
    emit("interviewer_speech_event", {
        "text":     data.get("text", ""),
        "name":     data.get("name", "Interviewer"),
        "textOnly": data.get("textOnly", False)
    }, to=room_id, skip_sid=request.sid)


# ── Disconnect cleanup ──

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    for room_id, info in list(rooms.items()):
        if info.get("candidate") == sid:
            info["candidate"] = None
            emit("peer_left", {"role": "candidate"}, to=room_id)
        if info.get("interviewer") == sid:
            info["interviewer"] = None
            emit("peer_left", {"role": "interviewer"}, to=room_id)


# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  SignBridge Backend Server  v2.1")
    print("  REST  →  http://localhost:5000")
    print("  WS    →  ws://localhost:5000")
    print("=" * 50)
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)