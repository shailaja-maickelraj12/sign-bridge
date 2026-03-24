# SignBridge 🤟
### Sign Language to Speech — Hackathon Project

---

## What This Does
Candidates who cannot speak use hand sign gestures in front of a webcam.
The system recognizes the gesture using AI, converts it to text and speech,
and displays it to the interviewer in English and Tamil.

---

## Project Structure
```
signbridge/
├── backend/
│   ├── app.py              ← Python Flask server (AI recognition)
│   └── requirements.txt    ← Python packages needed
├── frontend/
│   └── index.html          ← Open this in Chrome
├── START_SERVER.bat         ← Double-click to start (Windows)
└── README.md               ← This file
```

---

## Step 1: Install Python

1. Go to **https://www.python.org/downloads/**
2. Click "Download Python 3.11" (or newer)
3. Run the installer
4. **IMPORTANT:** Check the box ✅ **"Add Python to PATH"** before clicking Install
5. Click Install Now

To verify: Open Command Prompt and type:
```
python --version
```
You should see: `Python 3.11.x`

---

## Step 2: Start the Backend Server

**Option A — Easy way (double-click):**
- Double-click `START_SERVER.bat`
- It will install packages and start the server automatically

**Option B — Manual (in VS Code terminal):**
```bash
cd backend
pip install flask flask-cors numpy
python app.py
```

You should see:
```
==================================================
  SignBridge Backend Server
  Running at: http://localhost:5000
==================================================
```

---

## Step 3: Open the Frontend

1. Open VS Code
2. Navigate to `frontend/` folder
3. Right-click `index.html` → **Open with Live Server**
   (Install "Live Server" extension in VS Code if not installed)
4. OR just double-click `index.html` to open in Chrome

---

## Step 4: Use the App

1. Click **"Start Camera"** — allow camera permission
2. The green bar at top means backend is connected ✅
3. Show your hand gesture to the camera
4. Hold the gesture steady for ~1 second
5. The system speaks the phrase in English and Tamil
6. Click **"Interview Mode"** for the full-screen interviewer view

---

## Supported Gestures

| Gesture | Meaning | What it says |
|---------|---------|--------------|
| 👍 Thumbs up | Yes / OK | "Yes, I understand." |
| 👎 Thumbs down | No | "No, that is not correct." |
| ✋ Open hand | Please wait | "Please give me a moment to think." |
| ✊ Fist | Understand | "I understand the question." |
| ☝️ One finger | One moment | "I have one important point to make." |
| ✌️ Peace | Thank you | "Thank you for this opportunity." |
| 👌 OK sign | Perfect | "That sounds perfect to me." |
| 🤙 Call me | Contact | "Please contact me anytime." |
| 👋 Wave | Hello | "Hello! Thank you for having me." |
| ✌ Crossed fingers | Disagree | "I respectfully disagree." |
| 3 fingers | 3 years exp | "I have 3 years of experience." |
| 🤌 Pinch | Excellent | "That is an excellent point." |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML + CSS + JavaScript |
| Hand Tracking | MediaPipe Hands (Google) |
| Backend | Python + Flask |
| Gesture Recognition | NumPy landmark analysis |
| Text-to-Speech | Web Speech API (browser built-in) |
| Languages | English + Tamil + Hindi + Telugu |

---

## For the Hackathon Demo

1. Start backend server first
2. Open frontend in Chrome
3. Demo flow: Wave 👋 → Thumbs up 👍 → Peace ✌️ → Thank you
4. Switch to Interview Mode for the judge to read responses clearly
5. Show Tamil output running alongside English

---

## Team
Built for Tamil Nadu Hackathon — Assistive Technology Track
