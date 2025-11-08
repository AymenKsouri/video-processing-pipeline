# 🎬 Video Indexation & Object Detection Pipeline

This project performs a **complete video analysis pipeline** using Python.  
It extracts keyframes from a video, detects objects in each frame using **YOLOv5**,  
transcribes the audio track using **OpenAI Whisper**, and finally generates an **index file**
containing metadata, detected objects, and transcription results.

---

## 🧠 Features

- 🎞️ **Keyframe Extraction** — using OpenCV  
- 🧍 **Object Detection** — with YOLOv5 (Ultralytics)  
- 🔊 **Audio Transcription** — via Whisper  
- 🗂️ **Automatic Index Generation** — merging transcript and object data  
- 🔎 **Search Functionality** — find any word with precise timestamps  

---

## 🧩 Project Structure

video-indexation/
│
├── indexation.py        # Main script (pipeline)
├── README.md            # Project documentation
├── requirements.txt     # List of dependencies
├── .gitignore           # Git ignore rules
├── video.mp4            # Input video (not uploaded to GitHub)
├── frames/              # Extracted keyframes (auto-generated)
├── video_index.txt      # Final index (auto-generated)
└── ffmpeg/              # FFmpeg binaries (local use only)

---

## 🚀 How to Use

### 🧰 1. Clone the Repository

git clone https://github.com/AymenKsouri/video-indexation.git
cd video-indexation

---

### 🧱 2. Create & Activate a Virtual Environment

**Windows:**
python -m venv venv
venv\Scripts\activate

**Mac/Linux:**
python3 -m venv venv
source venv/bin/activate

---

### 📦 3. Install Dependencies

pip install -r requirements.txt

---

### 🎥 4. Add Your Video

Place your video in the same directory as `indexation.py`  
and rename it **video.mp4** (or modify the script to match your file name).

---

### ▶️ 5. Run the Script

python indexation.py

The script will:
- Extract keyframes into the `/frames` folder  
- Detect objects using YOLOv5  
- Transcribe audio with Whisper  
- Generate a `video_index.txt` file containing:
  - Video metadata  
  - Transcribed text  
  - Object detection results  

---

## 📄 Example Output

======= METADATA ===========
Frame Width: 1920
Frame Height: 1080
FPS: 30

===== TRANSCRIPTION =====
Hello everyone, welcome to the demo...

===== OBJECTS DETECTED =====
Frame: frame_0001.jpg, Object: person, Confidence: 0.88, Coordinates: (130.5, 220.1) - (450.3, 980.7)

---

## ⚙️ Requirements & Tools

| Tool | Purpose |
|------|----------|
| **Python 3.9+** | Base environment |
| **OpenCV** | Frame extraction |
| **MoviePy** | Video/audio processing |
| **Ultralytics YOLOv5** | Object detection |
| **OpenAI Whisper** | Speech-to-text transcription |
| **FFmpeg** | Required for MoviePy operations |

---

## ⚠️ Notes

- 🧱 Don’t upload heavy files like `.mp4`, `.wav`, or `.pt` to GitHub.  
- 🧩 Ensure **FFmpeg** is installed and added to your system’s PATH.  
- 📦 The YOLOv5 model (`yolov5s.pt`) is automatically downloaded the first time it runs.

---

## 👤 Author

**Aymen Ksouri**  
💻 Computer Science Student | AI & Vision Enthusiast  
🌍 [GitHub Profile](https://github.com/AymenKsouri)

---

⭐ *If you find this project helpful, consider giving it a star on GitHub!*
