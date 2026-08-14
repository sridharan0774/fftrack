1. AI Video Analysis: Uses YOLOv8 and OpenCV to automatically detect and track football players and the ball in uploaded match videos.
2. Jersey Number Recognition: Applies EasyOCR and image preprocessing to identify individual player jersey numbers on the field.
3. Automated Commentary & Stats: Generates live text commentary based on match activity and calculates player performance metrics like distance covered and field visibility time.
4. Interactive GUI: Features a user-friendly desktop application built with Tkinter for uploading videos, watching live analysis, and exporting reports.

-  **YOLOv8 Player & Ball Detection**: Real-time object detection and multi-object tracking for players and the football using `yolov8n.pt`.
- **Jersey Number OCR Recognition**: Automatically crops player shirt areas, applies advanced image preprocessing (contrast enhancement, Gaussian blur, adaptive thresholding), and reads jersey numbers using `EasyOCR`.
-  **Consensus Voting System**: Prevents false readings by confirming jersey numbers over multiple frames before permanent assignment.
-  **Live Match Commentary**: Generates real-time textual match events based on player proximity to the ball and overall pitch activity.
-  **Performance Analytics & CSV Export**: Calculates player distance covered (in pixels), field visibility time (in seconds), and rates player activity scores automatically exported to CSV format.
- **Interactive Desktop GUI**: Clean Tkinter user interface with video uploader, embedded live commentary viewer, status monitor, and video playback player.

---

## 🛠️ Tech Stack & Dependencies

- **Python 3.8+**
- **YOLOv8 (`ultralytics`)**: Object detection & tracking
- **OpenCV (`cv2`)**: Video processing & computer vision overlays
- **EasyOCR**: Optical Character Recognition for jersey numbers
- **Pandas & NumPy**: Data processing and statistical report creation
- **Tkinter**: Graphical User Interface (GUI)

---

## 📁 Project Structure

```text
fftrack/
│
├── app2.py                 # Main application (GUI + Tracking + OCR + Commentary)
├── app.py                  # Legacy alternative script
├── player.py               # Manual single-player ROI tracker script (CSRT)
├── yolov8n.pt              # Pre-trained YOLOv8 object detection model weights
├── requirements.txt        # Python package dependencies
│
├── output/ (Generated after analysis)
│   ├── jersey_output.avi        # Processed video with bounding box overlays
│   ├── live_commentary.txt      # Text file containing match commentary log
│   └── jersey_player_stats.csv  # CSV file containing visibility, distance & rating metrics
└── README.md               # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone https://github.com/your-username/fftrack.git
cd fftrack
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run the Application

Launch the main application by executing:

```bash
python app2.py
```

### Step-by-Step Usage:
1. Click **Upload Video** and select a match video (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. Click **Analyze Video**.
3. Watch live commentary appear in the embedded text window as the AI processes the match frames.
4. Once completed, click:
   - **Play Output Video**: View the annotated output video with player bounding boxes and jersey IDs.
   - **Open Player Stats**: Open the generated `jersey_player_stats.csv` report.
   - **Open Commentary File**: View the full log in `live_commentary.txt`.

---

## 📊 Sample Output Metrics

| Jersey ID | Visible Time Seconds | Movement Pixels | AI Rating |
| :--- | :--- | :--- | :--- |
| **7** | 42.50 | 1250.40 | 6.04 |
| **10** | 58.10 | 1980.20 | 6.65 |
| **T3** *(Unmapped Tracker)* | 15.20 | 320.10 | 5.27 |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
