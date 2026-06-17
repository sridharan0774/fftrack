import cv2
from tkinter import Tk, filedialog

# ---------------- Settings ----------------
MAX_WIDTH = 900
MAX_HEIGHT = 600

# ---------------- Select Video ----------------
Tk().withdraw()

video_path = filedialog.askopenfilename(
    title="Select Football Video",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
)

if not video_path:
    print("No video selected")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

# ---------------- Resize Function ----------------
def resize_for_display(frame):
    h, w = frame.shape[:2]

    scale_w = MAX_WIDTH / w
    scale_h = MAX_HEIGHT / h

    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    return resized, scale

# ---------------- Play Video ----------------
bbox = None
selected_frame = None

print("SPACE = Pause")
print("Q = Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    display_frame, scale = resize_for_display(frame)

    cv2.imshow("Football Video", display_frame)

    key = cv2.waitKey(delay) & 0xFF

    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if key == 32:  # SPACE
        selected_frame = frame.copy()

        paused_frame, scale = resize_for_display(frame)

        print("Video Paused")
        print("Draw box around player and press ENTER")

        small_bbox = cv2.selectROI(
            "Select Player",
            paused_frame,
            False
        )

        cv2.destroyWindow("Select Player")

        x, y, w, h = small_bbox

        bbox = (
            int(x / scale),
            int(y / scale),
            int(w / scale),
            int(h / scale)
        )

        break

cv2.destroyAllWindows()

# ---------------- Tracker ----------------
if bbox is None:
    print("No player selected")
    exit()

try:
    tracker = cv2.TrackerCSRT_create()
except:
    tracker = cv2.legacy.TrackerCSRT_create()

tracker.init(selected_frame, bbox)

# ---------------- Output ----------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "tracked_player.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("Tracking Started...")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    success, bbox = tracker.update(frame)

    if success:

        x, y, w, h = [int(v) for v in bbox]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Tracked Player",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Tracking Lost",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    out.write(frame)

    if frame_count % 100 == 0:
        print("Processed:", frame_count, "frames")

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")
print("Output saved as tracked_player.mp4")