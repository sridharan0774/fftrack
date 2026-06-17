import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import re
import math
import pandas as pd
import easyocr
from ultralytics import YOLO

# ---------------- Models ----------------
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(["en"], gpu=False)

# ---------------- Files ----------------
video_path = ""
output_video = "jersey_output.avi"
commentary_file = "live_commentary.txt"
stats_file = "jersey_player_stats.csv"

# ---------------- Data ----------------
tracker_to_jersey = {}
jersey_votes = {}
player_last_pos = {}
player_distance = {}
player_seen = {}
live_comments = []


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def read_jersey_number(player_crop):
    try:
        if player_crop is None or player_crop.size == 0:
            return None

        h, w, _ = player_crop.shape

        # focus chest/back area
        crop = player_crop[
            int(h * 0.15):int(h * 0.80),
            int(w * 0.10):int(w * 0.90)
        ]

        if crop.size == 0:
            return None

        crop = cv2.resize(crop, None, fx=3, fy=3)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        results = reader.readtext(
            thresh,
            allowlist="0123456789",
            detail=1,
            paragraph=False
        )

        for _, text, conf in results:
            text = re.sub(r"[^0-9]", "", text)

            if text.isdigit() and 1 <= len(text) <= 2 and conf > 0.20:
                return text

    except Exception:
        return None

    return None


def add_comment(text):
    live_comments.append(text)
    txt_commentary.insert(tk.END, text + "\n")
    txt_commentary.see(tk.END)
    root.update()


def upload_video():
    global video_path

    video_path = filedialog.askopenfilename(
        title="Select Football Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if video_path:
        lbl_video.config(text=os.path.basename(video_path))
        lbl_status.config(text="Video uploaded")


def analyze_video():
    if not video_path:
        messagebox.showerror("Error", "Upload video first")
        return

    tracker_to_jersey.clear()
    jersey_votes.clear()
    player_last_pos.clear()
    player_distance.clear()
    player_seen.clear()
    live_comments.clear()
    txt_commentary.delete("1.0", tk.END)

    lbl_status.config(text="Analyzing video...")
    root.update()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open input video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 25

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    frame_no = 0
    last_comment_frame = 0
    total_players = 0
    ball_frames = 0

    add_comment("Match analysis started.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1

        results = model.track(
            frame,
            persist=True,
            conf=0.35,
            verbose=False
        )

        players_now = 0
        ball_found = False
        visible_jerseys = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if label == "person":
                    players_now += 1

                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    player_crop = frame[y1:y2, x1:x2]

                    # OCR every 10 frames only
                    if frame_no % 10 == 0:
                        jersey = read_jersey_number(player_crop)

                        if jersey:
                            if track_id not in jersey_votes:
                                jersey_votes[track_id] = {}

                            jersey_votes[track_id][jersey] = (
                                jersey_votes[track_id].get(jersey, 0) + 1
                            )

                            best_jersey = max(
                                jersey_votes[track_id],
                                key=jersey_votes[track_id].get
                            )

                            # Assign jersey only after same number appears twice
                            if jersey_votes[track_id][best_jersey] >= 2:
                                if track_id not in tracker_to_jersey:
                                    add_comment(
                                        f"Jersey number {best_jersey} detected and assigned."
                                    )

                                tracker_to_jersey[track_id] = best_jersey

                    jersey_id = tracker_to_jersey.get(track_id, f"T{track_id}")

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if jersey_id not in player_last_pos:
                        player_last_pos[jersey_id] = (cx, cy)
                        player_distance[jersey_id] = 0
                        player_seen[jersey_id] = 0
                    else:
                        move = dist(player_last_pos[jersey_id], (cx, cy))

                        # ignore huge wrong tracker jumps
                        if move < 200:
                            player_distance[jersey_id] += move

                        player_last_pos[jersey_id] = (cx, cy)

                    player_seen[jersey_id] += 1
                    visible_jerseys.append(jersey_id)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Player #{jersey_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                elif label == "sports ball":
                    ball_found = True
                    ball_frames += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Ball",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        total_players += players_now

        # ---------------- Live Commentary ----------------
        if frame_no - last_comment_frame >= fps * 3:
            sec = frame_no // fps

            known_jerseys = [
                j for j in visible_jerseys
                if not str(j).startswith("T")
            ]

            if ball_found and known_jerseys:
                add_comment(
                    f"At {sec}s, the ball is active near Player #{known_jerseys[0]}."
                )

            elif ball_found:
                add_comment(
                    f"At {sec}s, the ball is visible and play is continuing."
                )

            elif known_jerseys:
                add_comment(
                    f"At {sec}s, Player #{known_jerseys[0]} is moving on the field."
                )

            elif players_now >= 5:
                add_comment(
                    f"At {sec}s, multiple players are visible in active play."
                )

            elif players_now > 0:
                add_comment(
                    f"At {sec}s, player movement is detected."
                )

            last_comment_frame = frame_no

        # Overlay
        cv2.putText(
            frame,
            f"Players: {players_now}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        if ball_found:
            cv2.putText(
                frame,
                "Ball Detected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        out.write(frame)

    cap.release()
    out.release()

    avg_players = total_players / frame_no if frame_no > 0 else 0

    # ---------------- Save Commentary ----------------
    with open(commentary_file, "w", encoding="utf-8") as f:
        f.write("AI Live Match Commentary\n")
        f.write("========================\n\n")

        for comment in live_comments:
            f.write(comment + "\n")

    # ---------------- Save Stats ----------------
    stats = []

    for jersey_id in player_distance:
        visible_time = player_seen[jersey_id] / fps
        movement = player_distance[jersey_id]
        rating = min(10, 5 + movement / 1200)

        stats.append({
            "Jersey ID": jersey_id,
            "Visible Time Seconds": round(visible_time, 2),
            "Movement Pixels": round(movement, 2),
            "AI Rating": round(rating, 2)
        })

    pd.DataFrame(stats).to_csv(stats_file, index=False)

    lbl_status.config(text="Analysis completed")
    add_comment("Match analysis completed.")

    messagebox.showinfo(
        "Completed",
        f"Analysis completed!\n\n"
        f"Total Frames: {frame_no}\n"
        f"Average Players: {avg_players:.2f}\n"
        f"Ball Detected Frames: {ball_frames}\n\n"
        f"Files created:\n"
        f"{output_video}\n"
        f"{commentary_file}\n"
        f"{stats_file}"
    )


def play_output():
    if not os.path.exists(output_video):
        messagebox.showerror("Error", "Analyze video first")
        return

    cap = cv2.VideoCapture(output_video)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open output video")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]
        max_w = 950
        scale = max_w / w
        new_h = int(h * scale)

        frame = cv2.resize(frame, (max_w, new_h))

        cv2.imshow("Output Video - Press Q to close", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def open_stats():
    if os.path.exists(stats_file):
        os.startfile(stats_file)
    else:
        messagebox.showerror("Error", "Analyze video first")


def open_commentary():
    if os.path.exists(commentary_file):
        os.startfile(commentary_file)
    else:
        messagebox.showerror("Error", "Analyze video first")


# ---------------- UI ----------------
root = tk.Tk()
root.title("AI Sports Analytics System")
root.geometry("780x650")
root.config(bg="#111827")

title = tk.Label(
    root,
    text="AI Sports Analytics System",
    font=("Arial", 22, "bold"),
    bg="#111827",
    fg="white"
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="YOLO + Jersey OCR + Tracking + Live Commentary",
    font=("Arial", 12),
    bg="#111827",
    fg="#d1d5db"
)
subtitle.pack()

btn_upload = tk.Button(
    root,
    text="Upload Video",
    width=28,
    font=("Arial", 12),
    command=upload_video
)
btn_upload.pack(pady=10)

lbl_video = tk.Label(
    root,
    text="No video selected",
    bg="#111827",
    fg="#9ca3af",
    font=("Arial", 11)
)
lbl_video.pack()

btn_analyze = tk.Button(
    root,
    text="Analyze Video",
    width=28,
    font=("Arial", 12),
    command=analyze_video
)
btn_analyze.pack(pady=10)

btn_play = tk.Button(
    root,
    text="Play Output Video",
    width=28,
    font=("Arial", 12),
    command=play_output
)
btn_play.pack(pady=5)

btn_stats = tk.Button(
    root,
    text="Open Player Stats",
    width=28,
    font=("Arial", 12),
    command=open_stats
)
btn_stats.pack(pady=5)

btn_comment = tk.Button(
    root,
    text="Open Commentary File",
    width=28,
    font=("Arial", 12),
    command=open_commentary
)
btn_comment.pack(pady=5)

lbl_status = tk.Label(
    root,
    text="Ready",
    bg="#111827",
    fg="#22c55e",
    font=("Arial", 12)
)
lbl_status.pack(pady=10)

comment_title = tk.Label(
    root,
    text="Live Commentary",
    bg="#111827",
    fg="white",
    font=("Arial", 14, "bold")
)
comment_title.pack()

txt_commentary = tk.Text(
    root,
    height=12,
    width=85,
    font=("Arial", 10)
)
txt_commentary.pack(pady=10)

note = tk.Label(
    root,
    text="Press Q to close output video. Jersey OCR works best with clear HD jersey numbers.",
    bg="#111827",
    fg="#9ca3af",
    font=("Arial", 10)
)
note.pack()

root.mainloop()