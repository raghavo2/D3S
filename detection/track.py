# ============================================================
# YOLO + BYTETRACK + CAMERA MOTION COMPENSATION
# LOCAL / VS CODE VERSION
# ============================================================

from ultralytics import YOLO
import cv2
import math
import numpy as np
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "D3S_YOLO_80.pt"
VIDEO_PATH = BASE_DIR / "input" / "input.mp4"

OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_VIDEO = OUTPUT_DIR / "tracked_compensated.mp4"
OUTPUT_JSON = OUTPUT_DIR / "object_tracking.json"

CONFIDENCE = 0.40

# Pixel movement threshold after camera compensation
MOVEMENT_THRESHOLD = 5.0

# ORB settings
ORB_FEATURES = 1500
MAX_MATCHES = 100


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

if not VIDEO_PATH.exists():

    raise FileNotFoundError(
        f"Input video not found:\n{VIDEO_PATH}"
    )


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(
    str(MODEL_PATH)
)

print("YOLO model loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("Opening video...")

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("--------------------------------------------")
print(f"Video          : {VIDEO_PATH.name}")
print(f"FPS            : {fps:.2f}")
print(f"Width          : {width}")
print(f"Height         : {height}")
print(f"Total Frames   : {total_frames}")
print("--------------------------------------------")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# TRACKING DATA
# ============================================================

previous_centers = {}

tracking_data = []

trajectories = defaultdict(list)


# ============================================================
# ORB CAMERA MOTION
# ============================================================

print()
print("Initializing ORB camera-motion estimation...")

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

previous_gray = None


# ============================================================
# CAMERA MOTION FUNCTION
# ============================================================

def estimate_camera_motion(
    previous_gray,
    current_gray
):

    kp1, des1 = orb.detectAndCompute(
        previous_gray,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        current_gray,
        None
    )

    if des1 is None or des2 is None:

        return None

    if len(kp1) < 10 or len(kp2) < 10:

        return None

    matches = bf.match(
        des1,
        des2
    )

    if len(matches) < 10:

        return None

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = matches[
        :MAX_MATCHES
    ]

    if len(good_matches) < 8:

        return None

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        5.0
    )

    return H


# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("Starting processing...")
print()

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_number += 1


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # ESTIMATE CAMERA MOTION
    # ========================================================

    H = None

    if previous_gray is not None:

        H = estimate_camera_motion(
            previous_gray,
            gray
        )


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=CONFIDENCE,

        verbose=False
    )

    result = results[0]


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ====================================================
        # PROCESS EACH OBJECT
        # ====================================================

        for box, track_id, cls, conf in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_name = model.names[
                int(cls)
            ]


            # ------------------------------------------------
            # MOTION VARIABLES
            # ------------------------------------------------

            motion = "NEW"

            raw_distance = 0.0

            compensated_distance = 0.0


            # =================================================
            # PREVIOUS POSITION
            # =================================================

            if track_id in previous_centers:

                old_x, old_y = (
                    previous_centers[
                        track_id
                    ]
                )


                # =============================================
                # RAW PIXEL MOVEMENT
                # =============================================

                raw_distance = math.sqrt(

                    (center_x - old_x) ** 2

                    +

                    (center_y - old_y) ** 2

                )


                # =============================================
                # CAMERA MOTION COMPENSATION
                # =============================================

                if H is not None:

                    old_center = np.array(
                        [
                            [
                                [
                                    old_x,
                                    old_y
                                ]
                            ]
                        ],
                        dtype=np.float32
                    )


                    predicted_center = (
                        cv2.perspectiveTransform(
                            old_center,
                            H
                        )
                    )


                    predicted_x = float(
                        predicted_center[
                            0,
                            0,
                            0
                        ]
                    )


                    predicted_y = float(
                        predicted_center[
                            0,
                            0,
                            1
                        ]
                    )


                    # =========================================
                    # RESIDUAL MOVEMENT
                    # =========================================

                    dx = (
                        center_x
                        - predicted_x
                    )

                    dy = (
                        center_y
                        - predicted_y
                    )


                    compensated_distance = math.sqrt(

                        dx ** 2

                        +

                        dy ** 2

                    )


                    # =========================================
                    # MOTION CLASSIFICATION
                    # =========================================

                    if (
                        compensated_distance
                        > MOVEMENT_THRESHOLD
                    ):

                        motion = "MOVING"

                    else:

                        motion = "STATIC"


                else:

                    # Camera motion unavailable
                    # fallback to raw movement

                    if (
                        raw_distance
                        > MOVEMENT_THRESHOLD
                    ):

                        motion = "MOVING"

                    else:

                        motion = "STATIC"


            # =================================================
            # SAVE CURRENT POSITION
            # =================================================

            previous_centers[
                track_id
            ] = (
                center_x,
                center_y
            )


            # =================================================
            # TRAJECTORY
            # =================================================

            trajectories[
                int(track_id)
            ].append(
                [
                    center_x,
                    center_y
                ]
            )


            # =================================================
            # OBJECT DATA
            # =================================================

            object_data = {

                "frame":
                    frame_number,

                "time_seconds":
                    round(
                        frame_number / fps,
                        3
                    ),

                "object_id":
                    int(track_id),

                "class":
                    class_name,

                "confidence":
                    round(
                        float(conf),
                        4
                    ),

                "bbox": {

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                },

                "center": {

                    "x": center_x,
                    "y": center_y

                },

                "motion":
                    motion,

                "raw_pixel_movement":
                    round(
                        raw_distance,
                        3
                    ),

                "compensated_pixel_movement":
                    round(
                        compensated_distance,
                        3
                    )

            }


            tracking_data.append(
                object_data
            )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            # =================================================
            # LABEL
            # =================================================

            label = (

                f"ID:{track_id} "

                f"{class_name} "

                f"{float(conf):.2f}"

            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(
                        y1 - 35,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2

            )


            # =================================================
            # CENTER
            # =================================================

            center_text = (

                f"Center: "

                f"({center_x},{center_y})"

            )


            cv2.putText(

                frame,

                center_text,

                (
                    x1,
                    max(
                        y1 - 15,
                        40
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 255),

                2

            )


            # =================================================
            # MOTION COLOR
            # =================================================

            if motion == "MOVING":

                motion_color = (
                    0,
                    0,
                    255
                )

            elif motion == "STATIC":

                motion_color = (
                    255,
                    255,
                    255
                )

            else:

                motion_color = (
                    255,
                    255,
                    0
                )


            # =================================================
            # MOTION
            # =================================================

            cv2.putText(

                frame,

                f"Motion: {motion}",

                (
                    x1,
                    y2 + 20
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                motion_color,

                2

            )


            # =================================================
            # MOVEMENT
            # =================================================

            cv2.putText(

                frame,

                (
                    f"Movement: "
                    f"{compensated_distance:.1f}px"
                ),

                (
                    x1,
                    y2 + 42
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 0),

                2

            )


    # ========================================================
    # SAVE PREVIOUS FRAME
    # ========================================================

    previous_gray = gray.copy()


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 30 == 0:

        if total_frames > 0:

            progress = (
                frame_number
                / total_frames
                * 100
            )

            print(
                f"Processing: "
                f"{progress:.1f}% "
                f"({frame_number}/"
                f"{total_frames})"
            )

        else:

            print(
                f"Processed "
                f"{frame_number} frames"
            )


# ============================================================
# RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        tracking_data,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("============================================")
print("PROCESSING COMPLETED")
print("============================================")

print(
    f"Output video:\n"
    f"{OUTPUT_VIDEO}"
)

print()

print(
    f"Tracking JSON:\n"
    f"{OUTPUT_JSON}"
)

print()

print(
    f"Tracking records: "
    f"{len(tracking_data)}"
)

print("============================================")
