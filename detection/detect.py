import os
import cv2
import json
from collections import Counter
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "D3S_YOLO_80.pt"
)

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "detected_video.mp4"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "detections.json"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.35


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded.")
print(f"Number of classes: {len(model.names)}")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\nVideo information")
print("-----------------")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# DETECTION DATA
# ============================================================

detections_data = []

class_counter = Counter()

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

print("\nStarting detection...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------------------------
    # YOLO PREDICTION
    # --------------------------------------------------------

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    frame_objects = []


    # --------------------------------------------------------
    # GET DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            # Bounding box
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # Confidence
            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )


            # Class ID
            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )


            # Class name
            class_name = model.names[
                class_id
            ]


            # Count
            class_counter[
                class_name
            ] += 1


            # ------------------------------------------------
            # STORE DETECTION
            # ------------------------------------------------

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    confidence,
                    4
                ),
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            }

            frame_objects.append(
                detection
            )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # SAVE FRAME DATA
    # --------------------------------------------------------

    detections_data.append({
        "frame": frame_number,
        "timestamp": round(
            frame_number / fps,
            3
        ),
        "objects": frame_objects
    })


    # --------------------------------------------------------
    # DISPLAY FRAME NUMBER
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    video_writer.write(frame)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 50 == 0:

        percentage = (
            frame_number /
            total_frames *
            100
        )

        print(
            f"Processed "
            f"{frame_number}/{total_frames} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()
video_writer.release()

cv2.destroyAllWindows()


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        detections_data,
        file,
        indent=4
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n===================================")
print("YOLO DETECTION COMPLETED")
print("===================================")

print(
    f"\nAnnotated video:\n"
    f"{OUTPUT_VIDEO}"
)

print(
    f"\nDetection JSON:\n"
    f"{OUTPUT_JSON}"
)

print("\nDetected classes:")

for class_name, count in (
    class_counter.most_common()
):

    print(
        f"{class_name:<20} {count}"
    )

print(
    "\nTotal bounding boxes:",
    sum(class_counter.values())
)

print("\nDone!")

