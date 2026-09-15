"""
Frame extraction from drone video.
Adapted from extract_frames.py — original script is preserved unchanged.
"""

import os
import cv2


def extract_frames(video_path, output_dir, fps=2.0, on_log=None):
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        fps: Frames to extract per second of video (default 2.0).
        on_log: Optional callback ``on_log(message)`` for progress reporting.

    Returns:
        dict with keys: frame_count, video_fps, video_duration, output_dir
    
    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If video cannot be opened or has invalid metadata.
    """
    log = on_log or (lambda msg: None)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Clean out old frames to prevent mixing
    log(f"Cleaning output directory: {output_dir}")
    cleaned = 0
    for fname in os.listdir(output_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(output_dir, fname))
            cleaned += 1
    if cleaned:
        log(f"Cleared {cleaned} old images.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps == 0:
        cap.release()
        raise RuntimeError("Video FPS is 0 — file may be corrupted.")

    duration = total_frames / video_fps
    log(f"Video: {total_frames} frames | {video_fps:.2f} FPS | {duration:.2f}s")

    frame_interval = max(1, int(video_fps / fps))
    frame_count = 0
    saved_count = 0

    log(f"Extracting ~{fps} frames/second (every {frame_interval} frames)…")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            # Report progress every 10 frames
            if saved_count % 10 == 0:
                pct = min(100, int(frame_count / total_frames * 100))
                log(f"Extracted {saved_count} frames ({pct}%)")

        frame_count += 1

    cap.release()
    log(f"Done — saved {saved_count} frames to {output_dir}")

    return {
        "frame_count": saved_count,
        "video_fps": video_fps,
        "video_duration": round(duration, 2),
        "output_dir": output_dir,
    }
