import os
import cv2
import argparse

def process_video(video_path, output_dir, frames_per_second=2):
    # Ensure the destination folder exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- NEW: Clean out old frames to prevent video mixing ---
    print(f"Sweeping out old frames from {output_dir}...")
    cleaned_count = 0
    for filename in os.listdir(output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(output_dir, filename))
            cleaned_count += 1
    print(f"Cleared {cleaned_count} old images. Ready for fresh extraction.")
    # ---------------------------------------------------------
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        print("Error: Video FPS is 0. Is the file corrupted?")
        return
        
    duration = total_frames / fps
    print("-" * 60)
    print(f"Video Loaded: {total_frames} frames | {fps:.2f} FPS | {duration:.2f} seconds")
    print("-" * 60)
    
    # Calculate how many frames to skip to hit our target extraction rate
    frame_interval = max(1, int(fps / frames_per_second))
    
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting {frames_per_second} frames per second...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Only save the frame if it matches our interval
        if frame_count % frame_interval == 0:
            # Format with leading zeros for perfect alphabetical sorting
            filename = os.path.join(output_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Success! Saved {saved_count} frames into {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a drone video for 3D reconstruction.")
    parser.add_argument("video_path", help="Path to the input video file (e.g., drone_flight.mp4)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames to extract per second (default: 2)")
    
    args = parser.parse_args()
    
    OUTPUT_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "AGZ_subset",
        "MAV Images"
    )
    
    process_video(args.video_path, OUTPUT_FOLDER, frames_per_second=args.fps)