"""
Metadata extraction: GPS from EXIF and telemetry from ROS bags.
Adapted from extract_gps.py and extract_telemetry.py.
"""

import os
import csv
import json


def _convert_to_degrees(value):
    """Convert GPS coordinate tuple (degrees, minutes, seconds) to decimal."""
    d, m, s = value
    return float(d) + float(m) / 60.0 + float(s) / 3600.0


def _get_gps_from_image(image_path):
    """Extract GPS data from a single image's EXIF tags."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for t in value:
                    sub_tag = GPSTAGS.get(t, t)
                    gps_info[sub_tag] = value[t]
            elif tag == "DateTimeOriginal":
                gps_info["DateTime"] = value

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef", "N") != "N":
                lat = -lat

            lon = _convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef", "E") != "E":
                lon = -lon

            alt = float(gps_info.get("GPSAltitude", 0))
            if "GPSAltitudeRef" in gps_info and gps_info["GPSAltitudeRef"] == b"\x01":
                alt = -alt

            return {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "datetime": gps_info.get("DateTime", "N/A"),
            }
    except Exception:
        pass
    return None


def extract_gps_from_images(image_dir, output_csv=None, output_json=None, on_log=None):
    """
    Scan images in a directory for EXIF GPS metadata.

    Args:
        image_dir: Directory containing images (.jpg, .jpeg, .png).
        output_csv: Optional path to write CSV output.
        output_json: Optional path to write JSON output.
        on_log: Optional progress callback.

    Returns:
        list of dicts with keys: filename, latitude, longitude, altitude, datetime
    """
    log = on_log or (lambda msg: None)

    if not os.path.isdir(image_dir):
        log(f"Image directory not found: {image_dir}")
        return []

    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)
    )
    log(f"Scanning {len(image_files)} images for GPS metadata…")

    flight_data = []
    for fname in image_files:
        path = os.path.join(image_dir, fname)
        gps = _get_gps_from_image(path)
        if gps:
            flight_data.append({"filename": fname, **gps})
            log(
                f"GPS found: {fname} → "
                f"Lat {gps['latitude']:.6f}, Lon {gps['longitude']:.6f}, "
                f"Alt {gps['altitude']:.1f}m"
            )

    log(f"GPS data extracted from {len(flight_data)}/{len(image_files)} images.")

    # Write CSV if requested
    if output_csv and flight_data:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "latitude", "longitude", "altitude", "datetime"]
            )
            writer.writeheader()
            writer.writerows(flight_data)
        log(f"GPS CSV saved: {output_csv}")

    # Write JSON if requested
    if output_json and flight_data:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(flight_data, f, indent=2)
        log(f"GPS JSON saved: {output_json}")

    return flight_data


def extract_telemetry_from_bag(bag_path, output_json=None, on_log=None):
    """
    Extract telemetry data from a ROS bag file.

    Args:
        bag_path: Path to the .bag file.
        output_json: Optional path to write JSON output.
        on_log: Optional progress callback.

    Returns:
        dict with keys: topics, message_count, messages (list of samples)
    """
    log = on_log or (lambda msg: None)

    if not os.path.isfile(bag_path):
        log(f"Bag file not found: {bag_path}")
        return None

    try:
        from pathlib import Path
        from rosbags.highlevel import AnyReader

        log(f"Reading telemetry from {bag_path}…")
        result = {"topics": [], "message_count": 0, "messages": []}

        with AnyReader([Path(bag_path)]) as reader:
            # Catalog available topics
            for conn in reader.connections:
                result["topics"].append(
                    {"topic": conn.topic, "type": conn.msgtype}
                )
                log(f"Topic: {conn.topic} | Type: {conn.msgtype}")

            # Filter for telemetry-relevant topics
            keywords = ["gps", "imu", "state", "fix", "odometry"]
            selected = [
                c
                for c in reader.connections
                if any(kw in c.topic.lower() for kw in keywords)
            ]
            if not selected:
                log("No GPS/IMU keywords found — reading all topics.")
                selected = list(reader.connections)

            for conn, timestamp, rawdata in reader.messages(connections=selected):
                result["message_count"] += 1
                # Store a sample of messages (first 100 for display)
                if len(result["messages"]) < 100:
                    result["messages"].append(
                        {"topic": conn.topic, "timestamp": int(timestamp)}
                    )

        log(f"Parsed {result['message_count']} telemetry messages from {len(result['topics'])} topics.")

        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            log(f"Telemetry JSON saved: {output_json}")

        return result

    except ImportError:
        log("rosbags package not installed — skipping telemetry extraction.")
        return None
    except Exception as e:
        log(f"Telemetry extraction error: {e}")
        return None
