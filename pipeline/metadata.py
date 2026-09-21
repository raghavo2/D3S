"""
Metadata extraction: GPS from EXIF, full GPS/IMU from ROS bags, and
optional frame extraction from bag image topics.

V3 upgrade: The ROS bag extraction now fully deserializes GPS (NavSatFix),
IMU, and CompressedImage messages instead of just reading topic metadata.

Adapted from extract_gps.py and extract_telemetry.py.
"""

import os
import csv
import json
import math


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


# ──────────────────────────────────────────────────────────────────────────────
# ROS Bag — Full Telemetry Extraction (V3)
# ──────────────────────────────────────────────────────────────────────────────

def _find_bag_connections(reader, topic_priorities, msgtype_priorities):
    """
    Find matching connections in a ROS bag by topic name or message type.
    
    Searches topic_priorities first (in order), then falls back to
    msgtype_priorities. Returns all matching connections.
    """
    all_topics = {conn.topic: conn for conn in reader.connections}
    
    # Strategy 1: Match by exact topic name
    for topic in topic_priorities:
        if topic in all_topics:
            return [all_topics[topic]]
    
    # Strategy 2: Match by message type
    matches = []
    for conn in reader.connections:
        if conn.msgtype in msgtype_priorities:
            matches.append(conn)
    if matches:
        return matches
    
    # Strategy 3: Fuzzy match by keywords in topic name
    keywords_map = {
        "gps": ["gps", "navsat", "fix", "global_position"],
        "imu": ["imu"],
        "image": ["image", "camera"],
    }
    # Determine category from priorities
    category = None
    for cat, kws in keywords_map.items():
        for topic in topic_priorities:
            if any(kw in topic.lower() for kw in kws):
                category = cat
                break
        if category:
            break
    
    if category:
        kws = keywords_map[category]
        for conn in reader.connections:
            if any(kw in conn.topic.lower() for kw in kws):
                matches.append(conn)
    
    return matches


def _parse_navsatfix(msg):
    """
    Extract GPS data from a deserialized NavSatFix message.
    
    Handles both ROS1 (sensor_msgs/NavSatFix) and ROS2 
    (sensor_msgs/msg/NavSatFix) message structures.
    """
    result = {}
    
    # Standard NavSatFix fields
    if hasattr(msg, "latitude"):
        result["latitude"] = float(msg.latitude)
    if hasattr(msg, "longitude"):
        result["longitude"] = float(msg.longitude)
    if hasattr(msg, "altitude"):
        result["altitude"] = float(msg.altitude)
    
    # Status (0=no fix, 1=fix, 2=SBAS, 3=GBAS/RTK)
    if hasattr(msg, "status"):
        status = msg.status
        if hasattr(status, "status"):
            result["fix_status"] = int(status.status)
        elif isinstance(status, (int, float)):
            result["fix_status"] = int(status)
    
    # Position covariance (3x3 row-major)
    if hasattr(msg, "position_covariance"):
        cov = list(msg.position_covariance)
        if len(cov) >= 9:
            # Diagonal elements = variance in ENU
            result["covariance_east"] = float(cov[0])
            result["covariance_north"] = float(cov[4])
            result["covariance_up"] = float(cov[8])
            # Horizontal accuracy estimate (sqrt of avg horizontal variance)
            h_var = (cov[0] + cov[4]) / 2.0
            if h_var > 0:
                result["horizontal_accuracy_m"] = math.sqrt(h_var)
    
    # Covariance type (0=unknown, 1=approximated, 2=diagonal, 3=known)
    if hasattr(msg, "position_covariance_type"):
        result["covariance_type"] = int(msg.position_covariance_type)
    
    return result


def _parse_imu(msg):
    """
    Extract IMU data from a deserialized Imu message.
    """
    result = {}
    
    # Orientation quaternion
    if hasattr(msg, "orientation"):
        q = msg.orientation
        if hasattr(q, "w"):
            result["orientation"] = {
                "w": float(q.w),
                "x": float(q.x),
                "y": float(q.y),
                "z": float(q.z),
            }
    
    # Angular velocity (rad/s)
    if hasattr(msg, "angular_velocity"):
        av = msg.angular_velocity
        if hasattr(av, "x"):
            result["angular_velocity"] = {
                "x": float(av.x),
                "y": float(av.y),
                "z": float(av.z),
            }
    
    # Linear acceleration (m/s²)
    if hasattr(msg, "linear_acceleration"):
        la = msg.linear_acceleration
        if hasattr(la, "x"):
            result["linear_acceleration"] = {
                "x": float(la.x),
                "y": float(la.y),
                "z": float(la.z),
            }
    
    return result


def extract_telemetry_from_bag(bag_path, output_json=None, on_log=None):
    """
    Extract telemetry data from a ROS bag file (legacy interface — V2 compat).

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


def extract_full_telemetry_from_bag(bag_path, output_json=None, on_log=None):
    """
    Extract full GPS + IMU telemetry from a ROS bag with message deserialization.
    
    This is the V3 upgrade over extract_telemetry_from_bag() — it actually
    parses GPS coordinates and IMU orientation from the raw messages.

    Args:
        bag_path: Path to the .bag file.
        output_json: Optional path to write structured JSON output.
        on_log: Optional progress callback.

    Returns:
        dict with keys:
            topics: list of {topic, type, message_count}
            gps_data: list of {timestamp_ns, latitude, longitude, altitude, ...}
            imu_data: list of {timestamp_ns, orientation, angular_velocity, ...}
            has_images: bool (whether image topics exist in the bag)
            image_topics: list of image topic names
            gps_count: number of GPS messages parsed
            imu_count: number of IMU messages parsed
            time_range_s: total duration of bag in seconds
        Returns None if the bag cannot be read.
    """
    log = on_log or (lambda msg: None)

    if not os.path.isfile(bag_path):
        log(f"Bag file not found: {bag_path}")
        return None

    try:
        from pathlib import Path
        from rosbags.highlevel import AnyReader
        from pipeline.config import ProjectConfig
        config = ProjectConfig()

        log(f"Reading full telemetry from {bag_path}…")

        result = {
            "topics": [],
            "gps_data": [],
            "imu_data": [],
            "has_images": False,
            "image_topics": [],
            "gps_count": 0,
            "imu_count": 0,
            "time_range_s": 0,
        }

        with AnyReader([Path(bag_path)]) as reader:
            # --- Catalog all topics ---
            topic_counts = {}
            for conn in reader.connections:
                result["topics"].append({
                    "topic": conn.topic,
                    "type": conn.msgtype,
                })
                log(f"  Topic: {conn.topic} | Type: {conn.msgtype}")

            # --- Find GPS connections ---
            gps_conns = _find_bag_connections(
                reader, config.ROSBAG_GPS_TOPICS, config.ROSBAG_GPS_MSG_TYPES
            )
            if gps_conns:
                log(f"  GPS topics found: {[c.topic for c in gps_conns]}")
            else:
                log("  ⚠ No GPS topics found in bag")

            # --- Find IMU connections ---
            imu_conns = _find_bag_connections(
                reader, config.ROSBAG_IMU_TOPICS, config.ROSBAG_IMU_MSG_TYPES
            )
            if imu_conns:
                log(f"  IMU topics found: {[c.topic for c in imu_conns]}")
            else:
                log("  ⚠ No IMU topics found in bag")

            # --- Check for image topics ---
            image_conns = _find_bag_connections(
                reader, config.ROSBAG_IMAGE_TOPICS, config.ROSBAG_IMAGE_MSG_TYPES
            )
            if image_conns:
                result["has_images"] = True
                result["image_topics"] = [c.topic for c in image_conns]
                log(f"  Image topics found: {result['image_topics']}")

            # --- Parse GPS messages ---
            min_ts = float("inf")
            max_ts = float("-inf")

            if gps_conns:
                log("  Parsing GPS messages…")
                for conn, timestamp, rawdata in reader.messages(connections=gps_conns):
                    try:
                        msg = reader.deserialize(rawdata, conn.msgtype)
                        gps = _parse_navsatfix(msg)
                        if "latitude" in gps and "longitude" in gps:
                            gps["timestamp_ns"] = int(timestamp)
                            gps["topic"] = conn.topic
                            result["gps_data"].append(gps)
                            min_ts = min(min_ts, timestamp)
                            max_ts = max(max_ts, timestamp)
                    except Exception as e:
                        # Skip unparseable messages silently
                        pass
                result["gps_count"] = len(result["gps_data"])
                log(f"  Parsed {result['gps_count']} GPS messages")

            # --- Parse IMU messages (subsample: every 10th for storage) ---
            if imu_conns:
                log("  Parsing IMU messages…")
                imu_msg_idx = 0
                for conn, timestamp, rawdata in reader.messages(connections=imu_conns):
                    imu_msg_idx += 1
                    # Subsample IMU (typically 100-400Hz, we want ~10-40Hz)
                    if imu_msg_idx % 10 != 0:
                        continue
                    try:
                        msg = reader.deserialize(rawdata, conn.msgtype)
                        imu = _parse_imu(msg)
                        if imu:
                            imu["timestamp_ns"] = int(timestamp)
                            imu["topic"] = conn.topic
                            result["imu_data"].append(imu)
                            min_ts = min(min_ts, timestamp)
                            max_ts = max(max_ts, timestamp)
                    except Exception:
                        pass
                result["imu_count"] = len(result["imu_data"])
                log(f"  Parsed {result['imu_count']} IMU messages (subsampled)")

            # Time range
            if min_ts < float("inf") and max_ts > float("-inf"):
                result["time_range_s"] = round((max_ts - min_ts) / 1e9, 2)
                log(f"  Bag duration: {result['time_range_s']}s")

        # --- Save output ---
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            log(f"Full telemetry JSON saved: {output_json}")

        log(
            f"Telemetry extraction complete: "
            f"{result['gps_count']} GPS, {result['imu_count']} IMU, "
            f"images={'yes' if result['has_images'] else 'no'}"
        )
        return result

    except ImportError:
        log("rosbags package not installed — skipping telemetry extraction.")
        return None
    except Exception as e:
        log(f"Full telemetry extraction error: {e}")
        import traceback
        log(traceback.format_exc())
        return None


def extract_frames_from_bag(bag_path, output_dir, fps=2.0, on_log=None):
    """
    Extract video frames from image topics in a ROS bag.
    
    Searches for CompressedImage and Image topics, decodes the images,
    and saves them as JPEGs at the specified frame rate.

    Args:
        bag_path: Path to the .bag file.
        output_dir: Directory to save extracted frames.
        fps: Target extraction rate (frames per second). 
             Subsamples bag images to approximately this rate.
        on_log: Optional progress callback.

    Returns:
        dict with keys: frame_count, source_topic, total_messages, output_dir
        Returns None if no image topics are found or extraction fails.
    """
    log = on_log or (lambda msg: None)

    if not os.path.isfile(bag_path):
        log(f"Bag file not found: {bag_path}")
        return None

    try:
        import cv2
        import numpy as np
        from pathlib import Path
        from rosbags.highlevel import AnyReader
        from pipeline.config import ProjectConfig
        config = ProjectConfig()

        os.makedirs(output_dir, exist_ok=True)

        # Clean out old frames
        cleaned = 0
        for fname in os.listdir(output_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(output_dir, fname))
                cleaned += 1
        if cleaned:
            log(f"Cleared {cleaned} old images.")

        log(f"Searching for image topics in {bag_path}…")

        with AnyReader([Path(bag_path)]) as reader:
            # Find image connections
            image_conns = _find_bag_connections(
                reader, config.ROSBAG_IMAGE_TOPICS, config.ROSBAG_IMAGE_MSG_TYPES
            )

            if not image_conns:
                log("No image topics found in bag.")
                return None

            source_topic = image_conns[0].topic
            source_type = image_conns[0].msgtype
            log(f"Using image topic: {source_topic} ({source_type})")

            # Use only the first (highest priority) image connection
            conns = [image_conns[0]]

            # First pass: count messages to estimate timing
            total_messages = 0
            first_ts = None
            last_ts = None
            for conn, timestamp, rawdata in reader.messages(connections=conns):
                total_messages += 1
                if first_ts is None:
                    first_ts = timestamp
                last_ts = timestamp

            if total_messages == 0:
                log("No image messages found.")
                return None

            # Calculate source FPS and subsampling interval
            duration_s = (last_ts - first_ts) / 1e9 if first_ts != last_ts else 1.0
            source_fps = total_messages / max(duration_s, 0.001)
            frame_interval = max(1, int(round(source_fps / fps)))

            log(
                f"Source: {total_messages} images over {duration_s:.1f}s "
                f"(~{source_fps:.1f} FPS) → extracting every {frame_interval}th frame"
            )

            # Second pass: decode and save frames
            is_compressed = "Compressed" in source_type or "compressed" in source_topic
            saved_count = 0
            msg_idx = 0

            for conn, timestamp, rawdata in reader.messages(connections=conns):
                if msg_idx % frame_interval != 0:
                    msg_idx += 1
                    continue
                msg_idx += 1

                try:
                    msg = reader.deserialize(rawdata, conn.msgtype)

                    if is_compressed:
                        # CompressedImage: data is JPEG/PNG bytes
                        if hasattr(msg, "data"):
                            img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        else:
                            continue
                    else:
                        # Raw Image: interpret encoding
                        if hasattr(msg, "data") and hasattr(msg, "width"):
                            encoding = getattr(msg, "encoding", "bgr8")
                            h = int(msg.height)
                            w = int(msg.width)
                            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

                            if encoding in ("bgr8", "rgb8"):
                                frame = raw.reshape(h, w, 3)
                                if encoding == "rgb8":
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            elif encoding == "mono8":
                                frame = raw.reshape(h, w)
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            elif encoding in ("16UC1", "mono16"):
                                raw16 = np.frombuffer(bytes(msg.data), dtype=np.uint16)
                                frame = (raw16.reshape(h, w) / 256).astype(np.uint8)
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            else:
                                # Try default 3-channel interpretation
                                try:
                                    frame = raw.reshape(h, w, 3)
                                except ValueError:
                                    continue
                        else:
                            continue

                    if frame is not None:
                        filename = os.path.join(output_dir, f"{saved_count:05d}.jpg")
                        cv2.imwrite(filename, frame)
                        saved_count += 1

                        if saved_count % 10 == 0:
                            log(f"Extracted {saved_count} frames from bag…")

                except Exception as e:
                    # Skip individual frame decode errors
                    pass

        log(f"Done — saved {saved_count} frames from {source_topic} to {output_dir}")

        return {
            "frame_count": saved_count,
            "source_topic": source_topic,
            "total_messages": total_messages,
            "output_dir": output_dir,
        }

    except ImportError as e:
        log(f"Missing dependency for bag frame extraction: {e}")
        return None
    except Exception as e:
        log(f"Bag frame extraction error: {e}")
        import traceback
        log(traceback.format_exc())
        return None


def interpolate_gps_for_frames(gps_data, frame_timestamps_ns, on_log=None):
    """
    Interpolate GPS positions to match frame extraction timestamps.
    
    Uses linear interpolation between the two nearest GPS readings
    for each frame timestamp.

    Args:
        gps_data: list of GPS dicts from extract_full_telemetry_from_bag(),
                  each must have 'timestamp_ns', 'latitude', 'longitude', 'altitude'.
        frame_timestamps_ns: list of frame timestamps in nanoseconds.
        on_log: Optional progress callback.

    Returns:
        list of dicts: {frame_index, latitude, longitude, altitude, timestamp_ns}
    """
    log = on_log or (lambda msg: None)

    if not gps_data or not frame_timestamps_ns:
        return []

    # Sort GPS data by timestamp
    sorted_gps = sorted(gps_data, key=lambda g: g["timestamp_ns"])
    gps_times = [g["timestamp_ns"] for g in sorted_gps]

    interpolated = []
    for idx, frame_ts in enumerate(frame_timestamps_ns):
        # Binary search for nearest GPS readings
        import bisect
        pos = bisect.bisect_left(gps_times, frame_ts)

        if pos == 0:
            nearest = sorted_gps[0]
        elif pos >= len(sorted_gps):
            nearest = sorted_gps[-1]
        else:
            # Linear interpolation between gps[pos-1] and gps[pos]
            g_before = sorted_gps[pos - 1]
            g_after = sorted_gps[pos]
            t_before = g_before["timestamp_ns"]
            t_after = g_after["timestamp_ns"]
            dt = t_after - t_before
            if dt == 0:
                nearest = g_before
            else:
                alpha = (frame_ts - t_before) / dt
                nearest = {
                    "latitude": g_before["latitude"] + alpha * (g_after["latitude"] - g_before["latitude"]),
                    "longitude": g_before["longitude"] + alpha * (g_after["longitude"] - g_before["longitude"]),
                    "altitude": g_before["altitude"] + alpha * (g_after["altitude"] - g_before["altitude"]),
                }

        interpolated.append({
            "frame_index": idx,
            "latitude": nearest["latitude"],
            "longitude": nearest["longitude"],
            "altitude": nearest.get("altitude", 0),
            "timestamp_ns": frame_ts,
        })

    log(f"Interpolated GPS for {len(interpolated)} frames "
        f"from {len(gps_data)} GPS readings")
    return interpolated
