import os
from pathlib import Path
from rosbags.highlevel import AnyReader

def main():
    bag_path = Path(__file__).parent / "AGZ_subset" / "AGZ.bag"
    
    if not bag_path.exists():
        print(f"Error: Could not find AGZ.bag at {bag_path}")
        return

    print(f"Reading flight telemetry from {bag_path}...")
    
    # Use AnyReader to handle ROS1/ROS2 bags seamlessly without a full ROS install
    with AnyReader([bag_path]) as reader:
        print("\n--- Available ROS Topics ---")
        for connection in reader.connections:
            print(f"Topic: {connection.topic} | Type: {connection.msgtype}")
            
        print("\n--- Parsing Flight Records ---")
        message_count = 0
        
        # Filter connections for GPS, IMU, or state telemetry topics
        selected_connections = [
            conn for conn in reader.connections 
            if any(keyword in conn.topic.lower() for keyword in ["gps", "imu", "state", "fix", "odometry"])
        ]
        
        if not selected_connections:
            print("No explicit GPS/IMU keywords found in topics. Reading all messages...")
            selected_connections = reader.connections

        for connection, timestamp, rawdata in reader.messages(connections=selected_connections):
            message_count += 1
            msg = reader.deserialize(rawdata, connection.msgtype)
            # Print sample telemetry data from fields if available
            print(f"[{connection.topic}] Timestamp: {timestamp}")
            
        print(f"\nScanned bag successfully. Processed {message_count} telemetry messages.")

if __name__ == "__main__":
    main()