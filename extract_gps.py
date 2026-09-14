import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def convert_to_degrees(value):
    """Helper function to convert GPS coordinates to decimal degrees."""
    d, m, s = value
    return float(d) + float(m) / 60.0 + float(s) / 3600.0

def get_gps_data(image_path):
    try:
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
            lat = convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef", "N") != "N":
                lat = -lat
                
            lon = convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef", "E") != "E":
                lon = -lon
                
            alt = float(gps_info.get("GPSAltitude", 0))
            if "GPSAltitudeRef" in gps_info and gps_info["GPSAltitudeRef"] == b'\x01': # Below sea level check
                alt = -alt
                
            return {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "datetime": gps_info.get("DateTime", "N/A")
            }
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
    return None

def main():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AGZ_subset", "MAV Images")
    output_csv = "drone_flight_gps.csv"
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = sorted([
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(valid_extensions)
    ])
    
    print(f"Scanning {len(image_files)} images for GPS metadata...")
    
    flight_data = []
    for filename in image_files:
        path = os.path.join(image_dir, filename)
        gps = get_gps_data(path)
        if gps:
            flight_data.append({
                "filename": filename,
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
                "altitude": gps["altitude"],
                "datetime": gps["datetime"]
            })
            print(f"Found GPS for {filename}: Lat {gps['latitude']:.6f}, Lon {gps['longitude']:.6f}")
        else:
            print(f"No GPS metadata found in {filename}")
            
    if flight_data:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "latitude", "longitude", "altitude", "datetime"])
            writer.writeheader()
            writer.writerows(flight_data)
        print(f"\nSuccess! Exported GPS data for {len(flight_data)} images to {output_csv}")
    else:
        print("\nNo valid GPS data found in any of the images.")

if __name__ == "__main__":
    main()