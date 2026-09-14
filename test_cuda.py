import subprocess
import os

# Point directly to the batch file, not the .exe
colmap_bat = r"C:\Users\lenovo\OneDrive\Desktop\code_for_vs\d3s\colmap-x64-windows-cuda\COLMAP.bat"

print("Testing COLMAP Environment...")
try:
    # Run the help command to see if it initializes without crashing
    result = subprocess.run([colmap_bat, "help"], capture_output=True, text=True, check=True)
    print("Success! COLMAP is accessible via the batch file.")
except subprocess.CalledProcessError as e:
    print(f"Error running COLMAP: {e}")
except FileNotFoundError:
    print("Could not find COLMAP.bat at the specified path.")