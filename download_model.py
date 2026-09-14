import os
import urllib.request

url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
out_dir = "checkpoints"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

print(f"Downloading {url} to {out_path}...")
urllib.request.urlretrieve(url, out_path)
print("Download complete.")
