# -*- coding: utf-8 -*-
"""
Download the DUSt3R ViTLarge checkpoint (full model).

Strategy:
  1. Try downloading from the official Naver Labs Europe server (direct .pth file).
  2. If that fails or is too slow, fall back to downloading via HuggingFace
     using DUSt3R's built-in from_pretrained (safetensors format).

Usage:
    python download_model.py
"""

import os
import sys
import time
import urllib.request

NAVER_URL = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
OUT_DIR = "checkpoints"
FILENAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
MIN_EXPECTED_SIZE = 1_300_000_000  # ~1.3 GB minimum


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_down:.1f}/{mb_total:.1f} MB")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb_down:.1f} MB...")
    sys.stdout.flush()


def validate_checkpoint(path):
    """Quick-check that torch can open the file."""
    try:
        import torch
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
        torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def download():
    out_path = os.path.join(OUT_DIR, FILENAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Check if valid file already exists
    if os.path.isfile(out_path):
        size = os.path.getsize(out_path)
        if size >= MIN_EXPECTED_SIZE:
            print(f"[OK] Model already exists ({size / (1024**2):.0f} MB): {out_path}")
            if validate_checkpoint(out_path):
                print("[OK] Model file is valid.")
                return
            else:
                print("[!!] Model file exists but is corrupted. Re-downloading...")
                os.remove(out_path)
        else:
            print(f"[!!] Existing file too small ({size / (1024**2):.0f} MB). Removing...")
            os.remove(out_path)

    # --- Attempt 1: Direct download from Naver Labs Europe ---
    print("\n[1/2] Downloading from Naver Labs Europe (direct .pth)...")
    print(f"  URL: {NAVER_URL}")
    tmp_path = out_path + ".tmp"
    try:
        start = time.time()
        urllib.request.urlretrieve(NAVER_URL, tmp_path, reporthook=progress_hook)
        elapsed = time.time() - start
        print()  # newline after progress bar

        size = os.path.getsize(tmp_path)
        if size >= MIN_EXPECTED_SIZE:
            os.replace(tmp_path, out_path)
            speed = size / (1024 * 1024) / max(elapsed, 0.1)
            print(f"  [OK] Downloaded {size / (1024**2):.0f} MB in {elapsed:.0f}s ({speed:.1f} MB/s)")
            print(f"  Saved to: {out_path}")
            return
        else:
            print(f"  [!!] File too small ({size / (1024**2):.0f} MB). Trying alternative...")
            os.remove(tmp_path)
    except Exception as e:
        print(f"\n  [!!] Naver download failed: {e}")
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    # --- Attempt 2: HuggingFace via DUSt3R's from_pretrained ---
    print("\n[2/2] Downloading from HuggingFace (naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt)...")
    print("  This uses DUSt3R's built-in model loading...")
    try:
        # Add dust3r to path
        project_dir = os.path.dirname(os.path.abspath(__file__))
        d3r_path = os.path.join(project_dir, "dust3r")
        croco_path = os.path.join(d3r_path, "croco")
        if d3r_path not in sys.path:
            sys.path.insert(0, d3r_path)
        if croco_path not in sys.path:
            sys.path.insert(0, croco_path)

        import torch
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])

        # pyrefly: ignore [missing-import]
        from dust3r.model import AsymmetricCroCo3DStereo
        model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")

        # Save as .pth so the pipeline can load it locally next time
        print("  Saving model checkpoint locally...")
        ckpt = {
            "model": model.state_dict(),
            "args": argparse.Namespace(
                model=repr(model).split("\n")[0]  # store constructor string
            ),
        }
        torch.save(ckpt, out_path)
        size = os.path.getsize(out_path)
        print(f"  [OK] Saved {size / (1024**2):.0f} MB to: {out_path}")
        return
    except Exception as e:
        print(f"  [!!] HuggingFace download failed: {e}")

    print("\n[FAIL] All download methods failed.")
    print("  Please check your internet connection and try again.")
    sys.exit(1)


if __name__ == "__main__":
    download()
