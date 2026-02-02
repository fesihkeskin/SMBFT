"""Batch-resize plotted PNG figures for consistent dimensions."""

from PIL import Image
import os
from pathlib import Path
# Set project root and import custom dataset utilities
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"Project root: {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

# Change to the directory where the images are stored
img_folder = PROJECT_ROOT / "notebooks/experiments"
max_width = 1500
max_height = 800

for img_file in img_folder.glob("*_gt_vs_pred.png"):
    img = Image.open(img_file)
    w, h = img.size
    # Rescale the image if it exceeds the maximum dimensions
    scale = min(max_width / w, max_height / h, 1.0)  # Ensure scale is at most 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    if scale < 1.0:
        img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
        img_resized.save(img_file)
        print(f"{img_file.name} resized: {w}x{h} -> {new_w}x{new_h}")
    else:
        print(f"{img_file.name} already small enough: {w}x{h}")
        
print("All images saved in:", img_folder)