import os
import cv2
import numpy as np
from glob import glob

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, np.uint8)
        return cv2.imdecode(chunk_arr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main():
    base_dir = "d:/初赛赛道3/data"
    label_dir = os.path.join(base_dir, "GID-label")
    
    # Get first label file
    label_files = glob(os.path.join(label_dir, "*.png"))
    if not label_files:
        print("No label files found.")
        return

    target_label_path = label_files[0]
    filename = os.path.basename(target_label_path)
    print(f"Analyzing Target Label: {filename}")
    
    # 1. Analyze Colors
    img = imread_unicode(target_label_path)
    if img is not None:
        print(f"Label Shape: {img.shape}")
        # Reshape to list of pixels
        pixels = img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        print(f"Unique Colors found (BGR Format):")
        for color in unique_colors:
            print(f"  - {color} (Blue={color[0]}, Green={color[1]}, Red={color[2]})")
            
        # Verify Water (Blue: [255, 0, 0] in BGR)
        # Note: In RGB water is usually (0, 0, 255). In BGR it is (255, 0, 0).
        # Let's check specifically for pure blue
        if np.any(np.all(unique_colors == [255, 0, 0], axis=1)):
            print("  -> CONFIRMED: Water class (BGR: 255,0,0) is present.")
    
    # 2. Find Corresponding Image
    print("\nLooking for corresponding image source...")
    name_no_ext = os.path.splitext(filename)[0]
    
    found = False
    # Search in all 4 img directories
    for i in range(1, 5):
        search_dir = os.path.join(base_dir, f"GID-img-{i}")
        if not os.path.exists(search_dir): continue
        
        print(f"  Scanning {search_dir}...")
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.startswith(name_no_ext):
                    full_path = os.path.join(root, f)
                    print(f"\nSUCCESS! Found matching image: {full_path}")
                    # Read metadata of original image
                    org_img = imread_unicode(full_path)
                    if org_img is not None:
                        print(f"  Original Image Shape: {org_img.shape}")
                    found = True
                    break
            if found: break
        if found: break
        
    if not found:
        print("\nWARNING: Corresponding original image NOT found.")

if __name__ == "__main__":
    main()
