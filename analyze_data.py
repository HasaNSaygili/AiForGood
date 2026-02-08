import os
import numpy as np
import cv2
from glob import glob

def imread_unicode(path):
    """Reads image from path containing unicode characters."""
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def analyze_labels(label_dir):
    print(f"Analyzing labels in: {label_dir}")
    label_files = glob(os.path.join(label_dir, "*.png"))
    
    if not label_files:
        print("No label files found!")
        return
    
    print(f"Found {len(label_files)} label files.")
    
    # Analyze first 5 randomly
    sample_files = label_files[:5]
    
    for f in sample_files:
        print(f"\nChecking: {os.path.basename(f)}")
        img = imread_unicode(f)
        
        if img is None:
            print("Failed to read image")
            continue
            
        print(f"Shape: {img.shape}")
        unique_vals = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
        print(f"Unique pixel values (BGR): \n{unique_vals}")
        
        # Check if water (Blue: 255, 0, 0 in BGR) exists
        # Note: Depending on how it's saved it might be RGB or BGR, usually OpenCV reads BGR
        water_mask = np.all(img == [255, 0, 0], axis=-1) # BGR for blue
        if np.any(water_mask):
            print("Contains Water (Blue [255,0,0]) pixels!")
        else:
            print("No Water pixels found in this sample.")

def check_image_match(data_dir, label_dir):
    print("\nChecking image-label correspondence...")
    label_files = glob(os.path.join(label_dir, "*.png"))
    if not label_files:
        return
        
    sample_label = os.path.basename(label_files[0])
    print(f"Sample Label File: {sample_label}")
    
    # Label: GF2_PMS1__L1A0000564539-MSS1_0_0_size512.png
    # The original image usually has the same name but likely .tif extension
    # Let's try to find the EXACT name first, then .tif
    
    img_dirs = glob(os.path.join(data_dir, "GID-img-*"))
    found = False
    
    start_time = os.times()
    
    for d in img_dirs:
        print(f"Searching in {d}...")
        # Recursive search for the filename (ignoring extension for matching)
        target_name_no_ext = os.path.splitext(sample_label)[0]
        
        # Use simple os.walk for better control than glob recursive
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.startswith(target_name_no_ext):
                    print(f"Found match: {os.path.join(root, file)}")
                    found = True
                    break
            if found: break
        if found: break
            
    if not found:
        print("Could NOT find matching original image for the sample label in GID-img-* folders.")

if __name__ == "__main__":
    data_dir = "d:/初赛赛道3/data"
    label_dir = os.path.join(data_dir, "GID-label")
    
    try:
        analyze_labels(label_dir)
        check_image_match(data_dir, label_dir)
    except Exception as e:
        print(f"Error: {e}")
