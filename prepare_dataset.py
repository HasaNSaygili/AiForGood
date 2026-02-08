import os
import cv2
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
SOURCE_DATA_DIR = "d:/初赛赛道3/data"
OUTPUT_DIR = "d:/初赛赛道3/dataset_processed"
LABEL_DIR = os.path.join(SOURCE_DATA_DIR, "GID-label")
IMG_DIRS = [os.path.join(SOURCE_DATA_DIR, f"GID-img-{i}") for i in range(1, 5)]

# Threshold strategy for Water detection
# GID Dataset: Water is Blue (0, 0, 255) in RGB.
# OpenCV reads as BGR -> Water is (255, 0, 0)
# Data is anti-aliased, so we use thresholding.
# Rule: Blue channel > 150 AND Red < 100 AND Green < 100
BLUE_THRESH = 150
OTHER_THRESH = 100

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, np.uint8)
        return cv2.imdecode(chunk_arr, cv2.IMREAD_UNCHANGED)
    except:
        return None

def imwrite_unicode(path, img):
    try:
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            with open(path, "wb") as f:
                f.write(buffer)
            return True
    except Exception as e:
        print(f"Error writing {path}: {e}")
    return False

def process_single_pair(label_path, image_path, output_img_dir, output_mask_dir):
    try:
        filename = os.path.basename(label_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # 1. Process Mask
        mask_img = imread_unicode(label_path)
        if mask_img is None:
            return False
            
        # Create Binary Mask for Water (Target = 1, Background = 0)
        # BGR format: mask_img[:, :, 0] is Blue
        b_channel = mask_img[:, :, 0]
        g_channel = mask_img[:, :, 1]
        r_channel = mask_img[:, :, 2]
        
        # Water condition: High Blue, Low Red/Green
        binary_mask = (b_channel > BLUE_THRESH) & (g_channel < OTHER_THRESH) & (r_channel < OTHER_THRESH)
        
        # Convert to uint8 (0 or 255 for visualization check, usually 0-1 for training but let's save as 0-255 png)
        # Standard: 0 is background, 255 is target (easier to see). We will normalize to 0-1 during training loading.
        final_mask = np.zeros_like(b_channel, dtype=np.uint8)
        final_mask[binary_mask] = 255 
        
        # Save Mask
        save_mask_path = os.path.join(output_mask_dir, name_no_ext + ".png")
        imwrite_unicode(save_mask_path, final_mask)
        
        # 2. Copy Image
        # We copy the image to have a clean dataset structure
        # Target image name same as mask name but .jpg
        save_img_path = os.path.join(output_img_dir, name_no_ext + ".jpg")
        shutil.copy2(image_path, save_img_path)
        
        return True
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return False

def build_image_map():
    print("Indexing all source images (this might take a minute)...")
    image_map = {}
    for d in IMG_DIRS:
        if not os.path.exists(d): continue
        print(f"Scanning {d}...")
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.jpg', '.tif', '.png')):
                    name_no_ext = os.path.splitext(f)[0]
                    image_map[name_no_ext] = os.path.join(root, f)
    return image_map

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_img_dir = os.path.join(OUTPUT_DIR, "images")
    out_mask_dir = os.path.join(OUTPUT_DIR, "masks")
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    # 1. Index Images
    image_map = build_image_map()
    print(f"Indexed {len(image_map)} images.")
    
    # 2. Get Labels
    label_files = glob(os.path.join(LABEL_DIR, "*.png"))
    print(f"Found {len(label_files)} label files.")
    
    # 3. Process
    print("Starting processing...")
    count = 0
    
    # Limit for testing? User can remove slice
    # process_files = label_files[:100] # For debug
    process_files = label_files 
    
    for label_path in tqdm(process_files):
        name = os.path.splitext(os.path.basename(label_path))[0]
        
        if name in image_map:
            success = process_single_pair(label_path, image_map[name], out_img_dir, out_mask_dir)
            if success:
                count += 1
        else:
            # print(f"Warning: No matching image for {name}")
            pass
            
    print(f"\nProcessing Complete!")
    print(f"Successfully processed {count} pairs.")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
