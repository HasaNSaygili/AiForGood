import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
# Adjust these paths based on where your test images are
TEST_IMG_DIR = "d:/初赛赛道3/data/GID-img-1" # Example source
OUTPUT_DIR = "d:/初赛赛道3/predictions"
MODEL_PATH = "best_model.pth"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        if img is not None:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except:
        return None

def get_preprocessing():
    # Only normalization is needed for inference
    test_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

def predict():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! You need to train first.")
        return

    # Load Model Structure
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=None, # No need to download pre-trained weights again
        classes=1, 
        activation="sigmoid",
    )
    
    # Load Trained Weights
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    
    preprocessing = get_preprocessing()
    
    # Find test images (Example: searching recursively or flat)
    # Adjust the glob pattern if needed
    test_files = glob(os.path.join(TEST_IMG_DIR, "**/*.jpg"), recursive=True)[:10] # LIMIT to 10 for test
    if not test_files:
        test_files = glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
        
    print(f"Found {len(test_files)} images to predict.")
    
    for img_path in tqdm(test_files):
        # Read Image
        image = imread_unicode(img_path)
        if image is None: continue
        
        # Preprocess
        sample = preprocessing(image=image)
        image_tensor = sample['image'].unsqueeze(0).to(DEVICE) # Add batch dim (1, 3, 512, 512)
        
        # Predict
        with torch.no_grad():
            pred_mask = model(image_tensor)
            # Output is (1, 1, 512, 512) -> Convert to valid mask
            pred_mask = pred_mask.squeeze().cpu().numpy()
            
        # Thresholding (Sigmoid output 0-1 -> Binary 0 or 255)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Save
        filename = os.path.basename(img_path).replace(".jpg", ".png")
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, pred_mask)
        
    print(f"Predictions saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    predict()
