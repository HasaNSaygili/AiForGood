import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_IMAGE_PATH = r"C:\Users\tanri\Desktop\AiForGood\test_image\test2.jpg"
OUTPUT_DIR = r"C:\Users\tanri\Desktop\AiForGood\test_image\outputs"
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

ENCODER = "resnet34"
ACTIVATION = "sigmoid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================


def imread_unicode(path):
    with open(path, "rb") as f:
        chunk = f.read()
    chunk_arr = np.frombuffer(chunk, np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_preprocessing():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def predict_single():

    print(f"Using device: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found:", MODEL_PATH)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== MODEL (TRAIN İLE AYNI) =====
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=1,
        activation=ACTIVATION,
        decoder_attention_type="scse",
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded.")

    # ===== IMAGE LOAD =====
    image = imread_unicode(TEST_IMAGE_PATH)

    preprocessing = get_preprocessing()
    sample = preprocessing(image=image)
    image_tensor = sample["image"].unsqueeze(0).to(DEVICE)

    # ===== INFERENCE =====
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = pred_mask.squeeze().cpu().numpy()

    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # ===== SAVE =====
    output_path = os.path.join(OUTPUT_DIR, "pred_mask.png")
    cv2.imwrite(output_path, pred_mask)

    print(f"🎉 Prediction saved to: {output_path}")


if __name__ == "__main__":
    predict_single()