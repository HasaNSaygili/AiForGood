import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- PERFORMANCE ---
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION ---
DATA_DIR = r"C:\Users\tanri\Desktop\AiForGood\data"
IMG_DIR = r"C:\Users\tanri\Desktop\AiForGood\dataset_processed\images"
MASK_DIR = r"C:\Users\tanri\Desktop\AiForGood\dataset_processed\masks"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "sigmoid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 40
LR = 0.0001
NUM_WORKERS = 0

# --- UTILS ---
def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def imread_mask_unicode(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, np.uint8)
        return cv2.imdecode(chunk_arr, cv2.IMREAD_GRAYSCALE)
    except:
        return None

# --- DATASET ---
class WaterDataset(Dataset):
    def __init__(self, images_filenames, images_dir, masks_dir, transform=None):
        self.images_filenames = images_filenames
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        img_name = self.images_filenames[idx]
        image_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        image = imread_unicode(image_path)
        mask = imread_mask_unicode(mask_path)

        if image is None or mask is None:
            print(f"Warning: Failed to load {img_name}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)

        # ✅ Binary mask zorunlu
        mask = mask.astype('float32') / 255.0
        mask = (mask > 0.5).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if hasattr(mask, 'shape') and len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask

# --- AUGMENTATION ---
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.GaussNoise(p=1),
            A.CLAHE(clip_limit=3.0, p=1),
            A.HueSaturationValue(p=1),
        ], p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

# --- TRAINING ---
def train_model():
    print(f"Using device: {DEVICE}")

    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    if not all_files:
        print("No images found!")
        return

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_files)}, Val size: {len(val_files)}")

    train_dataset = WaterDataset(train_files, IMG_DIR, MASK_DIR, transform=get_training_augmentation())
    val_dataset = WaterDataset(val_files, IMG_DIR, MASK_DIR, transform=get_validation_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
        decoder_attention_type="scse",
    )
    model.to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    focal_loss = smp.losses.FocalLoss(mode="binary", gamma=2.0)

    def loss_fn(pred, mask):
        return 0.6 * dice_loss(pred, mask) + 0.4 * focal_loss(pred, mask)

    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=LR)])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_iou = 0.0

    print("\nStarting Training...")

    for epoch in range(EPOCHS):
        model.train()
        train_logs = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for img, mask in pbar:
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(img)

            if torch.isnan(prediction).any():
                print("Prediction has NaN!")

            loss = loss_fn(prediction, mask)
            loss.backward()
            optimizer.step()

            train_logs.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = np.mean(train_logs)

        # --- VALIDATION ---
        model.eval()
        val_logs = []
        iou_scores = []

        with torch.no_grad():
            for img, mask in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                img, mask = img.to(DEVICE), mask.to(DEVICE)

                prediction = model(img)
                loss = loss_fn(prediction, mask)
                val_logs.append(loss.item())

                preds = (prediction > 0.5).long()

                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds,
                    mask.long(),
                    mode="binary"
                )

                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")

                # ✅ NaN temizleme
                iou = torch.nan_to_num(iou, nan=0.0)

                iou_scores.extend(iou.cpu().numpy())

        val_loss = np.mean(val_logs)
        mean_iou = float(np.mean(iou_scores))

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Valid Loss : {val_loss:.4f}")
        print(f"  Valid IoU  : {mean_iou:.4f}")
        print(f"  LR         : {optimizer.param_groups[0]['lr']:.6f}")

        # ✅ Scheduler NaN güvenli
        if not np.isnan(mean_iou):
            scheduler.step(mean_iou)

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  🎉 New Best Model Saved! IoU: {best_iou:.4f}")

    print(f"\nTraining Finished! Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    train_model()