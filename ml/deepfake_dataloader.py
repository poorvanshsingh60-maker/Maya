import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Frequency Domain Preprocessing (High-Pass Filter)
# -------------------------------------------------------------------
def extract_high_frequency_noise(img_rgb):
    """
    Converts to grayscale and applies a Laplacian filter to extract 
    microscopic pixel edges and frequency domain artifacts.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Apply Laplacian filter for high-frequency edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Normalize back to 0-255 uint8
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Expand to 1 channel (H, W, 1) so PyTorch can process it
    return np.expand_dims(laplacian, axis=-1)

# -------------------------------------------------------------------
# 2. Simplified Self-Blended Images (SBI)
# -------------------------------------------------------------------
def apply_self_blending(img_rgb):
    """
    Creates a pseudo-deepfake by heavily color-shifting a copy of the face, 
    generating a random mask, and blending it back onto the original face.
    This teaches the model to detect blending seams.
    """
    h, w, _ = img_rgb.shape
    
    # 1. Create a heavily altered copy (simulating a mismatched donor face)
    altered = cv2.convertScaleAbs(img_rgb, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
    altered = cv2.GaussianBlur(altered, (5, 5), 0)

    # 2. Create a random oval/circle mask in the center of the face
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x = random.randint(int(w*0.3), int(w*0.7))
    center_y = random.randint(int(h*0.3), int(h*0.7))
    axis_x = random.randint(int(w*0.2), int(w*0.4))
    axis_y = random.randint(int(h*0.2), int(h*0.4))
    
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
    
    # 3. Soften the mask heavily to simulate a deepfake Poisson blending seam
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_float = mask.astype(float) / 255.0
    mask_float = np.expand_dims(mask_float, axis=-1)

    # 4. Blend the altered face onto the real face
    blended = (mask_float * altered) + ((1.0 - mask_float) * img_rgb)
    return blended.astype(np.uint8)

# -------------------------------------------------------------------
# 3. The PyTorch Dataset Class
# -------------------------------------------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, manifest_path, split='train', root_dir='.'):
        self.root_dir = root_dir
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.split = split
        
        # Albumentations Dynamic Pipeline (Updated for v2.0+ syntax)
        if split == 'train':
            self.transform = A.Compose([
                A.ImageCompression(quality_range=(40, 90), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation/Test only gets Normalization (No augmentations)
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        vid_name = f"{row['dataset']}_{os.path.basename(row['vid']).split('.')[0]}"
        frame_dir = os.path.join(self.root_dir, row['split'], row['label'], vid_name)
        
        frames = glob.glob(os.path.join(frame_dir, "*.jpg"))
        if not frames:
            return torch.zeros((3, 380, 380)), torch.zeros((1, 380, 380)), torch.tensor(0.0, dtype=torch.float32)
            
        img_path = random.choice(frames) if self.split == 'train' else frames[0]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = 1.0 if row['label'] == 'FAKE' else 0.0

        if self.split == 'train' and label == 0.0 and random.random() < 0.3:
            image = apply_self_blending(image)
            label = 1.0 

        # Extract Frequency Domain (High Pass) BEFORE normalization
        high_pass_img = extract_high_frequency_noise(image)
        
        # Apply Albumentations
        augmented = self.transform(image=image, mask=high_pass_img)
        img_tensor = augmented['image']
        
        hp_tensor = augmented['mask'].permute(2, 0, 1).float() / 255.0

        return img_tensor, hp_tensor, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    dataset = DeepfakeDataset(manifest_path="Master_Manifest.csv", split='train', root_dir=".")
    
    print(f"Loaded {len(dataset)} training videos.")
    
    idx = random.randint(0, len(dataset)-1)
    img_t, hp_t, label = dataset[idx]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_view = img_t.permute(1, 2, 0).numpy() * std + mean
    img_view = np.clip(img_view, 0, 1)
    
    hp_view = hp_t.squeeze().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Augmented RGB Input\nLabel: {'FAKE' if label.item()==1 else 'REAL'}")
    plt.imshow(img_view)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("High-Pass Frequency Filter\n(Exposes Blending Seams)")
    plt.imshow(hp_view, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()