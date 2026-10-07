import os
import glob
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Point directly to the current directory where the script and folders live
DATASET_ROOT = "." 
MANIFEST_PATH = os.path.join(DATASET_ROOT, "Master_Manifest.csv")

# ---------------------------------------------------------
# 1. Verify Manifest Balance & Category Rules
# ---------------------------------------------------------
print("=== 1. MANIFEST & BALANCE CHECK ===")
if not os.path.exists(MANIFEST_PATH):
    print("❌ ERROR: Master_Manifest.csv not found!")
else:
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Total Videos Tracked: {len(df)}")
    
    # Check 1:1 Balance
    real_count = len(df[df['label'] == 'REAL'])
    fake_count = len(df[df['label'] == 'FAKE'])
    print(f"Strict 1:1 Balance: {real_count} REAL vs {fake_count} FAKE")
        
    # Check Lip-Sync / Manipulation presence
    print("\nManipulation Types Breakdown:")
    print(df['type'].value_counts())
    
    # Check Splits
    print("\nDataset Splits:")
    print(df['split'].value_counts())

# ---------------------------------------------------------
# 2. Verify Image Dimensions & Extraction Counts
# ---------------------------------------------------------
print("\n=== 2. PHYSICAL IMAGE CHECK ===")
all_images = glob.glob(os.path.join(DATASET_ROOT, "**", "*.jpg"), recursive=True)
print(f"Total Extracted Frames Found: {len(all_images)}")

if len(all_images) > 0:
    # Sample a random image to verify dimensions
    sample_path = random.choice(all_images)
    img = cv2.imread(sample_path)
    
    if img is not None and img.shape == (380, 380, 3):
        print(f"✅ Image Resolution Verified: {img.shape} (Perfect for EfficientNet-B4)")
    else:
        print(f"❌ ERROR: Incorrect image resolution or unreadable file: {img.shape if img is not None else 'None'}")
        
    # ---------------------------------------------------------
    # 3. Visual Crop Verification (30% Margin)
    # ---------------------------------------------------------
    print("\n=== 3. VISUAL CROP VERIFICATION ===")
    print("Opening a plot window to show 4 random faces. Close the window to continue.")
    
    samples = random.sample(all_images, min(4, len(all_images)))
    plt.figure(figsize=(10, 3))
    
    for i, img_path in enumerate(samples):
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        is_fake = "FAKE" in img_path.upper()
        label_text = "FAKE" if is_fake else "REAL"
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img_rgb)
        plt.title(label_text)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
else:
    print("❌ ERROR: No .jpg files found. Check if train/val/test folders contain images.")