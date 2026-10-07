import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepfake_dataloader import extract_high_frequency_noise
from model_architecture import DualStreamDeepfakeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_existing_video_folder():
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("best_maya_model.pth", map_location=DEVICE))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    df = pd.read_csv("Master_Manifest.csv")
    # Pull from the entire test split (both REAL and FAKE) so you can verify both
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    row = test_df.sample(1).iloc[0]

    true_label = row['label']  # Ground truth: 'FAKE' or 'REAL'
    dataset_source = row['dataset']
    manipulation_type = row.get('manipulation', true_label)

    vid_name = f"{dataset_source}_{os.path.basename(row['vid']).split('.')[0]}"
    frame_dir = os.path.join("test", true_label, vid_name)
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    if not frames:
        print(f"❌ No frames found in {frame_dir}. Run again to sample another folder.")
        return

    print("\n" + "="*65)
    print(f" 📂 TEST FOLDER           : {vid_name}")
    print(f" 🎯 ACTUAL GROUND TRUTH   : {true_label} (Source: {dataset_source} | Type: {manipulation_type})")
    print(f" 🖼️  TOTAL FRAMES IN FOLDER: {len(frames)}")
    print("="*65)

    probs = []

    with torch.no_grad():
        for idx, img_path in enumerate(frames):
            orig_bgr = cv2.imread(img_path)
            # Ensure frame is 380x380 just as in deepfake_dataloader.py
            orig_bgr = cv2.resize(orig_bgr, (380, 380))
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            hp_img = extract_high_frequency_noise(orig_rgb)

            aug = transform(image=orig_rgb, mask=hp_img)
            rgb_t = aug['image'].unsqueeze(0).to(DEVICE)
            hp_t = (aug['mask'].permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(DEVICE)

            prob = torch.sigmoid(model(rgb_t, hp_t)).item()
            probs.append(prob)
            pred_str = "🚨 FAKE" if prob >= 0.50 else "✅ REAL"
            print(f"   Frame {idx+1:02d} ({os.path.basename(img_path)}) -> Fake Prob: {prob*100:6.2f}% | Pred: {pred_str}")

    mean_prob = float(np.mean(probs))
    max_prob = float(np.max(probs))
    fake_count = sum(1 for p in probs if p >= 0.50)
    fake_ratio = fake_count / len(frames)

    predicted_label = "FAKE" if (mean_prob >= 0.50 or fake_ratio >= 0.30) else "REAL"
    is_correct = (predicted_label == true_label)

    print("-" * 65)
    print(f" Frames Flagged Fake : {fake_count} / {len(frames)} ({fake_ratio*100:.1f}%)")
    print(f" Average Probability : {mean_prob*100:.2f}% | Peak Frame: {max_prob*100:.2f}%")
    print("-" * 65)
    print(f" ACTUAL GROUND TRUTH : {true_label} ({manipulation_type})")
    print(f" MODEL FINAL VERDICT : {'🚨 FAKE (DEEPFAKE)' if predicted_label == 'FAKE' else '✅ REAL (AUTHENTIC)'}")
    print(f" DIAGNOSTIC RESULT   : {'🎯 CORRECT PREDICTION!' if is_correct else '❌ MISCLASSIFIED'}")
    print("="*65 + "\n")

if __name__ == "__main__":
    test_existing_video_folder()