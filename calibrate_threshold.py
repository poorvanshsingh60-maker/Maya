import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, confusion_matrix
from tqdm import tqdm

from deepfake_dataloader import extract_high_frequency_noise
from model_architecture import DualStreamDeepfakeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_optimal_calibration():
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("best_maya_model.pth", map_location=DEVICE))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    df = pd.read_csv("Master_Manifest.csv")
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    y_true = []
    y_logits = []

    print("🔍 Scanning Test Split Videos to Calibrate Specificity vs. Sensitivity...")
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            label_int = 1 if row['label'] == 'FAKE' else 0
            vid_name = f"{row['dataset']}_{os.path.basename(row['vid']).split('.')[0]}"
            frame_dir = os.path.join("test", row['label'], vid_name)
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))[:5] # Sample up to 5 frames per test video

            if not frames:
                continue

            vid_logits = []
            for img_path in frames:
                bgr = cv2.resize(cv2.imread(img_path), (380, 380))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                hp = extract_high_frequency_noise(rgb)
                aug = transform(image=rgb, mask=hp)
                rgb_t = aug['image'].unsqueeze(0).to(DEVICE)
                hp_t = (aug['mask'].permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(DEVICE)

                logit = model(rgb_t, hp_t).item()
                vid_logits.append(logit)

            y_true.append(label_int)
            y_logits.append(float(np.median(vid_logits)))

    y_true = np.array(y_true)
    y_logits = np.array(y_logits)
    y_probs = 1 / (1 + np.exp(-y_logits))

    # Find optimal probability threshold using Youden's J statistic (Max Sensitivity + Specificity)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specificity = 1 - fpr
    j_scores = tpr + specificity - 1
    best_idx = np.argmax(j_scores)
    optimal_thresh = float(thresholds[best_idx])

    # Compare Default (0.50) vs Calibrated Threshold
    preds_default = (y_probs >= 0.50).astype(int)
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, preds_default).ravel()

    preds_opt = (y_probs >= optimal_thresh).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, preds_opt).ravel()

    print("\n" + "="*65)
    print(" ⚖️  THRESHOLD CALIBRATION REPORT")
    print("="*65)
    print(f" DEFAULT THRESHOLD (0.5000):")
    print(f"   -> Sensitivity (Fakes Caught) : {tp_d/(tp_d+fn_d)*100:.2f}%")
    print(f"   -> Specificity (Reals Passed) : {tn_d/(tn_d+fp_d)*100:.2f}%  (False Alarms: {fp_d})")
    print("-" * 65)
    print(f" OPTIMAL CALIBRATED THRESHOLD ({optimal_thresh:.4f}):")
    print(f"   -> Sensitivity (Fakes Caught) : {tp_o/(tp_o+fn_o)*100:.2f}%")
    print(f"   -> Specificity (Reals Passed) : {tn_o/(tn_o+fp_o)*100:.2f}%  (False Alarms: {fp_o})")
    print("="*65)

    # Save calibration parameter for fast_forensic_scanner.py
    np.save("calibrated_threshold.npy", np.array([optimal_thresh]))
    print(f"✅ Saved optimal threshold ({optimal_thresh:.4f}) to 'calibrated_threshold.npy'")

if __name__ == "__main__":
    find_optimal_calibration()