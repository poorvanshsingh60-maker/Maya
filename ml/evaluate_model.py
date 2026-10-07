import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.deepfake_dataloader import DeepfakeDataset
from ml.model_architecture import DualStreamDeepfakeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

def evaluate():
    print("Loading Dual-Stream Xception Model for Evaluation...")
    
    # Initialize and load weights
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("best_maya_model.pth", map_location=DEVICE))
    model.eval()

    # Load the untouched TEST split
    test_dataset = DeepfakeDataset(manifest_path="Master_Manifest.csv", split='test', root_dir=".")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_targets, all_probs = [], [], []

    # Run inference across the entire test set
    with torch.no_grad():
        for rgb_img, hp_img, labels in tqdm(test_loader, desc="Evaluating Test Split"):
            rgb_img = rgb_img.to(DEVICE)
            hp_img = hp_img.to(DEVICE)

            logits = model(rgb_img, hp_img)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = [1 if p >= 0.5 else 0 for p in probs]

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy().flatten())

    # Calculate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate Statistical Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
    accuracy = accuracy_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_probs)

    print("\n" + "="*45)
    print(" 🎯 FORENSIC DIAGNOSTIC REPORT")
    print("="*45)
    print(f" Total Test Samples  : {len(all_targets)}")
    print(f" True Positives (TP) : {tp} (Fakes caught)")
    print(f" True Negatives (TN) : {tn} (Reals verified)")
    print(f" False Positives (FP): {fp} (Reals marked fake)")
    print(f" False Negatives (FN): {fn} (Fakes missed)")
    print("-" * 45)
    print(f" Accuracy            : {accuracy * 100:.2f}%")
    print(f" Sensitivity (Recall): {sensitivity * 100:.2f}%")
    print(f" Specificity         : {specificity * 100:.2f}%")
    print(f" ROC-AUC Score       : {roc_auc:.4f}")
    print("="*45)

    # Generate Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL (0)', 'FAKE (1)'], 
                yticklabels=['REAL (0)', 'FAKE (1)'])
    plt.title('Deepfake Detection Confusion Matrix')
    plt.ylabel('Actual Ground Truth')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("\n✅ Saved 'confusion_matrix.png' to your folder.")
    plt.show()

if __name__ == "__main__":
    evaluate()