import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from deepfake_dataloader import DeepfakeDataset
from model_architecture import DualStreamDeepfakeDetector

# Hyperparameters
BATCH_SIZE = 8       # Lower to 4 if you hit CUDA Out of Memory
EPOCHS = 10          # Increase to 20-30 for production-grade convergence
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f" Initializing Training on Device: {DEVICE}")
    
    # 1. Initialize Datasets and DataLoaders
    train_dataset = DeepfakeDataset(manifest_path="Master_Manifest.csv", split='train', root_dir=".")
    val_dataset = DeepfakeDataset(manifest_path="Master_Manifest.csv", split='val', root_dir=".")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Loaded {len(train_dataset)} Training videos | {len(val_dataset)} Validation videos.")

    # 2. Initialize Model, Loss, and Optimizer
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Mixed precision scaler for fast GPU training
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    best_val_auc = 0.0

    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}] - Training")
        for rgb_img, hp_img, labels in loop:
            rgb_img = rgb_img.to(DEVICE)
            hp_img = hp_img.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(rgb_img, hp_img)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(rgb_img, hp_img)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for rgb_img, hp_img, labels in tqdm(val_loader, desc=f"Epoch [{epoch}/{EPOCHS}] - Validating"):
                rgb_img = rgb_img.to(DEVICE)
                hp_img = hp_img.to(DEVICE)
                labels_gpu = labels.to(DEVICE).unsqueeze(1)
                
                logits = model(rgb_img, hp_img)
                loss = criterion(logits, labels_gpu)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_targets.extend(labels.numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Val Accuracy: {val_acc*100:.2f}% | Val ROC-AUC: {val_auc:.4f}")
        
        # Save Best Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_maya_model.pth")
            print(f"   ✅ Saved new best model! (AUC: {best_val_auc:.4f})\n")
        else:
            print("\n")

if __name__ == "__main__":
    train()