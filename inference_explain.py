import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from deepfake_dataloader import extract_high_frequency_noise
from model_architecture import DualStreamDeepfakeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_gradcam_and_bbox(model, rgb_tensor, hp_tensor, original_rgb):
    """Computes Grad-CAM heatmap and extracts the bounding box of the highest anomaly region."""
    model.eval()
    rgb_tensor.requires_grad = True
    
    # Forward pass with hook enabled
    logits = model(rgb_tensor, hp_tensor, return_cam_features=True)
    prob = torch.sigmoid(logits).item()
    
    # Backward pass to get gradients with respect to the predicted class
    model.zero_grad()
    logits.backward()
    
    gradients = model.gradients.detach().cpu()[0]     # (C, H, W)
    activations = model.activations.detach().cpu()[0] # (C, H, W)
    
    # Global average pooling on gradients (Grad-CAM weights)
    weights = torch.mean(gradients, dim=(1, 2))
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i]
        
    cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-8)
    cam_np = cam.numpy()
    
    # Resize CAM to 380x380
    cam_resized = cv2.resize(cam_np, (380, 380))
    
    # Binarize top 20% heat intensity to find the primary forensic anomaly box
    threshold = np.percentile(cam_resized, 80)
    binary_mask = (cam_resized >= threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = (0, 0, 0, 0)
    region_name = "Uniform / No localized anomaly"
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = (x, y, x + w, y + h)
        
        # Determine anatomical zone based on vertical center of the anomaly box
        center_y = y + (h / 2)
        if center_y > 230:
            region_name = "Lower Face / Mouth Cavity (Lip-Sync or Jawline Seam)"
        elif center_y < 160:
            region_name = "Upper Face / Ocular Region (Eye Blending / Brow Seam)"
        else:
            region_name = "Mid-Face / Nasal & Cheek Boundary (FaceSwap Blending)"

    return prob, cam_resized, bbox, region_name

def run_test_inference():
    if not os.path.exists("best_maya_model.pth"):
        print("❌ Error: 'best_maya_model.pth' not found! Run train_model.py first.")
        return

    # Load Model
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("best_maya_model.pth", map_location=DEVICE))
    
    # Grab a random video from the unseen TEST split
    df = pd.read_csv("Master_Manifest.csv")
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    row = test_df.sample(1).iloc[0]
    
    vid_name = f"{row['dataset']}_{os.path.basename(row['vid']).split('.')[0]}"
    frame_dir = os.path.join("test", row['label'], vid_name)
    frames = glob.glob(os.path.join(frame_dir, "*.jpg"))
    
    if not frames:
        print("Selected test folder was empty, run script again to pick another sample.")
        return
        
    img_path = frames[0]
    orig_bgr = cv2.imread(img_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    hp_img = extract_high_frequency_noise(orig_rgb)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    augmented = transform(image=orig_rgb, mask=hp_img)
    rgb_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    hp_tensor = (augmented['mask'].permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(DEVICE)
    
    # Run Explainable Inference
    prob, heatmap, (x1, y1, x2, y2), region_name = generate_gradcam_and_bbox(model, rgb_tensor, hp_tensor, orig_rgb)
    
    verdict = "FAKE (Manipulated)" if prob >= 0.5 else "REAL (Authentic)"
    confidence = prob * 100 if prob >= 0.5 else (1.0 - prob) * 100
    
    # Generate Visual Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    # Draw Bounding Box on Fake detections
    boxed_img = orig_rgb.copy()
    if prob >= 0.5:
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(boxed_img, "Manipulated Zone", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    print("\n" + "="*50)
    print("🔍 FORENSIC AI INFERENCE REPORT")
    print("="*50)
    print(f"File Analyzed     : {img_path}")
    print(f"Ground Truth      : {row['label']} ({row['type']})")
    print(f"AI Verdict        : {verdict}")
    print(f"Confidence Score  : {confidence:.2f}%")
    if prob >= 0.5:
        print(f"Primary Anomaly   : {region_name}")
        print(f"Bounding Box      : [X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}]")
    print("="*50 + "\n")

    # Plot Results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title(f"Original Input\nTruth: {row['label']} ({row['type']})")
    plt.imshow(orig_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM++ Attribution\n(Spatial Attention)")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Forensic BBox Output\nPred: {verdict} ({confidence:.1f}%)")
    plt.imshow(boxed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test_inference()