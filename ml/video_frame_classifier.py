import os
import shutil
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

from ml.deepfake_dataloader import extract_high_frequency_noise
from ml.model_architecture import DualStreamDeepfakeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEMP_DIR = "temp_video_frames"

def save_and_load_exact_training_crop(frame_bgr, box, margin_ratio, save_path):
    """
    Replicates the EXACT preprocessing used when building the training/test dataset:
    1. Takes MTCNN box [x1, y1, x2, y2] and adds margin_ratio (30%).
    2. Resizes to (380, 380) using standard cv2.resize (INTER_LINEAR).
    3. Saves physically to disk with cv2.imwrite(.jpg) and reads back with cv2.imread().
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(b) for b in box]
    w_box, h_box = x2 - x1, y2 - y1

    mx = int(w_box * margin_ratio)
    my = int(h_box * margin_ratio)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w_img, x2 + mx)
    cy2 = min(h_img, y2 + my)

    face_crop = frame_bgr[cy1:cy2, cx1:cx2]
    if face_crop.size == 0:
        return None

    # Exact resize used in dataset generation
    face_380_bgr = cv2.resize(face_crop, (380, 380))

    # Write to disk as .jpg and read back identically to deepfake_dataloader.py
    cv2.imwrite(save_path, face_380_bgr)
    disk_bgr = cv2.imread(save_path)
    disk_rgb = cv2.cvtColor(disk_bgr, cv2.COLOR_BGR2RGB)
    return disk_rgb

def compute_gradcam_and_bbox(model, rgb_tensor, hp_tensor):
    model.eval()
    rgb_tensor = rgb_tensor.clone().detach().requires_grad_(True)

    logits = model(rgb_tensor, hp_tensor, return_cam_features=True)
    prob = torch.sigmoid(logits).item()

    model.zero_grad()
    logits.backward()

    gradients = model.gradients.detach().cpu()[0]
    activations = model.activations.detach().cpu()[0]

    weights = torch.mean(gradients, dim=(1, 2))
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = F.relu(cam)
    cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam) + 1e-8)
    cam_380 = cv2.resize(cam.numpy(), (380, 380))

    threshold = np.percentile(cam_380, 80)
    binary_mask = (cam_380 >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = (0, 0, 0, 0)
    zone_desc = "Uniform facial texture"
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, x + w, y + h)
        cy = y + (h / 2.0)
        if cy > 225:
            zone_desc = "Lower Face / Mouth & Jawline Blending Seam"
        elif cy < 155:
            zone_desc = "Upper Face / Ocular & Brow Boundary"
        else:
            zone_desc = "Mid-Face / Cheek & Nasal Synthesis Region"

    return prob, cam_380, bbox, zone_desc

def classify_video_by_frames(video_path, num_frames_to_extract=20):
    print(f"\n🎬 Converting Video into Exact Dataset-Matched JPG Frames: {video_path}")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    mtcnn = MTCNN(keep_all=True, select_largest=True, device=DEVICE)
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("best_maya_model.pth", map_location=DEVICE))
    model.eval()

    # Exact validation/