import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from collections import Counter

from deepfake_dataloader import extract_high_frequency_noise
from model_architecture import DualStreamDeepfakeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_cam_for_video(model, rgb_tensor, hp_tensor):
    """Generates raw probability and 380x380 Grad-CAM heatmap."""
    model.eval()
    rgb_tensor.requires_grad = True
    
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
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-8)
    
    return prob, cv2.resize(cam.numpy(), (380, 380))

def classify_anatomical_zone(cy_norm):
    """Maps normalized vertical center (0.0 to 1.0) of the anomaly box to facial anatomy."""
    if cy_norm > 0.60:
        return "Lower Face / Mouth Cavity (Lip-Sync or Jawline Seam)"
    elif cy_norm < 0.42:
        return "Upper Face / Ocular Region (Eye Blending / Brow Seam)"
    else:
        return "Mid-Face / Nasal & Cheek Boundary (FaceSwap Blending)"

def process_video(input_path, output_path):
    print(f"🎬 Loading Video for Full Forensic Analysis: {input_path}")
    
    mtcnn = MTCNN(keep_all=False, device=DEVICE)
    model = DualStreamDeepfakeDetector(model_name='xception', pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("best_maya_model.pth", map_location=DEVICE))
    model.eval()
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # --- Temporal Tracking & Aggregation Variables ---
    frame_count = 0
    raw_probs = []
    smoothed_probs = []
    anomaly_regions = []
    
    ema_prob = None
    ALPHA = 0.30  # Smoothing factor (lower = more resistant to sudden lighting/compression spikes)
    
    consecutive_fake_streak = 0
    max_fake_streak = 0
    MIN_PERSISTENCE_FRAMES = max(5, int(fps * 0.25)) # Requires ~0.25s of continuous anomaly to count as sustained

    print("⏳ Processing frames with Temporal EMA Smoothing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is not None:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            w_box, h_box = x2 - x1, y2 - y1
            mx, my = int(w_box * 0.3), int(h_box * 0.3)
            
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(width, x2 + mx), min(height, y2 + my)
            
            face_crop = frame_rgb[y1:y2, x1:x2]
            
            if face_crop.size != 0:
                crop_h, crop_w = face_crop.shape[:2]
                face_380 = cv2.resize(face_crop, (380, 380))
                
                hp_img = extract_high_frequency_noise(face_380)
                augmented = transform(image=face_380, mask=hp_img)
                rgb_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
                hp_tensor = (augmented['mask'].permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(DEVICE)
                
                # 1. Raw Frame Inference
                raw_prob, cam_380 = generate_cam_for_video(model, rgb_tensor, hp_tensor)
                raw_probs.append(raw_prob)
                
                # 2. Apply Temporal EMA Smoothing
                if ema_prob is None:
                    ema_prob = raw_prob
                else:
                    ema_prob = (ALPHA * raw_prob) + ((1.0 - ALPHA) * ema_prob)
                smoothed_probs.append(ema_prob)
                
                # Track consecutive fake frames to filter out single-frame compression glitches
                if ema_prob >= 0.5:
                    consecutive_fake_streak += 1
                    max_fake_streak = max(max_fake_streak, consecutive_fake_streak)
                else:
                    consecutive_fake_streak = 0

                is_sustained_fake = (ema_prob >= 0.5) and (consecutive_fake_streak >= 3)
                display_conf = ema_prob * 100 if ema_prob >= 0.5 else (1.0 - ema_prob) * 100
                
                # 3. Map Grad-CAM Heatmap back to video
                cam_scaled = cv2.resize(cam_380, (crop_w, crop_h))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_scaled), cv2.COLORMAP_JET)
                blended_face = cv2.addWeighted(frame[y1:y2, x1:x2], 0.6, heatmap_colored, 0.4, 0)
                frame[y1:y2, x1:x2] = blended_face
                
                # 4. Box Binarizer (Only triggers if smoothed probability confirms anomaly)
                threshold = np.percentile(cam_scaled, 80)
                binary_mask = (cam_scaled >= threshold).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours and is_sustained_fake:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                    
                    # Log anatomical zone for final report
                    cy_norm = (cy + (ch / 2.0)) / float(crop_h)
                    zone = classify_anatomical_zone(cy_norm)
                    anomaly_regions.append(zone)
                    
                    cv2.rectangle(frame, (x1 + cx, y1 + cy), (x1 + cx + cw, y1 + cy + ch), (0, 0, 255), 3)
                    cv2.putText(frame, "Anomaly", (x1 + cx, max(20, y1 + cy - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Outer Face Tracker Box
                color = (0, 0, 255) if is_sustained_fake else (0, 255, 0)
                status_text = "FAKE" if is_sustained_fake else "REAL"
                label = f"{status_text} (Smoothed: {display_conf:.1f}%)"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(30, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    # =========================================================================
    # 5. FINAL VIDEO-LEVEL VERDICT & OUTLIER REJECTION
    # =========================================================================
    if not smoothed_probs:
        print("❌ No faces were detected in the video.")
        return

    # Sort probabilities and trim top 5% and bottom 5% outliers (lighting/blur spikes)
    sorted_probs = np.sort(smoothed_probs)
    trim_count = max(1, int(len(sorted_probs) * 0.05))
    if len(sorted_probs) > 20:
        trimmed_probs = sorted_probs[trim_count:-trim_count]
    else:
        trimmed_probs = sorted_probs

    mean_fake_prob = float(np.mean(trimmed_probs))
    fake_frame_ratio = sum(1 for p in smoothed_probs if p >= 0.5) / len(smoothed_probs)

    # Final Decision Logic:
    # Requires BOTH a persistent streak of fake frames AND either a high average probability (>0.50)
    # or a significant portion of the video flagged (>30% of frames) to catch partial manipulations.
    if max_fake_streak >= MIN_PERSISTENCE_FRAMES and (mean_fake_prob >= 0.50 or fake_frame_ratio >= 0.30):
        final_verdict = "🚨 DEEPFAKE (MANIPULATED VIDEO)"
        # Calculate overall confidence based on the stronger signal
        overall_signal = max(mean_fake_prob, fake_frame_ratio)
        final_confidence = min(99.9, max(50.1, overall_signal * 100))
    else:
        final_verdict = "✅ AUTHENTIC (REAL VIDEO)"
        final_confidence = (1.0 - mean_fake_prob) * 100

    print("\n" + "="*65)
    print(" 🏛️  FINAL EXECUTIVE VIDEO FORENSIC VERDICT")
    print("="*65)
    print(f" Analyzed Video File       : {input_path}")
    print(f" Total Face Frames Tracked : {len(smoothed_probs)} / {total_frames}")
    print(f" Trimmed Mean Fake Score   : {mean_fake_prob * 100:.2f}% (Outliers Removed)")
    print(f" Flagged Frame Ratio       : {fake_frame_ratio * 100:.1f}% of video duration")
    print(f" Longest Anomaly Streak    : {max_fake_streak} consecutive frames (Threshold: {MIN_PERSISTENCE_FRAMES})")
    print("-" * 65)
    print(f" FINAL VERDICT             : {final_verdict}")
    print(f" OVERALL CONFIDENCE        : {final_confidence:.2f}%")
    
    if "DEEPFAKE" in final_verdict and anomaly_regions:
        most_common_zone, count = Counter(anomaly_regions).most_common(1)[0]
        zone_pct = (count / len(anomaly_regions)) * 100
        print(f" Primary Forensic Evidence : {most_common_zone}")
        print(f"                             (Localized in {zone_pct:.1f}% of anomalous frames)")
    elif max_fake_streak > 0 and "AUTHENTIC" in final_verdict:
        print(" Diagnostic Note           : Minor transient spikes were detected but rejected")
        print("                             as lighting/compression noise (failed persistence gate).")
    print("="*65)
    print(f"💾 Annotated video saved to: {output_path}\n")

if __name__ == "__main__":
    INPUT_VIDEO = "test_video.mp4" 
    OUTPUT_VIDEO = "analyzed_output.mp4"
    
    if os.path.exists(INPUT_VIDEO):
        process_video(INPUT_VIDEO, OUTPUT_VIDEO)
    else:
        print(f"❌ Could not find '{INPUT_VIDEO}'. Place an mp4 file in the folder and name it test_video.mp4")