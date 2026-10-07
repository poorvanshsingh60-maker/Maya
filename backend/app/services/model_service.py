"""
ModelService — Singleton that loads the deepfake detection model once at startup
and exposes `analyze_video()` for the API layer.

Pipeline (single pass):
  1. Read every frame → MTCNN face detect → crop + resize 380×380
  2. Run inference (no_grad) → EMA temporal smoothing → annotate frame → write to output video
  3. Track peak frame (highest smoothed probability)
  4. After all frames: aggregate verdict (trimmed mean, flagged ratio, persistence gate)
  5. Run Grad-CAM on peak frame → heatmap + breakdown signals + zone classification
  6. Return structured dict consumed by the API schemas
"""
import json
import logging
import os
import time
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from app.config import (
    DEVICE as DEVICE_CONFIG,
    EMA_ALPHA,
    MODEL_PATH,
    PROJECT_ROOT,
)

# Ensure project root is on sys.path (config.py already does this at import time)
import app.config  # noqa: F401 — triggers the sys.path side-effect

from ml.model_architecture import DualStreamDeepfakeDetector


def extract_high_frequency_noise(img_rgb: np.ndarray) -> np.ndarray:
    """
    Converts to grayscale and applies a Laplacian filter to extract
    microscopic pixel edges and frequency domain artifacts.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.normalize(
        laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return np.expand_dims(laplacian, axis=-1)

logger = logging.getLogger(__name__)


class ModelService:
    """Thread-safe singleton — instantiated once, shared across requests."""

    _instance: "ModelService | None" = None

    def __init__(self) -> None:
        self.model: DualStreamDeepfakeDetector | None = None
        self.mtcnn: Any = None
        self.device: torch.device | None = None
        self.transform: A.Compose | None = None
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Startup loader (called once from lifespan)
    # ------------------------------------------------------------------
    def load(self) -> None:
        from facenet_pytorch import MTCNN

        # Resolve device
        if DEVICE_CONFIG == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE_CONFIG)

        logger.info("Loading model on device: %s", self.device)

        # MTCNN face detector
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # Dual-stream deepfake detector
        self.model = DualStreamDeepfakeDetector(
            model_name="xception", pretrained=False
        ).to(self.device)
        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        except Exception:
            state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Transform identical to training-time validation transform
        self.transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        self.is_loaded = True
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_video(
        self, video_path: str, job_id: str, results_dir: str
    ) -> dict:
        """Run full forensic analysis.  Returns a result dict (or an error dict)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "corrupt_video", "message": "Could not open video file."}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output annotated video
        output_video_path = os.path.join(results_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # ── Tracking state ──────────────────────────────────────────────
        frame_records: list[dict] = []
        ema_prob: float | None = None
        consecutive_fake_streak = 0
        max_fake_streak = 0

        # Peak frame (only keep one face crop in memory)
        peak_smoothed = -1.0
        peak_face_rgb: np.ndarray | None = None
        peak_hp: np.ndarray | None = None
        peak_info: dict | None = None

        frame_count = 0

        logger.info(
            "Processing video: %d frames @ %d fps (%dx%d)",
            total_frames, fps, width, height,
        )

        # ── SINGLE-PASS: inference + annotation ────────────────────────
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes, _ = self.mtcnn.detect(frame_rgb)

            if boxes is not None:
                x1, y1, x2, y2 = (int(b) for b in boxes[0])
                w_box, h_box = x2 - x1, y2 - y1
                mx, my = int(w_box * 0.3), int(h_box * 0.3)

                x1e = max(0, x1 - mx)
                y1e = max(0, y1 - my)
                x2e = min(width, x2 + mx)
                y2e = min(height, y2 + my)

                face_crop = frame_rgb[y1e:y2e, x1e:x2e]

                if face_crop.size != 0:
                    face_380 = cv2.resize(face_crop, (380, 380))
                    hp_img = extract_high_frequency_noise(face_380)

                    augmented = self.transform(image=face_380, mask=hp_img)
                    rgb_t = augmented["image"].unsqueeze(0).to(self.device)
                    hp_t = (
                        augmented["mask"]
                        .permute(2, 0, 1)
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    with torch.no_grad():
                        logits = self.model(rgb_t, hp_t)
                        raw_prob = torch.sigmoid(logits).item()

                    # EMA temporal smoothing
                    if ema_prob is None:
                        ema_prob = raw_prob
                    else:
                        ema_prob = (EMA_ALPHA * raw_prob) + (
                            (1.0 - EMA_ALPHA) * ema_prob
                        )

                    # Streak tracking
                    if ema_prob >= 0.5:
                        consecutive_fake_streak += 1
                        max_fake_streak = max(max_fake_streak, consecutive_fake_streak)
                    else:
                        consecutive_fake_streak = 0

                    is_sustained_fake = (ema_prob >= 0.5) and (
                        consecutive_fake_streak >= 3
                    )
                    display_conf = (
                        ema_prob * 100 if ema_prob >= 0.5 else (1.0 - ema_prob) * 100
                    )

                    # Record for aggregation
                    frame_records.append(
                        {
                            "frame_idx": frame_count,
                            "timestamp": frame_count / fps,
                            "raw_prob": raw_prob,
                            "smoothed_prob": ema_prob,
                        }
                    )

                    # Track peak
                    if ema_prob > peak_smoothed:
                        peak_smoothed = ema_prob
                        peak_face_rgb = face_380.copy()
                        peak_hp = hp_img.copy()
                        peak_info = {
                            "frame_idx": frame_count,
                            "timestamp": frame_count / fps,
                        }

                    # ── Annotate frame for output video ─────────────────
                    color = (0, 0, 255) if is_sustained_fake else (0, 255, 0)
                    status_text = "FAKE" if is_sustained_fake else "REAL"
                    label = f"{status_text} ({display_conf:.1f}%)"
                    cv2.rectangle(frame, (x1e, y1e), (x2e, y2e), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1e, max(30, y1e - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            out.write(frame)

            if frame_count % 50 == 0:
                logger.info(
                    "  Processed %d / %d frames…", frame_count, total_frames
                )

        cap.release()
        out.release()

        # ── Guard: no faces ─────────────────────────────────────────────
        if not frame_records:
            return {
                "error": "no_faces_detected",
                "message": "No faces were detected in the video.",
            }

        # ── AGGREGATION ─────────────────────────────────────────────────
        smoothed_probs = np.array([r["smoothed_prob"] for r in frame_records])

        sorted_probs = np.sort(smoothed_probs)
        trim_n = max(1, int(len(sorted_probs) * 0.05))
        trimmed_probs = (
            sorted_probs[trim_n:-trim_n] if len(sorted_probs) > 20 else sorted_probs
        )

        mean_fake = float(np.mean(trimmed_probs))
        fake_ratio = float(
            np.sum(smoothed_probs >= 0.5) / len(smoothed_probs)
        )
        min_persist = max(5, int(fps * 0.25))

        if max_fake_streak >= min_persist and (
            mean_fake >= 0.50 or fake_ratio >= 0.30
        ):
            classification = "DEEPFAKE"
            signal = max(mean_fake, fake_ratio)
            confidence = float(np.clip(signal, 0.501, 0.999))
        else:
            classification = "AUTHENTIC"
            confidence = float(np.clip(1.0 - mean_fake, 0.501, 0.999))

        # ── PEAK-FRAME GRAD-CAM ────────────────────────────────────────
        heatmap_path = os.path.join(results_dir, "heatmap.png")
        breakdown, primary_evidence, evidence_pct = self._generate_peak_gradcam(
            peak_face_rgb, peak_hp, heatmap_path, classification  # type: ignore[arg-type]
        )

        # ── Timeline ───────────────────────────────────────────────────
        timeline = [
            {
                "timestamp": round(r["timestamp"], 3),
                "score": round(r["smoothed_prob"], 4),
            }
            for r in frame_records
        ]

        elapsed = time.time() - start_time

        result = {
            "classification": classification,
            "confidence": round(confidence, 4),
            "analysis_duration_seconds": round(elapsed, 2),
            "frames_analyzed": len(frame_records),
            "total_frames": total_frames,
            "trimmed_mean_fake_score": round(mean_fake, 4),
            "flagged_frame_ratio": round(fake_ratio, 4),
            "longest_anomaly_streak": max_fake_streak,
            "peak_frame_index": peak_info["frame_idx"] if peak_info else 0,
            "peak_timestamp_seconds": (
                round(peak_info["timestamp"], 2) if peak_info else 0.0
            ),
            "breakdown": breakdown,
            "primary_forensic_evidence": primary_evidence,
            "evidence_zone_percentage": evidence_pct,
            "annotated_video_url": f"/api/results/{job_id}/video",
            "heatmap_url": f"/api/results/{job_id}/heatmap",
            "timeline_data": timeline,
        }

        # Persist result JSON alongside the video/heatmap
        with open(os.path.join(results_dir, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)

        logger.info(
            "Analysis complete — %s (%.2f%%) in %.1fs",
            classification,
            confidence * 100,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_peak_gradcam(
        self,
        face_rgb: np.ndarray,
        hp_img: np.ndarray,
        save_path: str,
        classification: str,
    ) -> tuple[dict, str, float]:
        """
        Run Grad-CAM on a single 380×380 face crop.
        Returns (breakdown_dict, primary_evidence_str, evidence_zone_pct).
        """
        augmented = self.transform(image=face_rgb, mask=hp_img)
        rgb_t = augmented["image"].unsqueeze(0).to(self.device)
        hp_t = (
            augmented["mask"]
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        # Need gradients for Grad-CAM
        rgb_t = rgb_t.clone().detach().requires_grad_(True)

        self.model.eval()
        logits = self.model(rgb_t, hp_t, return_cam_features=True)

        self.model.zero_grad()
        logits.backward()

        gradients = self.model.gradients.detach().cpu()[0]   # (C, H, W)
        activations = self.model.activations.detach().cpu()[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_380 = cv2.resize(cam.numpy(), (380, 380))

        # ── Save heatmap overlay ───────────────────────────────────────
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * cam_380), cv2.COLORMAP_JET
        )
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(face_bgr, 0.55, heatmap_colored, 0.45, 0)
        cv2.imwrite(save_path, overlay)

        # ── Breakdown: zone-based Grad-CAM intensities ─────────────────
        h = cam_380.shape[0]  # 380
        upper_intensity = float(np.mean(cam_380[: h // 3, :]))
        middle_intensity = float(np.mean(cam_380[h // 3 : 2 * h // 3, :]))
        lower_intensity = float(np.mean(cam_380[2 * h // 3 :, :]))
        overall_intensity = float(np.mean(cam_380))

        breakdown = {
            "facial_blend_artifacts": round(middle_intensity, 4),
            "lip_sync_coherence": round(lower_intensity, 4),
            "frequency_spectrum_gan_traces": round(overall_intensity, 4),
        }

        # ── Primary forensic evidence from anomaly binarisation ────────
        threshold = np.percentile(cam_380, 80)
        binary_mask = (cam_380 >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        primary_evidence = "No localized anomaly detected"
        evidence_pct = 0.0

        if contours and classification == "DEEPFAKE":
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h_box = cv2.boundingRect(largest)
            cy = y + h_box / 2.0

            if cy > 253:
                primary_evidence = (
                    "Lower Face / Mouth Cavity (Lip-Sync or Jawline Seam)"
                )
            elif cy < 127:
                primary_evidence = (
                    "Upper Face / Ocular Region (Eye Blending / Brow Seam)"
                )
            else:
                primary_evidence = (
                    "Mid-Face / Nasal & Cheek Boundary (FaceSwap Blending)"
                )

            evidence_pct = round(
                cv2.contourArea(largest) / (380 * 380) * 100, 1
            )

        return breakdown, primary_evidence, evidence_pct
