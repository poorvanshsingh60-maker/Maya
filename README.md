# Maya — Dual-Stream Deepfake Detection & Forensic Analysis System

Maya is an end-to-end, production-grade video deepfake detection system combining a dual-stream neural architecture (RGB spatial transfer learning via Xception + High-Pass Laplacian frequency domain filtering) with an explainable AI (Grad-CAM spatial attribution & anatomical landmark localization) pipeline and a modern React web dashboard.

---

## 🏛️ System Architecture

```
                                 [ Video Upload (.mp4, .mov, .webm, .avi) ]
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │       Client-Side Validation Gate       │
                               │  (MIME allowlist, 100 MB max hard cap)  │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                                    [ Axios Multipart Stream ]
                                (onUploadProgress + AbortSignal)
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │             FastAPI Backend             │
                               │   POST /api/analyze → BackgroundTask    │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                            ┌───────────────────────────────────────────────┐
                            │            Maya Model Pipeline                │
                            │                                               │
                            │  1. MTCNN Face Tracking & 380x380 Crop        │
                            │  2. DualStreamDeepfakeDetector Inference      │
                            │     ├── RGB Stream (Xception backbone)        │
                            │     └── Freq Stream (Laplacian High-Pass CNN) │
                            │  3. Exponential Moving Average (EMA) Smoothing│
                            │  4. Anomaly Persistence Gating (≥ 3 frames)   │
                            │  5. Grad-CAM Spatial Heatmap & Anomaly Box    │
                            │  6. Anatomical Zone Landmark Classification   │
                            └───────────────────────┬───────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │           Forensic Results              │
                               │  - Executive Verdict & Confidence       │
                               │  - Sub-Signal Decomposition (3 metrics) │
                               │  - Annotated MP4 Stream                 │
                               │  - Peak-Frame Grad-CAM Heatmap          │
                               │  - Temporal Manipulation Timeline       │
                               └─────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Backend Setup & Startup

The backend is built with **FastAPI** and Python 3.10+.

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Start the FastAPI server (runs on port 8000)
python -m uvicorn backend.main:app --reload --port 8000
```

- **Swagger / OpenAPI Documentation:** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/api/health`

#### Running Backend Tests
```powershell
.\.venv\Scripts\pytest.exe -v backend/tests/
```

---

### 2. Frontend Setup & Startup

The frontend is built with **React 18 + Vite**.

```powershell
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies (if not already installed)
npm install

# 3. Launch Vite development server
npm run dev
```

- **Web Dashboard:** `http://localhost:5173`

#### Running Frontend Tests
```powershell
cd frontend
npm run test
```

---

## ⚙️ Configuration (.env)

### Backend (`.env` or Environment Variables)
| Variable | Default | Description |
|---|---|---|
| `MAYA_MODEL_PATH` | `ml/best_maya_model.pth` | Path to trained model weights |
| `MAYA_UPLOAD_DIR` | `backend/uploads` | Directory for uploaded videos |
| `MAYA_RESULTS_DIR` | `backend/results` | Directory for output videos & heatmaps |
| `MAYA_MAX_FILE_SIZE_MB`| `100` | Maximum upload size in megabytes |
| `MAYA_CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins for Vite |
| `MAYA_DEVICE` | `auto` | Execution device (`auto`, `cuda`, `cpu`) |

### Frontend (`frontend/.env`)
| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Target FastAPI backend URL |
| `VITE_MOCK_API` | `false` | Enable simulated responses without backend |
| `VITE_MAX_FILE_SIZE_MB`| `100` | Client-side validation hard limit |

---

## 🔬 REST API Specifications

### `POST /api/analyze`
Accepts `multipart/form-data` with field `video`.
Returns `202 Accepted`:
```json
{
  "job_id": "4a8e1b2f",
  "status": "analyzing",
  "message": "Video uploaded successfully. Analysis started."
}
```

### `GET /api/results/{job_id}`
Returns the current job status or completed forensic report:
```json
{
  "job_id": "4a8e1b2f",
  "status": "completed",
  "result": {
    "classification": "DEEPFAKE",
    "confidence": 0.8924,
    "analysis_duration_seconds": 12.4,
    "frames_analyzed": 142,
    "total_frames": 180,
    "trimmed_mean_fake_score": 0.7412,
    "flagged_frame_ratio": 0.648,
    "longest_anomaly_streak": 35,
    "peak_frame_index": 84,
    "peak_timestamp_seconds": 2.8,
    "breakdown": {
      "facial_blend_artifacts": 0.824,
      "lip_sync_coherence": 0.712,
      "frequency_spectrum_gan_traces": 0.895
    },
    "primary_forensic_evidence": "Lower Face / Mouth Cavity (Lip-Sync or Jawline Seam)",
    "evidence_zone_percentage": 74.2,
    "annotated_video_url": "/api/results/4a8e1b2f/video",
    "heatmap_url": "/api/results/4a8e1b2f/heatmap",
    "timeline_data": [
      { "timestamp": 0.0, "score": 0.12 },
      { "timestamp": 0.033, "score": 0.14 }
    ]
  }
}
```
