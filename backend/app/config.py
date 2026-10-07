"""
Maya Backend — Centralized Configuration (12-factor: all values from env vars with defaults).
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path Anchors
# ---------------------------------------------------------------------------
# config.py lives at  backend/app/config.py
# PROJECT_ROOT  →     Maya/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml"
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Ensure the backend directory and project root are on sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Model / Weights
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv("MAYA_MODEL_PATH", str(ML_DIR / "best_maya_model.pth"))
THRESHOLD_PATH: str = os.getenv("MAYA_THRESHOLD_PATH", str(ML_DIR / "calibrated_threshold.npy"))

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
UPLOAD_DIR: str = os.getenv("MAYA_UPLOAD_DIR", str(BACKEND_DIR / "uploads"))
RESULTS_DIR: str = os.getenv("MAYA_RESULTS_DIR", str(BACKEND_DIR / "results"))

# ---------------------------------------------------------------------------
# Upload Limits
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = int(os.getenv("MAYA_MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_MIME_TYPES: set[str] = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-msvideo",
}
ALLOWED_EXTENSIONS: set[str] = {".mp4", ".mov", ".webm", ".avi"}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = os.getenv("MAYA_DEVICE", "auto")  # "auto" | "cuda" | "cpu"

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
OPTIMAL_THRESHOLD: float = float(os.getenv("MAYA_THRESHOLD", "0.52"))
EMA_ALPHA: float = float(os.getenv("MAYA_EMA_ALPHA", "0.30"))

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ORIGINS: list[str] = os.getenv(
    "MAYA_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")
