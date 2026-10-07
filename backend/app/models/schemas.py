"""
Pydantic response schemas — single source of truth for API contract types.
"""
from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeResponse(BaseModel):
    """Returned immediately after a video is uploaded (HTTP 202)."""
    job_id: str
    status: str = "analyzing"
    message: str = "Video uploaded successfully. Analysis started."


class TimelinePoint(BaseModel):
    timestamp: float = Field(..., description="Seconds from video start")
    score: float = Field(..., ge=0, le=1, description="Smoothed fake probability")


class ForensicBreakdown(BaseModel):
    """Sub-model signal decomposition derived from Grad-CAM zone analysis."""
    facial_blend_artifacts: float = Field(..., ge=0, le=1)
    lip_sync_coherence: float = Field(..., ge=0, le=1)
    frequency_spectrum_gan_traces: float = Field(..., ge=0, le=1)


class ForensicResult(BaseModel):
    """Full forensic analysis output."""
    classification: str = Field(..., description="DEEPFAKE or AUTHENTIC")
    confidence: float = Field(..., ge=0, le=1)
    analysis_duration_seconds: float
    frames_analyzed: int
    total_frames: int
    trimmed_mean_fake_score: float
    flagged_frame_ratio: float
    longest_anomaly_streak: int
    peak_frame_index: int
    peak_timestamp_seconds: float
    breakdown: ForensicBreakdown
    primary_forensic_evidence: str
    evidence_zone_percentage: float
    annotated_video_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    timeline_data: list[TimelinePoint] = []


class JobStatusResponse(BaseModel):
    """Returned when the frontend polls for job status."""
    job_id: str
    status: str = Field(..., description="analyzing | completed | error")
    result: Optional[ForensicResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    device: str = "unknown"
