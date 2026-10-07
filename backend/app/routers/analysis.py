"""
Analysis router — video upload, job polling, and result retrieval.
"""
import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    RESULTS_DIR,
    UPLOAD_DIR,
)
from app.models.schemas import AnalyzeResponse, JobStatusResponse
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])

# ---------------------------------------------------------------------------
# In-memory job store  (dict[job_id → {status, result, error}])
# Sufficient for single-server / demo use.
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _run_analysis(job_id: str, video_path: str) -> None:
    """Executed in a BackgroundTasks thread — outside the request lifecycle."""
    try:
        service = ModelService.get_instance()
        results_dir = os.path.join(RESULTS_DIR, job_id)
        os.makedirs(results_dir, exist_ok=True)

        result = service.analyze_video(video_path, job_id, results_dir)

        if "error" in result:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = result["message"]
        else:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result
    except Exception:
        logger.exception("Analysis failed for job %s", job_id)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = "Internal server error during analysis."


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------
@router.post("/api/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze_video(
    video: UploadFile,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    # ── Validate MIME type ──────────────────────────────────────────────
    if video.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {video.content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    # ── Validate extension ──────────────────────────────────────────────
    ext = Path(video.filename or "upload.mp4").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported extension: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Generate job ID ─────────────────────────────────────────────────
    job_id = uuid4().hex[:8]

    # ── Stream file to disk (check size as we go) ───────────────────────
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    file_size = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):  # 1 MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
                    )
                f.write(chunk)
    except HTTPException:
        # Clean up partial file on size violation
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

    logger.info(
        "Uploaded %s (%.2f MB) → job %s",
        video.filename,
        file_size / (1024 * 1024),
        job_id,
    )

    # ── Create job entry & schedule background work ─────────────────────
    jobs[job_id] = {"status": "analyzing", "result": None, "error": None}
    background_tasks.add_task(_run_analysis, job_id, save_path)

    return AnalyzeResponse(
        job_id=job_id,
        status="analyzing",
        message="Video uploaded successfully. Analysis started.",
    )


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}
# ---------------------------------------------------------------------------
@router.get("/api/results/{job_id}", response_model=JobStatusResponse)
async def get_result(job_id: str) -> JobStatusResponse:
    if job_id not in jobs:
        # Check disk cache in case server restarted
        import json
        result_file = os.path.join(RESULTS_DIR, job_id, "result.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    cached_result = json.load(f)
                jobs[job_id] = {
                    "status": "completed",
                    "result": cached_result,
                    "error": None,
                }
            except Exception:
                pass

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}/video
# ---------------------------------------------------------------------------
@router.get("/api/results/{job_id}/video")
async def get_annotated_video(job_id: str) -> FileResponse:
    video_path = os.path.join(RESULTS_DIR, job_id, "annotated.mp4")
    if not os.path.exists(video_path):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found.")
        if jobs[job_id]["status"] != "completed":
            raise HTTPException(status_code=409, detail="Analysis not yet complete.")
        raise HTTPException(status_code=404, detail="Annotated video not found.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"maya_analysis_{job_id}.mp4",
    )


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}/heatmap
# ---------------------------------------------------------------------------
@router.get("/api/results/{job_id}/heatmap")
async def get_heatmap(job_id: str) -> FileResponse:
    heatmap_path = os.path.join(RESULTS_DIR, job_id, "heatmap.png")
    if not os.path.exists(heatmap_path):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found.")
        if jobs[job_id]["status"] != "completed":
            raise HTTPException(status_code=409, detail="Analysis not yet complete.")
        raise HTTPException(status_code=404, detail="Heatmap not found.")

    return FileResponse(
        heatmap_path,
        media_type="image/png",
        filename=f"maya_heatmap_{job_id}.png",
    )
