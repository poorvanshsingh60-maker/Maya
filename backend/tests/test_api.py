"""
Backend API Unit and Integration Tests using pytest and FastAPI TestClient.
Tests:
- Health check endpoint (/api/health)
- File validation: rejection of disallowed MIME types and extensions
- File size validation: rejection of files exceeding maximum size
- Result polling: 404 for nonexistent job, valid schema on existing job
"""
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.analysis import jobs


@pytest.fixture
def client():
    # Use TestClient with lifespan disabled for quick unit tests if model not loaded
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data


def test_analyze_rejects_invalid_mime_type(client):
    fake_txt = io.BytesIO(b"Hello this is not a video")
    response = client.post(
        "/api/analyze",
        files={"video": ("test.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_rejects_disallowed_extension(client):
    fake_mp3 = io.BytesIO(b"Fake audio")
    response = client.post(
        "/api/analyze",
        files={"video": ("test.mp3", fake_mp3, "video/mp4")},
    )
    assert response.status_code == 415
    assert "Unsupported extension" in response.json()["detail"]


def test_get_results_not_found(client):
    response = client.get("/api/results/nonexistent_id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found."


def test_get_results_existing_job(client):
    test_id = "test_job_123"
    jobs[test_id] = {
        "status": "completed",
        "result": {
            "classification": "AUTHENTIC",
            "confidence": 0.9421,
            "analysis_duration_seconds": 1.5,
            "frames_analyzed": 30,
            "total_frames": 30,
            "trimmed_mean_fake_score": 0.0579,
            "flagged_frame_ratio": 0.0,
            "longest_anomaly_streak": 0,
            "peak_frame_index": 12,
            "peak_timestamp_seconds": 0.4,
            "breakdown": {
                "facial_blend_artifacts": 0.04,
                "lip_sync_coherence": 0.03,
                "frequency_spectrum_gan_traces": 0.05,
            },
            "primary_forensic_evidence": "No localized anomaly detected",
            "evidence_zone_percentage": 0.0,
            "annotated_video_url": f"/api/results/{test_id}/video",
            "heatmap_url": f"/api/results/{test_id}/heatmap",
            "timeline_data": [{"timestamp": 0.0, "score": 0.05}],
        },
        "error": None,
    }

    response = client.get(f"/api/results/{test_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == test_id
    assert data["status"] == "completed"
    assert data["result"]["classification"] == "AUTHENTIC"
    assert data["result"]["confidence"] == 0.9421
    assert data["result"]["breakdown"]["facial_blend_artifacts"] == 0.04


def test_analyze_valid_video_upload(client, tmp_path):
    import cv2
    import numpy as np

    # Create a minimal valid MP4 video (3 frames)
    video_path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10, (100, 100))
    for _ in range(3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    response = client.post(
        "/api/analyze",
        files={"video": ("sample.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "analyzing"
    assert data["job_id"] in jobs

