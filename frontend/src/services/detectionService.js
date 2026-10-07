import { apiClient } from '../api/api';

/**
 * Allowed MIME types and extensions for client-side pre-validation.
 */
export const ALLOWED_MIME_TYPES = [
  'video/mp4',
  'video/quicktime',
  'video/webm',
  'video/x-msvideo',
];

export const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.webm', '.avi'];

export const DEFAULT_MAX_FILE_SIZE_MB = 100;

/**
 * Validates a selected file against MIME allowlist and size cap before network transmission.
 * @param {File} file
 * @returns {{ valid: boolean, error?: string }}
 */
export function validateVideoFile(file) {
  if (!file) {
    return { valid: false, error: 'No video file provided for analysis.' };
  }

  // File size validation (100 MB hard cap)
  const maxMb = Number(import.meta.env?.VITE_MAX_FILE_SIZE_MB) || DEFAULT_MAX_FILE_SIZE_MB;
  const maxBytes = maxMb * 1024 * 1024;
  if (file.size > maxBytes) {
    return {
      valid: false,
      error: `File size exceeds the ${maxMb} MB hard limit (selected: ${(file.size / (1024 * 1024)).toFixed(1)} MB).`,
    };
  }

  if (file.size === 0) {
    return { valid: false, error: 'Selected file is empty (0 bytes).' };
  }

  // MIME type validation
  const fileType = file.type?.toLowerCase();
  const fileName = file.name?.toLowerCase() || '';
  const fileExt = fileName.includes('.') ? `.${fileName.split('.').pop()}` : '';

  const mimeMatches = fileType && ALLOWED_MIME_TYPES.includes(fileType);
  const extMatches = fileExt && ALLOWED_EXTENSIONS.includes(fileExt);

  if (!mimeMatches && !extMatches) {
    return {
      valid: false,
      error: `Unsupported file format. Please upload an MP4, MOV, WebM, or AVI video.`,
    };
  }

  return { valid: true };
}

/**
 * Generates a mock forensic response simulating either deepfake or authentic detection.
 * @param {string} filename
 * @returns {object}
 */
export function generateMockResult(filename = 'test_video.mp4') {
  const isDeepfake = Math.random() > 0.45; // 55% chance of deepfake in mock mode
  const confidence = isDeepfake
    ? +(0.82 + Math.random() * 0.16).toFixed(4)
    : +(0.88 + Math.random() * 0.10).toFixed(4);

  return {
    classification: isDeepfake ? 'DEEPFAKE' : 'AUTHENTIC',
    confidence,
    analysis_duration_seconds: +(3.2 + Math.random() * 2.5).toFixed(2),
    frames_analyzed: 145,
    total_frames: 180,
    trimmed_mean_fake_score: isDeepfake ? 0.7642 : 0.0821,
    flagged_frame_ratio: isDeepfake ? 0.683 : 0.021,
    longest_anomaly_streak: isDeepfake ? 42 : 0,
    peak_frame_index: isDeepfake ? 94 : 12,
    peak_timestamp_seconds: isDeepfake ? 3.13 : 0.40,
    breakdown: {
      facial_blend_artifacts: isDeepfake ? 0.8412 : 0.0421,
      lip_sync_coherence: isDeepfake ? 0.7291 : 0.0315,
      frequency_spectrum_gan_traces: isDeepfake ? 0.9104 : 0.0512,
    },
    primary_forensic_evidence: isDeepfake
      ? 'Lower Face / Mouth Cavity (Lip-Sync or Jawline Seam)'
      : 'No localized anomaly detected',
    evidence_zone_percentage: isDeepfake ? 78.4 : 0.0,
    annotated_video_url: null,
    heatmap_url: null,
    timeline_data: Array.from({ length: 30 }, (_, i) => ({
      timestamp: +(i * 0.1).toFixed(2),
      score: isDeepfake
        ? Math.min(1, Math.max(0, +(0.7 + Math.sin(i / 3) * 0.25).toFixed(4)))
        : Math.min(1, Math.max(0, +(0.05 + Math.random() * 0.08).toFixed(4))),
    })),
  };
}

/**
 * Service Layer: Dispatches video upload to /api/analyze with real-time progress & cancellation.
 * @param {File} file
 * @param {object} options
 * @param {(progress: number) => void} [options.onProgress]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{ job_id: string, status: string, message: string }>}
 */
export async function uploadVideoForAnalysis(file, { onProgress, signal } = {}) {
  // Pre-flight client-side input validation
  const validation = validateVideoFile(file);
  if (!validation.valid) {
    const error = new Error(validation.error);
    error.category = 'client';
    error.statusCode = 400;
    throw error;
  }

  // Mock API branch
  const isMock = import.meta.env?.VITE_MOCK_API === 'true';
  if (isMock) {
    return runMockUploadFlow(file, { onProgress, signal });
  }

  // Real API multipart upload
  const formData = new FormData();
  formData.append('video', file);

  const response = await apiClient.post('/api/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    signal,
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(Math.min(percent, 100));
      }
    },
  });

  return response.data;
}

/**
 * Polls /api/results/{jobId} to check status or retrieve finalized results.
 * @param {string} jobId
 * @param {AbortSignal} [signal]
 * @returns {Promise<{ job_id: string, status: string, result?: object, error?: string }>}
 */
export async function getAnalysisResult(jobId, signal) {
  const isMock = import.meta.env?.VITE_MOCK_API === 'true';
  if (isMock) {
    return {
      job_id: jobId,
      status: 'completed',
      result: generateMockResult(),
    };
  }

  const response = await apiClient.get(`/api/results/${jobId}`, { signal });
  return response.data;
}

/**
 * Simulates multipart upload progress and mock response in mock mode.
 */
async function runMockUploadFlow(file, { onProgress, signal }) {
  const steps = [15, 35, 60, 85, 100];
  for (const percent of steps) {
    if (signal?.aborted) {
      const err = new Error('Upload cancelled by user.');
      err.category = 'aborted';
      throw err;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
    if (onProgress) onProgress(percent);
  }

  const mockJobId = 'mock_' + Math.random().toString(36).substring(2, 10);
  return {
    job_id: mockJobId,
    status: 'analyzing',
    message: 'Mock video uploaded successfully. Analysis started.',
  };
}
