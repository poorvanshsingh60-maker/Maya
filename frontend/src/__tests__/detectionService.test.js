import { describe, it, expect, vi } from 'vitest';
import { uploadVideoForAnalysis, getAnalysisResult, generateMockResult } from '../services/detectionService';

describe('Detection Service Layer (detectionService.js)', () => {
  it('generates a valid forensic result schema in mock mode', () => {
    const mock = generateMockResult('sample.mp4');

    expect(['DEEPFAKE', 'AUTHENTIC']).toContain(mock.classification);
    expect(mock.confidence).toBeGreaterThanOrEqual(0);
    expect(mock.confidence).toBeLessThanOrEqual(1);
    expect(mock.frames_analyzed).toBeGreaterThan(0);
    expect(mock.breakdown).toBeDefined();
    expect(typeof mock.breakdown.facial_blend_artifacts).toBe('number');
    expect(typeof mock.breakdown.lip_sync_coherence).toBe('number');
    expect(typeof mock.breakdown.frequency_spectrum_gan_traces).toBe('number');
    expect(Array.isArray(mock.timeline_data)).toBe(true);
    expect(mock.timeline_data.length).toBeGreaterThan(0);
  });

  it('rejects invalid files before network transmission', async () => {
    const textFile = new File(['bad'], 'bad.txt', { type: 'text/plain' });

    await expect(uploadVideoForAnalysis(textFile)).rejects.toThrow(/Unsupported file format/);
  });

  it('simulates upload progress in mock flow', async () => {
    const validFile = new File(['valid_bytes'], 'video.mp4', { type: 'video/mp4' });
    const progressList = [];

    // Temporarily ensure VITE_MOCK_API is 'true'
    vi.stubEnv('VITE_MOCK_API', 'true');

    const result = await uploadVideoForAnalysis(validFile, {
      onProgress: (p) => progressList.push(p),
    });

    expect(result.job_id).toBeDefined();
    expect(result.status).toBe('analyzing');
    expect(progressList.length).toBeGreaterThan(0);
    expect(progressList[progressList.length - 1]).toBe(100);

    vi.unstubAllEnvs();
  });

  it('aborts upload cleanly when signal is aborted', async () => {
    const validFile = new File(['valid_bytes'], 'video.mp4', { type: 'video/mp4' });
    const controller = new AbortController();

    vi.stubEnv('VITE_MOCK_API', 'true');

    // Abort immediately
    controller.abort();

    await expect(
      uploadVideoForAnalysis(validFile, { signal: controller.signal })
    ).rejects.toThrow(/cancelled/i);

    vi.unstubAllEnvs();
  });

  it('retrieves completed status and forensic result in mock mode', async () => {
    vi.stubEnv('VITE_MOCK_API', 'true');

    const res = await getAnalysisResult('mock_123');
    expect(res.job_id).toBe('mock_123');
    expect(res.status).toBe('completed');
    expect(res.result).toBeDefined();
    expect(res.result.breakdown).toBeDefined();

    vi.unstubAllEnvs();
  });
});
