import { http, HttpResponse } from 'msw';
import { generateMockResult } from '../services/detectionService';

const baseURL = import.meta.env?.VITE_API_URL || 'http://localhost:8000';

export const handlers = [
  // 1. Health check
  http.get(`${baseURL}/api/health`, () => {
    return HttpResponse.json({
      status: 'ok',
      model_loaded: true,
      device: 'cpu',
    });
  }),

  // 2. Upload video
  http.post(`${baseURL}/api/analyze`, async ({ request }) => {
    const formData = await request.formData();
    const videoFile = formData.get('video');

    if (!videoFile) {
      return HttpResponse.json(
        { detail: 'No video file provided.' },
        { status: 422 }
      );
    }

    // Simulate 500 server error trigger for error testing
    if (typeof videoFile.name === 'string' && videoFile.name.includes('server-error')) {
      return HttpResponse.json(
        { detail: 'Internal ML pipeline execution error.' },
        { status: 500 }
      );
    }

    const mockJobId = 'msw_' + Math.random().toString(36).substring(2, 9);
    return HttpResponse.json(
      {
        job_id: mockJobId,
        status: 'analyzing',
        message: 'Video uploaded successfully. Analysis started.',
      },
      { status: 202 }
    );
  }),

  // 3. Poll analysis status & results
  http.get(`${baseURL}/api/results/:jobId`, ({ params }) => {
    const { jobId } = params;

    if (jobId === 'invalid-job') {
      return HttpResponse.json(
        { detail: 'Job not found.' },
        { status: 404 }
      );
    }

    const isFake = !jobId.includes('authentic');
    const mockResult = generateMockResult();
    mockResult.classification = isFake ? 'DEEPFAKE' : 'AUTHENTIC';
    mockResult.confidence = isFake ? 0.8924 : 0.9412;

    return HttpResponse.json({
      job_id: jobId,
      status: 'completed',
      result: mockResult,
    });
  }),
];
