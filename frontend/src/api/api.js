import axios from 'axios';

/**
 * 12-factor Axios client instance.
 * Base URL injected from VITE_API_URL environment variable.
 * Default timeout: 60,000ms (60 seconds) to allow initial multipart uploads.
 */
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL,
  timeout: 60000,
  headers: {
    'Accept': 'application/json',
  },
});

/**
 * Global response interceptor that normalizes HTTP errors before surfacing to UI.
 * Categorizes errors into:
 * - 'server' (5xx): transient/server-side failure
 * - 'network': connection dropped or refused (no response)
 * - 'client' (4xx): client-side or validation rejection
 * - 'aborted': user cancelled request via AbortController
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Check if cancellation
    if (axios.isCancel(error) || error.name === 'CanceledError' || error.code === 'ERR_CANCELED') {
      const normalized = new Error('Upload cancelled by user.');
      normalized.name = 'AbortError';
      normalized.category = 'aborted';
      return Promise.reject(normalized);
    }

    let message = 'An unexpected error occurred.';
    let category = 'unknown';
    let statusCode = null;

    if (error.response) {
      // Server responded with non-2xx status code
      statusCode = error.response.status;
      const detail = error.response.data?.detail || error.response.data?.message;

      if (statusCode >= 500) {
        category = 'server';
        message = detail || 'Internal server error occurred during video analysis.';
      } else if (statusCode === 413) {
        category = 'client';
        message = detail || 'Uploaded video exceeds the maximum file size limit (100 MB).';
      } else if (statusCode === 415) {
        category = 'client';
        message = detail || 'Unsupported video file format or MIME type.';
      } else {
        category = 'client';
        message = detail || `Request failed with status code ${statusCode}.`;
      }
    } else if (error.request) {
      // Request made but no response received (network failure, timeout, CORS error)
      category = 'network';
      if (error.code === 'ECONNABORTED') {
        message = 'Request timed out after 60 seconds. Please check your network connection.';
      } else {
        message = 'Unable to connect to the backend server. Please ensure the API is running.';
      }
    } else {
      message = error.message || 'Error occurred while dispatching request.';
    }

    const normalizedError = new Error(message);
    normalizedError.category = category;
    normalizedError.statusCode = statusCode;
    normalizedError.originalError = error;

    return Promise.reject(normalizedError);
  }
);

export default apiClient;
