import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import { uploadVideoForAnalysis, getAnalysisResult, validateVideoFile } from '../services/detectionService';

const DetectionContext = createContext(null);

const MAX_HISTORY_ITEMS = 50;
const POLL_INTERVAL_MS = 2000;

export function DetectionProvider({ children }) {
  // 5 discrete states: 'idle' | 'uploading' | 'analyzing' | 'completed' | 'error'
  const [state, setState] = useState('idle');

  // Active File & Preview Object URL
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Upload & Analysis Progress / Job Info
  const [uploadProgress, setUploadProgress] = useState(0);
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // In-Memory Ring Buffer for Session History (Capped at 50 entries)
  const [history, setHistory] = useState([]);

  // Shared useRef-backed AbortController for concurrent-safe cancellation
  const abortControllerRef = useRef(null);
  const pollTimerRef = useRef(null);

  // Revoke object URL helper to prevent browser memory leaks
  const cleanupPreviewUrl = useCallback((url) => {
    if (url && typeof url === 'string' && url.startsWith('blob:')) {
      try {
        URL.revokeObjectURL(url);
      } catch (err) {
        console.warn('Failed to revoke object URL:', err);
      }
    }
  }, []);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
      if (previewUrl) {
        cleanupPreviewUrl(previewUrl);
      }
    };
  }, [cleanupPreviewUrl, previewUrl]);

  /**
   * Sets a new video file for inspection and creates an object URL for video preview.
   */
  const selectFile = useCallback((file) => {
    // Abort any ongoing operations
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    // Revoke previous object URL
    if (previewUrl) {
      cleanupPreviewUrl(previewUrl);
    }

    // Pre-flight check
    const validation = validateVideoFile(file);
    if (!validation.valid) {
      setSelectedFile(null);
      setPreviewUrl(null);
      setError({
        message: validation.error,
        category: 'client',
      });
      setState('error');
      return false;
    }

    const newUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(newUrl);
    setUploadProgress(0);
    setJobId(null);
    setResult(null);
    setError(null);
    setState('idle');
    return true;
  }, [cleanupPreviewUrl, previewUrl]);

  /**
   * Appends a completed analysis to the 50-entry ring buffer.
   */
  const appendToHistory = useCallback((entry) => {
    setHistory((prev) => {
      const updated = [entry, ...prev];
      if (updated.length > MAX_HISTORY_ITEMS) {
        return updated.slice(0, MAX_HISTORY_ITEMS);
      }
      return updated;
    });
  }, []);

  /**
   * Initiates the upload and forensic analysis lifecycle.
   */
  const startAnalysis = useCallback(async () => {
    if (!selectedFile) return;

    // Reset previous controller and establish new one
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setError(null);
    setResult(null);
    setUploadProgress(0);
    setState('uploading');

    try {
      // 1. Upload Phase (tracked via onProgress)
      const uploadData = await uploadVideoForAnalysis(selectedFile, {
        onProgress: (percent) => setUploadProgress(percent),
        signal: controller.signal,
      });

      const currentJobId = uploadData.job_id;
      setJobId(currentJobId);
      setState('analyzing');

      // 2. Polling Phase
      pollTimerRef.current = setInterval(async () => {
        try {
          const pollData = await getAnalysisResult(currentJobId, controller.signal);

          if (pollData.status === 'completed' && pollData.result) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setResult(pollData.result);
            setState('completed');

            // Persist to session ring buffer
            appendToHistory({
              id: currentJobId,
              filename: selectedFile.name,
              fileSizeBytes: selectedFile.size,
              timestamp: new Date().toISOString(),
              result: pollData.result,
            });
          } else if (pollData.status === 'error') {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setError({
              message: pollData.error || 'Deepfake analysis failed.',
              category: 'server',
            });
            setState('error');
          }
        } catch (pollErr) {
          if (controller.signal.aborted) return;
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
          setError({
            message: pollErr.message || 'Error occurred while checking analysis status.',
            category: pollErr.category || 'network',
          });
          setState('error');
        }
      }, POLL_INTERVAL_MS);

    } catch (uploadErr) {
      if (controller.signal.aborted) {
        setState('idle');
        return;
      }
      setError({
        message: uploadErr.message || 'Upload failed.',
        category: uploadErr.category || 'unknown',
      });
      setState('error');
    }
  }, [selectedFile, appendToHistory]);

  /**
   * Cancels in-flight upload or analysis without memory leaks.
   */
  const cancelAnalysis = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setUploadProgress(0);
    setState('idle');
  }, []);

  /**
   * Resets the entire detection state back to pristine idle.
   */
  const resetDetection = useCallback(() => {
    cancelAnalysis();
    if (previewUrl) {
      cleanupPreviewUrl(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setJobId(null);
    setState('idle');
  }, [cancelAnalysis, cleanupPreviewUrl, previewUrl]);

  const value = {
    state,
    selectedFile,
    previewUrl,
    uploadProgress,
    jobId,
    result,
    error,
    history,
    selectFile,
    startAnalysis,
    cancelAnalysis,
    resetDetection,
  };

  return <DetectionContext.Provider value={value}>{children}</DetectionContext.Provider>;
}

export function useDetection() {
  const context = useContext(DetectionContext);
  if (!context) {
    throw new Error('useDetection must be used within a DetectionProvider');
  }
  return context;
}
