import React, { useRef, useState } from 'react';
import { UploadCloud, FileVideo, X, Play, RefreshCw, AlertCircle, Cpu } from 'lucide-react';
import { useDetection } from '../context/DetectionContext';

export default function UploadSection() {
  const {
    state,
    selectedFile,
    previewUrl,
    uploadProgress,
    selectFile,
    startAnalysis,
    cancelAnalysis,
    resetDetection,
  } = useDetection();

  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      selectFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      selectFile(e.target.files[0]);
    }
  };

  const isUploading = state === 'uploading';
  const isAnalyzing = state === 'analyzing';

  return (
    <section style={{ maxWidth: '900px', margin: '40px auto', padding: '0 20px' }}>
      {/* Hero Title */}
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <h2 style={{ fontSize: '32px', fontWeight: '800', letterSpacing: '-0.03em', marginBottom: '8px' }}>
          Autonomous Deepfake Video Forensics
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '15px', maxWidth: '600px', margin: '0 auto' }}>
          Upload high-definition video files for frame-by-frame dual-stream spectral analysis, Grad-CAM spatial attribution, and temporal anomaly persistence tracking.
        </p>
      </div>

      {/* Main Upload / Inspection Card */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '16px',
        border: '1px solid var(--border-color)',
        padding: '32px',
        boxShadow: '0 20px 40px -15px rgba(0, 0, 0, 0.5)'
      }}>
        {!selectedFile ? (
          // Drag & Drop Dropzone
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${isDragOver ? 'var(--accent-blue)' : 'var(--border-color)'}`,
              borderRadius: '12px',
              padding: '60px 24px',
              textAlign: 'center',
              backgroundColor: isDragOver ? 'rgba(59, 130, 246, 0.05)' : 'var(--bg-card)',
              cursor: 'pointer',
              transition: 'all 0.2s ease-in-out'
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,.mp4,.mov,.webm,.avi"
              style={{ display: 'none' }}
            />
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '50%',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-blue)',
              margin: '0 auto 16px'
            }}>
              <UploadCloud size={32} />
            </div>
            <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
              Drag & Drop your video here, or <span style={{ color: 'var(--accent-blue)', textDecoration: 'underline' }}>Browse</span>
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '16px' }}>
              Supported formats: MP4, MOV, WebM, AVI • Hard limit: 100 MB
            </p>
            <div style={{ display: 'inline-flex', gap: '8px' }}>
              {['MP4', 'MOV', 'WEBM', 'AVI'].map((ext) => (
                <span key={ext} style={{
                  fontSize: '11px',
                  fontWeight: '600',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}>
                  {ext}
                </span>
              ))}
            </div>
          </div>
        ) : (
          // Video Preview & Controller
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <FileVideo size={20} color="var(--accent-blue)" />
                <div>
                  <h4 style={{ fontSize: '15px', fontWeight: '600' }}>{selectedFile.name}</h4>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || 'video'}
                  </p>
                </div>
              </div>
              {state === 'idle' && (
                <button
                  onClick={resetDetection}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 12px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    background: 'transparent',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  <X size={14} /> Remove
                </button>
              )}
            </div>

            {/* Video Player */}
            <div style={{
              borderRadius: '12px',
              overflow: 'hidden',
              backgroundColor: '#000',
              maxHeight: '420px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '24px',
              border: '1px solid var(--border-color)',
              position: 'relative'
            }}>
              <video
                src={previewUrl}
                controls={state === 'idle'}
                style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }}
              />

              {/* Scanning Overlay during Analysis */}
              {isAnalyzing && (
                <div style={{
                  position: 'absolute',
                  inset: 0,
                  backgroundColor: 'rgba(10, 13, 20, 0.75)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '16px',
                  backdropFilter: 'blur(3px)'
                }}>
                  <div style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '50%',
                    border: '3px solid rgba(59, 130, 246, 0.2)',
                    borderTopColor: 'var(--accent-blue)',
                    animation: 'spin 1s linear infinite'
                  }} />
                  <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
                  <div style={{ textAlign: 'center' }}>
                    <h4 style={{ fontSize: '16px', fontWeight: '600', color: '#fff', marginBottom: '4px' }}>
                      Running Deep Forensic Pipeline…
                    </h4>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                      MTCNN Tracking → Dual-Stream Inference → Grad-CAM Attribution
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Upload Progress Bar */}
            {isUploading && (
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '8px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Streaming video chunks to server…</span>
                  <span style={{ fontWeight: '600', color: 'var(--accent-blue)' }}>{uploadProgress}%</span>
                </div>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: 'var(--bg-card)',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{
                    width: `${uploadProgress}%`,
                    height: '100%',
                    backgroundColor: 'var(--accent-blue)',
                    transition: 'width 0.2s ease-out'
                  }} />
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              {(isUploading || isAnalyzing) ? (
                <button
                  onClick={cancelAnalysis}
                  style={{
                    padding: '12px 24px',
                    borderRadius: '8px',
                    border: '1px solid rgba(239, 68, 68, 0.4)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    color: 'var(--fake-red)',
                    fontWeight: '600',
                    fontSize: '14px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <X size={16} /> Cancel Analysis
                </button>
              ) : (
                <button
                  onClick={startAnalysis}
                  style={{
                    padding: '12px 28px',
                    borderRadius: '8px',
                    border: 'none',
                    backgroundColor: 'var(--accent-blue)',
                    color: '#fff',
                    fontWeight: '600',
                    fontSize: '14px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    boxShadow: '0 4px 14px rgba(59, 130, 246, 0.35)',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--accent-blue-hover)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--accent-blue)'}
                >
                  <Cpu size={18} /> Run Forensic Analysis
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
