import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Layers,
  Clock,
  Flame,
  Camera,
  RefreshCw,
  Eye,
  Crosshair,
  BarChart3,
  Video
} from 'lucide-react';
import { useDetection } from '../context/DetectionContext';

export default function ForensicDashboard() {
  const { result, selectedFile, resetDetection } = useDetection();
  const [activeMediaTab, setActiveMediaTab] = useState('heatmap'); // 'heatmap' | 'annotated' | 'original'

  if (!result) return null;

  const isFake = result.classification === 'DEEPFAKE';
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const annotatedVideoUrl = result.annotated_video_url ? `${baseURL}${result.annotated_video_url}` : null;
  const heatmapUrl = result.heatmap_url ? `${baseURL}${result.heatmap_url}` : null;

  return (
    <div style={{ maxWidth: '1100px', margin: '40px auto', padding: '0 20px' }}>
      {/* 1. Executive Verdict Banner */}
      <div style={{
        backgroundColor: isFake ? 'var(--fake-red-bg)' : 'var(--real-green-bg)',
        border: `1px solid ${isFake ? 'var(--fake-red-border)' : 'var(--real-green-border)'}`,
        borderRadius: '16px',
        padding: '32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '24px',
        marginBottom: '32px',
        boxShadow: isFake
          ? '0 10px 30px rgba(239, 68, 68, 0.15)'
          : '0 10px 30px rgba(16, 185, 129, 0.15)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '16px',
            backgroundColor: isFake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: isFake ? 'var(--fake-red)' : 'var(--real-green)'
          }}>
            {isFake ? <ShieldAlert size={36} /> : <ShieldCheck size={36} />}
          </div>
          <div>
            <div style={{
              display: 'inline-block',
              fontSize: '12px',
              fontWeight: '700',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: isFake ? 'var(--fake-red)' : 'var(--real-green)',
              marginBottom: '4px'
            }}>
              Final Forensic Classification
            </div>
            <h2 style={{ fontSize: '28px', fontWeight: '800', letterSpacing: '-0.02em', color: '#fff' }}>
              {isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC MEDIA'}
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>
              File: <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>{selectedFile?.name}</span>
            </p>
          </div>
        </div>

        {/* Confidence Gauge */}
        <div style={{
          textAlign: 'right',
          backgroundColor: 'var(--bg-card)',
          padding: '16px 24px',
          borderRadius: '12px',
          border: '1px solid var(--border-color)'
        }}>
          <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Forensic Confidence
          </div>
          <div style={{
            fontSize: '36px',
            fontWeight: '800',
            fontFamily: 'var(--font-mono)',
            color: isFake ? 'var(--fake-red)' : 'var(--real-green)'
          }}>
            {confidencePercent}%
          </div>
        </div>
      </div>

      {/* 2. Top-Level Metric Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px',
        marginBottom: '32px'
      }}>
        {[
          { label: 'Frames Analyzed', val: `${result.frames_analyzed} / ${result.total_frames}`, icon: <Camera size={18} /> },
          { label: 'Execution Time', val: `${result.analysis_duration_seconds}s`, icon: <Clock size={18} /> },
          { label: 'Trimmed Mean Fake', val: `${(result.trimmed_mean_fake_score * 100).toFixed(1)}%`, icon: <BarChart3 size={18} /> },
          { label: 'Flagged Frame Ratio', val: `${(result.flagged_frame_ratio * 100).toFixed(1)}%`, icon: <Activity size={18} /> },
          { label: 'Max Anomaly Streak', val: `${result.longest_anomaly_streak} frames`, icon: <Flame size={18} /> },
        ].map((item, idx) => (
          <div key={idx} style={{
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '12px',
            padding: '18px 20px',
            border: '1px solid var(--border-color)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', fontWeight: '500' }}>{item.label}</span>
              <span style={{ color: 'var(--accent-blue)' }}>{item.icon}</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: '700', fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {item.val}
            </div>
          </div>
        ))}
      </div>

      {/* 3. Forensic Signal Decomposition & Primary Evidence */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: '24px',
        marginBottom: '32px'
      }}>
        {/* Sub-Model Signals */}
        <div style={{
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '16px',
          border: '1px solid var(--border-color)',
          padding: '24px'
        }}>
          <h3 style={{ fontSize: '16px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
            <Layers size={18} color="var(--accent-cyan)" /> Forensic Sub-Signal Decomposition
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {[
              {
                title: 'Facial Blend Artifacts',
                desc: 'Spatial seam boundary & Grad-CAM attribution',
                val: result.breakdown?.facial_blend_artifacts || 0
              },
              {
                title: 'Lip-Sync / Lower Face Coherence',
                desc: 'Mouth cavity anomaly localization',
                val: result.breakdown?.lip_sync_coherence || 0
              },
              {
                title: 'Frequency Spectrum GAN Traces',
                desc: 'Laplacian high-pass micro-artifact energy',
                val: result.breakdown?.frequency_spectrum_gan_traces || 0
              },
            ].map((sig, idx) => (
              <div key={idx}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '4px' }}>
                  <span style={{ fontWeight: '600' }}>{sig.title}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    {(sig.val * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{
                  width: '100%',
                  height: '6px',
                  backgroundColor: 'var(--bg-card)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginBottom: '4px'
                }}>
                  <div style={{
                    width: `${Math.min(100, sig.val * 100)}%`,
                    height: '100%',
                    backgroundColor: sig.val > 0.5 ? 'var(--fake-red)' : 'var(--accent-blue)',
                    borderRadius: '3px'
                  }} />
                </div>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{sig.desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Primary Evidence Zone */}
        <div style={{
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '16px',
          border: '1px solid var(--border-color)',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between'
        }}>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
              <Crosshair size={18} color="var(--fake-red)" /> Localized Forensic Anomaly Zone
            </h3>

            <div style={{
              padding: '16px',
              borderRadius: '12px',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              marginBottom: '16px'
            }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
                Primary Landmark Attribution
              </div>
              <div style={{ fontSize: '16px', fontWeight: '700', color: isFake ? 'var(--fake-red)' : 'var(--real-green)' }}>
                {result.primary_forensic_evidence}
              </div>
              {result.evidence_zone_percentage > 0 && (
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Anomalous area occupies <span style={{ color: '#fff', fontWeight: '600' }}>{result.evidence_zone_percentage}%</span> of facial bounding box
                </div>
              )}
            </div>

            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              {isFake
                ? `Peak artifact detected at timestamp ${result.peak_timestamp_seconds}s (Frame #${result.peak_frame_index}). The dual-stream network flagged localized frequency seams and spatial inconsistencies.`
                : 'Facial landmarks demonstrate smooth Laplacian gradient continuity and natural micro-texture distributions without GAN synthesis seams.'}
            </p>
          </div>

          <div style={{ marginTop: '20px' }}>
            <button
              onClick={resetDetection}
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                backgroundColor: 'var(--bg-card)',
                color: 'var(--text-primary)',
                fontWeight: '600',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'background 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card)'}
            >
              <RefreshCw size={14} /> Analyze Another Video
            </button>
          </div>
        </div>
      </div>

      {/* 4. Forensic Visual Inspection Deck */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '16px',
        border: '1px solid var(--border-color)',
        padding: '24px',
        marginBottom: '32px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '16px',
          marginBottom: '20px',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <h3 style={{ fontSize: '16px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Eye size={18} color="var(--accent-blue)" /> Visual Forensic Inspection
          </h3>

          {/* Media Switcher Tabs */}
          <div style={{ display: 'flex', gap: '8px' }}>
            {[
              { id: 'heatmap', label: 'Grad-CAM Heatmap' },
              { id: 'annotated', label: 'Annotated Video' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveMediaTab(tab.id)}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: '600',
                  border: activeMediaTab === tab.id ? '1px solid var(--accent-blue)' : '1px solid var(--border-color)',
                  backgroundColor: activeMediaTab === tab.id ? 'rgba(59, 130, 246, 0.15)' : 'var(--bg-card)',
                  color: activeMediaTab === tab.id ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  cursor: 'pointer'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Active Media Container */}
        <div style={{
          minHeight: '360px',
          backgroundColor: '#000',
          borderRadius: '12px',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid var(--border-color)'
        }}>
          {activeMediaTab === 'heatmap' ? (
            heatmapUrl ? (
              <img
                src={heatmapUrl}
                alt="Peak Grad-CAM Heatmap"
                style={{ maxWidth: '100%', maxHeight: '450px', objectFit: 'contain' }}
              />
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px' }}>
                <p>Grad-CAM spatial heatmap asset is available on the live backend stream.</p>
              </div>
            )
          ) : (
            annotatedVideoUrl ? (
              <video
                src={annotatedVideoUrl}
                controls
                style={{ width: '100%', maxHeight: '450px', objectFit: 'contain' }}
              />
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px' }}>
                <p>Annotated stream is available on live backend evaluation.</p>
              </div>
            )
          )}
        </div>
      </div>

      {/* 5. Temporal Anomaly Timeline */}
      {result.timeline_data && result.timeline_data.length > 0 && (
        <div style={{
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '16px',
          border: '1px solid var(--border-color)',
          padding: '24px'
        }}>
          <h3 style={{ fontSize: '16px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Activity size={18} color="var(--accent-cyan)" /> Frame-by-Frame Temporal Timeline
          </h3>

          {/* Simple SVG Chart */}
          <div style={{ height: '140px', width: '100%', position: 'relative' }}>
            <svg viewBox="0 0 500 120" preserveAspectRatio="none" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
              {/* Threshold Line at 0.52 (Score = 52%) */}
              <line x1="0" y1="58" x2="500" y2="58" stroke="rgba(239, 68, 68, 0.4)" strokeDasharray="4" strokeWidth="1" />
              <text x="4" y="54" fill="rgba(239, 68, 68, 0.7)" fontSize="9">Threshold 52%</text>

              {/* Manipulation Curve */}
              <polyline
                fill="none"
                stroke={isFake ? '#ef4444' : '#10b981'}
                strokeWidth="2.5"
                points={result.timeline_data
                  .map((pt, i) => {
                    const x = (i / (result.timeline_data.length - 1)) * 500;
                    const y = 120 - pt.score * 120;
                    return `${x},${y}`;
                  })
                  .join(' ')}
              />
            </svg>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
            <span>0.0s (Start)</span>
            <span>Duration: {result.analysis_duration_seconds}s</span>
          </div>
        </div>
      )}
    </div>
  );
}
