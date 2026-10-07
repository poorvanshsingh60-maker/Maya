import React from 'react';
import { X, Clock, FileVideo, ShieldAlert, ShieldCheck } from 'lucide-react';
import { useDetection } from '../context/DetectionContext';

export default function HistoryDrawer({ isOpen, onClose }) {
  const { history } = useDetection();

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      backdropFilter: 'blur(4px)',
      display: 'flex',
      justifyContent: 'flex-end',
      zIndex: 100
    }}>
      <div style={{
        width: '100%',
        maxWidth: '460px',
        backgroundColor: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border-color)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '-10px 0 30px rgba(0, 0, 0, 0.5)'
      }}>
        {/* Header */}
        <div style={{
          padding: '20px 24px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: '700' }}>Session Audit History</h3>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              In-memory ring buffer (capped at 50 recent runs)
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: '6px'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* List of Entries */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 24px' }}>
          {history.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '80px' }}>
              <Clock size={36} style={{ margin: '0 auto 12px', opacity: 0.5 }} />
              <p>No detection runs recorded in this session yet.</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {history.map((item) => {
                const isFake = item.result.classification === 'DEEPFAKE';
                return (
                  <div
                    key={item.id}
                    style={{
                      padding: '16px',
                      borderRadius: '12px',
                      backgroundColor: 'var(--bg-card)',
                      border: '1px solid var(--border-color)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '12px',
                        fontWeight: '700',
                        color: isFake ? 'var(--fake-red)' : 'var(--real-green)'
                      }}>
                        {isFake ? <ShieldAlert size={14} /> : <ShieldCheck size={14} />}
                        {item.result.classification}
                      </span>
                      <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {(item.result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                      {item.filename}
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>{(item.fileSizeBytes / (1024 * 1024)).toFixed(1)} MB</span>
                      <span>{new Date(item.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
