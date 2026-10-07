import React from 'react';
import { AlertTriangle, ServerCrash, WifiOff, X, RefreshCw } from 'lucide-react';
import { useDetection } from '../context/DetectionContext';

export default function ErrorAlert() {
  const { error, resetDetection, startAnalysis, state } = useDetection();

  if (state !== 'error' || !error) return null;

  const isNetwork = error.category === 'network';
  const isServer = error.category === 'server';

  return (
    <div style={{ maxWidth: '900px', margin: '20px auto 0', padding: '0 20px' }}>
      <div style={{
        padding: '16px 20px',
        borderRadius: '12px',
        backgroundColor: 'rgba(239, 68, 68, 0.12)',
        border: '1px solid rgba(239, 68, 68, 0.35)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ color: 'var(--fake-red)' }}>
            {isNetwork ? <WifiOff size={22} /> : isServer ? <ServerCrash size={22} /> : <AlertTriangle size={22} />}
          </div>
          <div>
            <div style={{ fontSize: '14px', fontWeight: '700', color: '#fff' }}>
              {isNetwork ? 'Network Connection Failure' : isServer ? 'Server Pipeline Error' : 'Validation Error'}
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              {error.message}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {isServer && (
            <button
              onClick={startAnalysis}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 14px',
                borderRadius: '6px',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                color: '#fff',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              <RefreshCw size={12} /> Retry
            </button>
          )}
          <button
            onClick={resetDetection}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              border: '1px solid var(--border-color)',
              backgroundColor: 'var(--bg-card)',
              color: 'var(--text-primary)',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            <X size={12} /> Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
