import React, { useEffect, useState } from 'react';
import { ShieldAlert, ShieldCheck, History, Activity } from 'lucide-react';
import { apiClient } from '../api/api';

export default function Navbar({ onOpenHistory, historyCount }) {
  const [serverHealth, setServerHealth] = useState({ online: false, device: 'unknown' });
  const isMock = import.meta.env.VITE_MOCK_API === 'true';

  useEffect(() => {
    if (isMock) {
      setServerHealth({ online: true, device: 'mock-engine' });
      return;
    }

    const checkHealth = async () => {
      try {
        const res = await apiClient.get('/api/health');
        if (res.data?.status === 'ok') {
          setServerHealth({ online: true, device: res.data.device || 'active' });
        } else {
          setServerHealth({ online: false, device: 'unavailable' });
        }
      } catch {
        setServerHealth({ online: false, device: 'offline' });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [isMock]);

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '16px 32px',
      borderBottom: '1px solid var(--border-color)',
      backgroundColor: 'var(--bg-secondary)',
      position: 'sticky',
      top: 0,
      zIndex: 50
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '10px',
          backgroundColor: 'rgba(59, 130, 246, 0.15)',
          border: '1px solid rgba(59, 130, 246, 0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--accent-blue)'
        }}>
          <ShieldAlert size={24} />
        </div>
        <div>
          <h1 style={{ fontSize: '20px', fontWeight: '700', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
            MAYA <span style={{ fontSize: '11px', fontWeight: '600', padding: '2px 8px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border-color)', color: 'var(--accent-cyan)' }}>FORENSIC AI</span>
          </h1>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Dual-Stream Deepfake Detection & Spatial-Temporal XAI</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Backend / Mock Status Indicator */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          borderRadius: '20px',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-color)',
          fontSize: '12px',
          fontFamily: 'var(--font-mono)'
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: serverHealth.online ? '#10b981' : '#ef4444',
            boxShadow: serverHealth.online ? '0 0 8px #10b981' : 'none'
          }} />
          <span style={{ color: 'var(--text-secondary)' }}>
            {isMock ? 'MOCK API' : `API: ${serverHealth.online ? 'ONLINE (' + serverHealth.device + ')' : 'OFFLINE'}`}
          </span>
        </div>

        {/* History Ring Buffer Button */}
        <button
          onClick={onOpenHistory}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 16px',
            borderRadius: '8px',
            backgroundColor: 'var(--bg-card)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-color)',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '500',
            transition: 'background 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card)'}
        >
          <History size={16} />
          <span>Audit Log</span>
          {historyCount > 0 && (
            <span style={{
              padding: '1px 6px',
              fontSize: '11px',
              fontWeight: '700',
              borderRadius: '10px',
              backgroundColor: 'var(--accent-blue)',
              color: '#fff'
            }}>
              {historyCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
