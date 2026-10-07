import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadSection from './components/UploadSection';
import ForensicDashboard from './components/ForensicDashboard';
import HistoryDrawer from './components/HistoryDrawer';
import ErrorAlert from './components/ErrorAlert';
import { useDetection } from './context/DetectionContext';

function MainContent() {
  const { state, history } = useDetection();
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        onOpenHistory={() => setIsHistoryOpen(true)}
        historyCount={history.length}
      />

      <main style={{ flex: 1, paddingBottom: '60px' }}>
        <ErrorAlert />

        {state === 'completed' ? (
          <ForensicDashboard />
        ) : (
          <UploadSection />
        )}
      </main>

      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
      />

      <footer style={{
        borderTop: '1px solid var(--border-color)',
        padding: '24px 32px',
        textAlign: 'center',
        fontSize: '12px',
        color: 'var(--text-muted)',
        backgroundColor: 'var(--bg-secondary)'
      }}>
        Maya Forensic AI • Dual-Stream Xception & High-Pass Laplacian Spatial-Temporal Detection System
      </footer>
    </div>
  );
}

export default function App() {
  return <MainContent />;
}
