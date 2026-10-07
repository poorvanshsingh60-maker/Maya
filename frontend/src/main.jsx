import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { DetectionProvider } from './context/DetectionContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <DetectionProvider>
      <App />
    </DetectionProvider>
  </React.StrictMode>
);
