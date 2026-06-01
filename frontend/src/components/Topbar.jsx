import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Wifi, Shield, Radio, WifiOff, AlertTriangle, TrendingUp, Activity } from 'lucide-react';
import { useConnectionStatus } from '../hooks/useApi';
import { fetchSystemStatus } from '../utils/apiClient';

const PAGE_TITLES = {
  '/': 'COMMAND DASHBOARD',
  '/nlp': 'SIGINT // EARLY WARNING',
  '/vision': 'GEO-VISION // DAMAGE MAP',
  '/infra': 'INFRA-METRICS // GRID • TRAFFIC • ENV',
  '/dispatch': 'UNIT DISPATCH // TACTICAL',
  '/zones': 'THREAT MATRIX // ZONE ANALYSIS',
};

const DEFCON_COLORS = {
  1: '#ff003c',
  2: '#ff6b00',
  3: '#ffb000',
  4: '#00f3ff',
  5: '#00ff88',
};

export default function Topbar({ systemStatus: wsSysStatus }) {
  const [time, setTime] = useState('');
  const location = useLocation();
  const { mode, healthData } = useConnectionStatus();
  const [polledStatus, setPolledStatus] = useState(null);

  // Use WS-pushed systemStatus if available, fall back to polled
  const sys = wsSysStatus || polledStatus;
  


  useEffect(() => {
    const tick = () => {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, '0');
      const m = String(now.getMinutes()).padStart(2, '0');
      const s = String(now.getSeconds()).padStart(2, '0');
      setTime(`${h}:${m}:${s}`);
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  // Poll system status every 10s as fallback
  useEffect(() => {
    const poll = async () => {
      const data = await fetchSystemStatus();
      if (data) setPolledStatus(data);
    };
    poll();
    const timer = setInterval(poll, 10000);
    return () => clearInterval(timer);
  }, []);

  const defcon = sys?.defcon ?? '—';
  const defconColor = DEFCON_COLORS[defcon] || '#ffb000';
  const flRound = sys?.fl_round ?? '—';
  const epsilon = (sys?.epsilon !== undefined && sys?.epsilon !== null) ? Number(sys.epsilon).toFixed(3) : '—';
  const hetero = (sys?.hetero_index !== undefined && sys?.hetero_index !== null) ? Number(sys.hetero_index).toFixed(3) : '—';
  const divergence = (sys?.divergence !== undefined && sys?.divergence !== null) ? Number(sys.divergence).toFixed(3) : '—';
  const paused = sys?.paused;

  const modeBadge = mode === 'LIVE' ? (
    <div className="flex items-center gap-1.5 px-2 py-0.5 border border-success/50 bg-success/10">
      <Radio size={10} className="text-success animate-pulse" />
      <span className="font-mono text-[10px] text-success tracking-widest">LIVE</span>
    </div>
  ) : mode === 'OFFLINE' ? (
    <div className="flex items-center gap-1.5 px-2 py-0.5 border border-danger/50 bg-danger/10">
      <WifiOff size={10} className="text-danger" />
      <span className="font-mono text-[10px] text-danger tracking-widest">OFFLINE</span>
    </div>
  ) : (
    <div className="flex items-center gap-1.5 px-2 py-0.5 border border-gray-600 bg-gray-900">
      <Wifi size={10} className="text-gray-400 animate-pulse" />
      <span className="font-mono text-[10px] text-gray-400 tracking-widest">CONNECTING</span>
    </div>
  );

  return (
    <div className="h-12 w-full flex items-center justify-between px-6 z-20 relative" style={{ background: 'rgba(5,5,8,0.9)', borderBottom: '1px solid rgba(0,243,255,0.08)' }}>
      <div className="flex items-center gap-5">
        {modeBadge}
        <div className="h-3 w-px bg-primary/10"></div>

        {/* Dynamic DEFCON */}
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full animate-pulse shadow-[0_0_6px]" style={{ backgroundColor: defconColor, boxShadow: `0 0 6px ${defconColor}` }}></div>
          <span className="font-mono text-[10px] tracking-[0.2em]" style={{ color: defconColor }}>DEFCON {defcon}</span>
        </div>

        {paused && (
          <>
            <div className="h-3 w-px bg-primary/10"></div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 border border-secondary/50 bg-secondary/10">
              <AlertTriangle size={10} className="text-secondary" />
              <span className="font-mono text-[10px] text-secondary tracking-widest">PAUSED</span>
            </div>
          </>
        )}

        <div className="h-3 w-px bg-primary/10"></div>
        <span className="font-mono text-[10px] text-gray-500 tracking-widest">{PAGE_TITLES[location.pathname] || ''}</span>
      </div>

      <div className="flex items-center gap-5">
        {/* HETERO INDEX */}
        <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500 border-x border-primary/5 px-4 h-6">
           <TrendingUp size={11} className="text-primary/70" /> HETERO={hetero}
        </div>

        {/* AGGREGATION DIVERGENCE */}
        <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500 border-r border-primary/5 pr-4 h-6">
           <Activity size={11} className="text-secondary/70" /> DIV={divergence}
        </div>

        {/* Dynamic Privacy Epsilon */}
        <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500 h-6">
          <Shield size={12} className="text-success" /> PRIVACY ε={epsilon}
        </div>

        {/* Dynamic FL Round */}
        <div className="flex items-center gap-2 text-[10px] font-mono text-gray-400 bg-primary/5 px-2 py-1 rounded border border-primary/10">
          <Wifi size={12} className="text-primary" /> ROUND {flRound}
        </div>

        {mode === 'LIVE' && healthData && (
          <div className="text-[9px] font-mono text-gray-600">
            ▲ {Math.round(healthData.uptime_seconds)}s
          </div>
        )}
        <div className="font-mono text-sm text-primary text-glow tabular-nums">
          {time}
        </div>
      </div>
    </div>
  );
}
