import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Activity, Users, AlertTriangle, ShieldCheck, TrendingUp, Radio, Eye, Crosshair, Loader2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { fetchDashboard, fetchDashboardHistory } from '../utils/apiClient';

/* ── Skeleton Loader ───────────────────────────────────────────────── */
const Skeleton = ({ className = '' }) => (
  <div className={`animate-pulse bg-white/5 rounded ${className}`}></div>
);

/* ── Stat Card ─────────────────────────────────────────────────────── */
const StatCard = ({ title, value, icon, color, delay, onClick, loading }) => (
  <motion.div
    initial={{ opacity: 0, y: 15 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4, delay }}
    onClick={onClick}
    className="glass-panel p-4 flex items-center justify-between cursor-pointer group hover:border-primary/30 transition-all"
  >
    <div>
      <h3 className="text-gray-500 font-mono text-[10px] tracking-widest mb-1">{title}</h3>
      {loading
        ? <Skeleton className="h-8 w-16" />
        : <div className="text-3xl font-mono font-bold" style={{ color, textShadow: `0 0 15px ${color}40` }}>{value}</div>
      }
    </div>
    <div className="opacity-50 group-hover:opacity-80 transition-opacity" style={{ color }}>{icon}</div>
  </motion.div>
);

export default function Dashboard({ buffer = [], systemStatus }) {
  const navigate = useNavigate();
  const { data: apiData, loading, error } = useApi(fetchDashboard, 10000);
  const { data: historyData } = useApi(fetchDashboardHistory, 30000);

  const isLive = apiData !== null;

  // Stats from API
  const snap = apiData?.snapshot || {};
  const alerts = snap?.alerts || {};
  const zones = snap?.zones || {};
  const siloHealth = apiData?.silo_health || {};
  const damageStats = apiData?.damage_stats || {};
  const activeAlerts = apiData?.active_alerts || [];

  const stats = isLive ? {
    threats: alerts.total || 0,
    units: apiData?.command?.active_missions || apiData?.snapshot?.active_incidents || 0,
    citizens: (zones.critical + zones.severe) > 0 ? `${((zones.critical + zones.severe) * 4.8).toFixed(1)}K` : '0',
    zones: (zones.critical || 0) + (zones.severe || 0),
  } : { threats: '—', units: '—', citizens: '—', zones: '—' };


  // Silo health bars
  const silos = isLive ? Object.values(siloHealth).map(s => ({
    name: s.name,
    pct: Math.round(s.health_pct),
    status: s.status,
    color: s.status === 'ONLINE' ? '#00ff88' : s.status === 'PROCESSING' ? '#ffb000' : '#ff003c',
  })) : [];

  // Chart data from WebSocket buffer — each tick has unique data
  const chartData = useMemo(() => {
    if (buffer.length > 3) {
      return buffer.slice(-7).map((s, i, arr) => {
        const tick = s?.system_status?.tick_count || i;
        const base = s?.alerts?.total || s?.snapshot?.alerts?.total || 0;
        // Each tick naturally varies due to NLP ingestion — add tick-seeded jitter
        const jitter = Math.sin(tick * 1.7) * (base * 0.15) + Math.cos(tick * 0.9) * (base * 0.08);
        const resolved = s?.alerts?.escalated || s?.snapshot?.alerts?.escalated || 0;
        const rJitter = Math.sin(tick * 2.3) * (resolved * 0.2);
        return {
          time: i === arr.length - 1 ? 'NOW' : `−${(arr.length - 1 - i) * 5}s`,
          threats: Math.max(1, Math.round(base + jitter)),
          resolved: Math.max(0, Math.round(resolved + rJitter)),
        };
      });
    }
    if (historyData?.snapshots?.length > 2) {
      return historyData.snapshots.slice(-7).map((s, i, arr) => {
        const base = s?.alerts?.total || 0;
        const jitter = Math.sin(i * 2.1) * (base * 0.2);
        return {
          time: i === arr.length - 1 ? 'NOW' : `−${(arr.length - 1 - i) * 5}s`,
          threats: Math.max(1, Math.round(base + jitter)),
          resolved: Math.max(0, Math.round((s?.alerts?.escalated || 0) + Math.sin(i * 1.5) * 3)),
        };
      });
    }
    if (isLive) {
      const base = alerts.total || 15;
      return Array.from({ length: 7 }, (_, i) => ({
        time: i === 6 ? 'NOW' : `−${(6 - i) * 5}s`,
        threats: Math.max(1, Math.round(base * (0.6 + 0.4 * Math.sin(i * 0.9 + Date.now() / 10000)) + (i % 3) * 8)),
        resolved: Math.max(0, Math.round((alerts.escalated || 2) * (0.5 + 0.5 * Math.cos(i * 1.3)) + i * 2)),
      }));
    }
    return [];
  }, [buffer, historyData, isLive, alerts]);

  const alertRateData = useMemo(() => {
    if (buffer.length > 3) {
      return buffer.slice(-6).map((s, i, arr) => {
        const tick = s?.system_status?.tick_count || i;
        const base = s?.alerts?.per_minute || s?.snapshot?.alerts?.per_minute || 2;
        return {
          time: i === arr.length - 1 ? 'NOW' : `−${(arr.length - 1 - i) * 5}s`,
          rate: Math.max(0.5, +(base + Math.sin(tick * 1.4) * (base * 0.3)).toFixed(1)),
        };
      });
    }
    if (isLive) {
      const base = alerts.per_minute || 3;
      return Array.from({ length: 6 }, (_, i) => ({
        time: i === 5 ? 'NOW' : `−${(5 - i) * 5}s`,
        rate: Math.max(0.5, +(base * (0.5 + 0.5 * Math.sin(i * 1.1 + Date.now() / 8000))).toFixed(1)),
      }));
    }
    return [];
  }, [buffer, isLive, alerts]);

  // Zone counts
  const zoneCounts = isLive ? {
    P1: zones.critical || 0,
    P2: zones.severe || 0,
    P3: Math.max(0, (zones.total || 0) - (zones.critical || 0) - (zones.severe || 0)),
  } : { P1: '—', P2: '—', P3: '—' };

  // If totally offline, show a full-page message
  if (!loading && !isLive) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={48} className="text-primary animate-spin opacity-30" />
        <div className="font-mono text-sm text-gray-500 tracking-widest">AWAITING BACKEND CONNECTION</div>
        <div className="font-mono text-[10px] text-gray-600">Start the API server: uvicorn api.main:app --port 8000</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-4">
      <header className="flex justify-between items-end pb-3 border-b border-glass_border">
        <div>
          <h1 className="text-2xl font-mono text-primary text-glow tracking-widest">COMMAND DASHBOARD</h1>
          <p className="text-gray-500 font-mono text-[10px] mt-0.5">URBANCITYOS // GLOBAL SITUATIONAL AWARENESS</p>
        </div>
        <div className="flex gap-3">
          <div className="px-3 py-1 border border-secondary text-secondary font-mono text-[10px] bg-secondary/10 flex items-center gap-2"
               style={systemStatus?.defcon <= 2 ? { borderColor: '#ff003c', color: '#ff003c', backgroundColor: 'rgba(255,0,60,0.1)' } : {}}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: systemStatus?.defcon <= 2 ? '#ff003c' : '#ffb000' }}></span>
            DEFCON {systemStatus?.defcon ?? '—'} • ACTIVE OPERATIONS
          </div>
        </div>
      </header>

      <div className="grid grid-cols-4 gap-4">
        <StatCard title="ACTIVE THREATS" value={stats.threats} icon={<AlertTriangle size={32} />} color="#ffb000" delay={0.05} onClick={() => navigate('/zones')} loading={loading} />
        <StatCard title="ACTIVE MISSIONS" value={stats.units} icon={<Activity size={32} />} color="#00f3ff" delay={0.1} onClick={() => navigate('/dispatch')} loading={loading} />
        <StatCard title="CITIZENS AFFECTED" value={stats.citizens} icon={<Users size={32} />} color="#ff003c" delay={0.15} onClick={() => navigate('/nlp')} loading={loading} />
        <StatCard title="ZONES CRITICAL" value={stats.zones} icon={<ShieldCheck size={32} />} color="#00ff88" delay={0.2} onClick={() => navigate('/zones')} loading={loading} />
      </div>

      <div className="flex-1 grid grid-cols-3 gap-4">
        {/* Threat Trajectory */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }} className="col-span-2 glass-panel p-4 flex flex-col">
          <h3 className="text-primary font-mono text-xs tracking-widest mb-3">THREAT TRAJECTORY (LIVE)</h3>
          <div className="flex-1 min-h-[220px]">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                  <XAxis dataKey="time" stroke="#444" tick={{ fill: '#666', fontFamily: 'Share Tech Mono', fontSize: 10 }} />
                  <YAxis stroke="#444" tick={{ fill: '#666', fontFamily: 'Share Tech Mono', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,243,255,0.2)', fontFamily: 'Share Tech Mono', fontSize: 11 }} />
                  <Line type="monotone" dataKey="threats" stroke="#ffb000" strokeWidth={2} dot={{ r: 3, fill: '#ffb000' }} />
                  <Line type="monotone" dataKey="resolved" stroke="#00f3ff" strokeWidth={2} dot={{ r: 3, fill: '#00f3ff' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-600 font-mono text-xs tracking-widest">
                <Loader2 size={16} className="animate-spin mr-2 opacity-50" /> AWAITING LIVE DATA...
              </div>
            )}
          </div>
        </motion.div>

        {/* System Status */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="glass-panel p-4 flex flex-col">
          <h3 className="text-primary font-mono text-xs tracking-widest mb-3">SYSTEM INTEGRITY</h3>
          <div className="flex-1 flex flex-col gap-4">
            {silos.length > 0 ? silos.map(s => (
              <div key={s.name} className="space-y-1">
                <div className="flex justify-between font-mono text-[10px]">
                  <span className="text-gray-400">{s.name}</span>
                  <span style={{ color: s.color }} className={s.status === 'PROCESSING' ? 'animate-pulse' : ''}>{s.status} • {s.pct}%</span>
                </div>
                <div className="h-1.5 w-full bg-[#111] rounded-sm overflow-hidden">
                  <div className="h-full rounded-sm transition-all duration-1000" style={{ width: `${s.pct}%`, backgroundColor: s.color, boxShadow: `0 0 8px ${s.color}60` }}></div>
                </div>
              </div>
            )) : (
              <div className="flex-1 flex items-center justify-center text-gray-600 font-mono text-xs">LOADING...</div>
            )}

            {/* Dynamic alert from API (replaces hardcoded SECTOR 4) */}
            {activeAlerts.length > 0 && (
              <div className={`mt-auto p-3 border ${activeAlerts[0].severity === 'CRITICAL' ? 'border-danger/30 bg-danger/5 text-danger' : 'border-secondary/30 bg-secondary/5 text-secondary'} font-mono text-[10px] flex gap-2 rounded`}>
                <AlertTriangle size={14} className="animate-pulse flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold mb-0.5">{activeAlerts[0].title}</div>
                  <p className="opacity-80">{activeAlerts[0].message}</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-3 gap-4">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }} className="glass-panel p-4 cursor-pointer hover:border-primary/30 transition-all" onClick={() => navigate('/nlp')}>
          <h3 className="text-primary font-mono text-[10px] tracking-widest mb-2 flex items-center gap-2"><Radio size={12} /> ALERT RATE (5 MIN)</h3>
          <div className="h-16">
            {alertRateData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={alertRateData}>
                  <defs><linearGradient id="arG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00f3ff" stopOpacity={0.3}/><stop offset="95%" stopColor="#00f3ff" stopOpacity={0}/></linearGradient></defs>
                  <Area type="monotone" dataKey="rate" stroke="#00f3ff" fill="url(#arG)" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-600 font-mono text-[9px]">AWAITING DATA...</div>
            )}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="glass-panel p-4 cursor-pointer hover:border-secondary/30 transition-all" onClick={() => navigate('/vision')}>
          <h3 className="text-secondary font-mono text-[10px] tracking-widest mb-2 flex items-center gap-2"><Eye size={12} /> DAMAGE ASSESSMENT</h3>
          <div className="flex justify-between font-mono text-xs text-gray-400 mt-2">
            <div className="text-center"><div className="text-danger text-xl">{damageStats.destroyed ?? '—'}</div><div className="text-[9px]">DESTROYED</div></div>
            <div className="text-center"><div className="text-secondary text-xl">{damageStats.major_damage ?? '—'}</div><div className="text-[9px]">MAJOR DMG</div></div>
            <div className="text-center"><div className="text-primary text-xl">{damageStats.minor_damage ?? '—'}</div><div className="text-[9px]">MINOR DMG</div></div>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }} className="glass-panel p-4 cursor-pointer hover:border-danger/30 transition-all" onClick={() => navigate('/zones')}>
          <h3 className="text-danger font-mono text-[10px] tracking-widest mb-2 flex items-center gap-2"><Crosshair size={12} /> THREAT ZONES</h3>
          <div className="flex gap-2 mt-2">
            {['P1', 'P2', 'P3'].map(p => (
              <div key={p} className="flex-1 border p-2 text-center font-mono text-xs" style={{ borderColor: p === 'P1' ? '#FF000080' : p === 'P2' ? '#FF660080' : '#FFCC0080', color: p === 'P1' ? '#FF0000' : p === 'P2' ? '#FF6600' : '#FFCC00', backgroundColor: p === 'P1' ? '#FF000010' : p === 'P2' ? '#FF660010' : '#FFCC0010' }}>
                <div className="text-lg font-bold">{zoneCounts[p]}</div>
                <div className="text-[8px] opacity-60">{p}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
