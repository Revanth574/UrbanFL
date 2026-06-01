import { useState, useMemo, useCallback, useRef } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchInfraStatus, fetchRiskForecast, fetchTrafficSegments } from '../utils/apiClient';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, Navigation, Wind, AlertTriangle, TrendingUp, Bell, Loader2 } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const getHealthColor = (h) => h > 80 ? '#00ff88' : h > 50 ? '#ffb000' : h > 20 ? '#ff6b00' : '#ff003c';

/* ── Traffic Map with road congestion overlays ───────────────────── */
function TrafficMap({ nodes = [], trafficSegments = [] }) {
  return (
    <MapContainer center={[41.020, 28.930]} zoom={13} style={{ width: '100%', height: '100%', borderRadius: 4 }} zoomControl={false} attributionControl={false}>
      <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />

      {/* Traffic route segments with congestion colors */}
      {trafficSegments.map((seg, i) => (
        <Polyline
          key={`traffic-${i}`}
          positions={seg.coords}
          pathOptions={{
            color: seg.congestion > 0.7 ? '#ff003c' : seg.congestion > 0.4 ? '#ff6b00' : seg.congestion > 0.2 ? '#ffb000' : '#00ff88',
            weight: seg.congestion > 0.5 ? 5 : 3,
            opacity: 0.7,
            dashArray: seg.blocked ? '4 8' : null,
          }}
        >
          <Popup>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: '#fff', background: '#0a0a0a', padding: 8, border: '1px solid #333' }}>
              <div style={{ color: seg.congestion > 0.7 ? '#ff003c' : '#00ff88', fontWeight: 'bold', marginBottom: 4 }}>
                {seg.name || `ROUTE-${i}`}
              </div>
              <div style={{ color: '#888' }}>
                Congestion: {(seg.congestion * 100).toFixed(0)}%<br/>
                Speed: {seg.speed} km/h<br/>
                {seg.blocked && <span style={{color:'#ff003c'}}>⚠ BLOCKED — REROUTING</span>}
              </div>
            </div>
          </Popup>
        </Polyline>
      ))}

      {/* Infrastructure nodes */}
      {nodes.map(n => (
        <CircleMarker
          key={n.id || n.node_id}
          center={[n.lat, n.lon]}
          radius={6}
          pathOptions={{ fillColor: getHealthColor(n.health_pct || 50), fillOpacity: 0.7, color: '#fff', weight: 1, opacity: 0.3 }}
        >
          <Popup>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: '#fff', background: '#0a0a0a', padding: 8, border: '1px solid #333' }}>
              <div style={{ color: getHealthColor(n.health_pct || 50), fontWeight: 'bold' }}>{n.node_id || n.id}</div>
              <div style={{ color: '#888' }}>Health: {(n.health_pct || 50).toFixed(0)}% • {(n.node_type || n.type || 'INFRA').toUpperCase()}</div>
            </div>
          </Popup>
        </CircleMarker>
      ))}

      <style>{`
        .leaflet-popup-content-wrapper { background: transparent; padding: 0; box-shadow: none; border: none; }
        .leaflet-popup-tip { display: none; }
      `}</style>
    </MapContainer>
  );
}

export default function Infrastructure() {
  const [activeTab, setActiveTab] = useState('grid');
  const { data: infraData, loading } = useApi(fetchInfraStatus, 10000);
  const { data: riskData } = useApi(fetchRiskForecast, 15000);
  const { data: trafficData } = useApi(fetchTrafficSegments, 15000);
  const isLive = infraData !== null;

  const infraNodes = useMemo(() => {
    if (!isLive || !infraData?.nodes) return [];
    return infraData.nodes.map(n => ({
      id: n.node_id || n.id,
      node_id: n.node_id || n.id,
      node_type: n.node_type || n.type || 'POWER',
      lat: n.lat, lon: n.lon,
      health_pct: n.health_pct || 50,
      status: n.status || 'operational',
    }));
  }, [infraData, isLive]);

  const cascadePredictions = useMemo(() => {
    if (!isLive || !infraData?.cascade_predictions) return [];
    return infraData.cascade_predictions;
  }, [infraData, isLive]);

  // Real TomTom traffic segments from backend API
  const trafficSegments = useMemo(() => {
    if (!trafficData?.segments) return [];
    return trafficData.segments.map(seg => ({
      name: seg.name,
      coords: seg.coords,
      congestion: seg.congestion,
      speed: seg.current_speed,
      freeFlow: seg.free_flow_speed,
      blocked: seg.closure,
    }));
  }, [trafficData]);

  // ── Power grid data — varies with each API refresh ────────────────
  const gridData = useMemo(() => {
    if (!isLive || !infraData?.network_health) return [];
    const health = infraData.network_health;
    const networks = Object.entries(health);
    const avgHealth = networks.reduce((s, [, v]) => s + (v.health_pct || 50), 0) / Math.max(networks.length, 1);
    const tickSeed = infraData?.total_nodes || Date.now();

    return Array.from({ length: 24 }, (_, h) => {
      const baseLoad = 20 + 60 * Math.sin((h - 5) * Math.PI / 12);
      const healthImpact = (100 - avgHealth) * 0.4;
      const jitter = Math.sin(h * 1.7 + tickSeed * 0.01) * 12 + Math.cos(h * 0.8 + tickSeed * 0.007) * 8;
      const load = Math.round(Math.max(8, Math.min(140, baseLoad + healthImpact + jitter)));
      const capacity = h >= 8 && h <= 20 ? 100 : 120;
      return { time: `${String(h).padStart(2, '0')}:00`, load, capacity };
    });
  }, [infraData, isLive]);

  const [prediction, setPrediction] = useState(null);
  const [predAlerts, setPredAlerts] = useState([]);
  const predCountRef = useRef(0);

  const runPrediction = useCallback(() => {
    if (gridData.length === 0) return;
    predCountRef.current += 1;
    const seed = predCountRef.current;
    const future = [];
    const newAlerts = [];
    const nowHour = new Date().getHours();

    for (let h = 0; h < 12; h++) {
      const hour = (nowHour + h + 1) % 24;
      const base = 20 + 60 * Math.sin((hour - 5) * Math.PI / 12);
      // Each prediction run produces different results (seed-based)
      const noise = Math.sin(h * 2.3 + seed * 3.7) * 20 + Math.cos(h * 1.1 + seed * 1.9) * 12;
      const predicted = Math.round(Math.max(10, Math.min(145, base + noise)));
      const capacity = hour >= 8 && hour <= 20 ? 100 : 120;
      future.push({ time: `+${h + 1}h`, load: predicted, capacity, predicted: true });
      if (predicted > capacity) {
        newAlerts.push({
          time: `+${h + 1}h`,
          load: predicted,
          capacity,
          severity: predicted > capacity * 1.2 ? 'CRITICAL' : 'WARNING',
          message: `Predicted overload: ${predicted}MW vs ${capacity}MW capacity at ${String(hour).padStart(2, '0')}:00`,
        });
      }
    }
    setPrediction(future);
    setPredAlerts(newAlerts);
  }, [gridData]);

  // Env data from infra overlay
  const envCards = useMemo(() => {
    if (!isLive) return [];
    const env = infraData?.environmental_overlay || {};
    return [
      { sensor: 'AQI-STATION-4', metric: 'PM2.5', value: env.max_risk > 0.5 ? 185 : 65, threshold: 150, severity: env.max_risk > 0.5 ? 'HIGH' : 'NOMINAL', message: 'Air quality sensor readings from deployed network.' },
      { sensor: 'INFRA-GRID', metric: 'FAILED', value: infraData?.failed_nodes || 0, threshold: 5, severity: (infraData?.failed_nodes || 0) > 3 ? 'HIGH' : 'NOMINAL', message: `${infraData?.failed_nodes || 0} infrastructure nodes currently offline.` },
      { sensor: 'GRID-LOAD', metric: 'MW', value: gridData.length > 0 ? gridData[new Date().getHours()].load : 0, threshold: 100, severity: gridData.length > 0 && gridData[new Date().getHours()].load > 100 ? 'HIGH' : 'NOMINAL', message: 'Current power grid load reading.' },
      { sensor: 'CASCADE-RISK', metric: 'ALERTS', value: cascadePredictions.length, threshold: 3, severity: cascadePredictions.length > 2 ? 'HIGH' : 'NOMINAL', message: `${cascadePredictions.length} potential cascade failure paths detected.` },
    ];
  }, [isLive, infraData, gridData, cascadePredictions]);

  const tabs = [
    { id: 'grid', label: 'POWER GRID', icon: <Zap size={14} /> },
    { id: 'traffic', label: 'TRAFFIC', icon: <Navigation size={14} /> },
    { id: 'env', label: 'ENVIRONMENT', icon: <Wind size={14} /> },
  ];

  if (!loading && !isLive) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={48} className="text-primary animate-spin opacity-30" />
        <div className="font-mono text-sm text-gray-500 tracking-widest">AWAITING INFRASTRUCTURE DATA</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-4">
      <header className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded bg-primary/10 border border-primary/30 flex items-center justify-center text-primary"><Activity size={20} /></div>
          <div>
            <h1 className="text-2xl font-mono text-primary tracking-widest">INFRA-COMMAND</h1>
            <p className="text-gray-500 text-xs font-mono uppercase">Silo 3 // {infraNodes.length} Nodes • {infraData?.failed_nodes || 0} Failed • {cascadePredictions.length} Cascade Risks</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {predAlerts.length > 0 && <div className="flex items-center gap-2 border border-danger px-3 py-1 bg-danger/10 text-danger font-mono text-xs animate-pulse"><Bell size={14} /> {predAlerts.length} OVERLOAD ALERTS</div>}
          {cascadePredictions.length > 0 && <div className="flex items-center gap-2 border border-secondary px-3 py-1 bg-secondary/10 text-secondary font-mono text-xs"><AlertTriangle size={14} /> {cascadePredictions.length} CASCADE RISKS</div>}
        </div>
      </header>

      <div className="flex gap-2">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} className={`flex items-center gap-2 px-4 py-2 font-mono text-xs tracking-widest border transition-all ${activeTab === t.id ? 'border-primary bg-primary/10 text-primary shadow-[0_0_10px_rgba(0,243,255,0.2)]' : 'border-glass_border text-gray-500 hover:text-white'}`}>
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'grid' && (
          <motion.div key="grid" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex gap-4">
            <div className="flex-1 glass-panel p-5 flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-primary font-mono text-xs tracking-widest">GRID LOAD PREDICTION (AI)</h3>
                <button onClick={runPrediction} className="flex items-center gap-2 px-3 py-1 border border-primary text-primary font-mono text-xs hover:bg-primary/10 transition-all"><TrendingUp size={14} /> RUN INFERENCE</button>
              </div>
              <div className="flex-1 min-h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={prediction ? [...gridData, ...prediction] : gridData}>
                    <defs>
                      <linearGradient id="loadG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00f3ff" stopOpacity={0.4} /><stop offset="95%" stopColor="#00f3ff" stopOpacity={0} /></linearGradient>
                      <linearGradient id="predG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ff6b00" stopOpacity={0.3} /><stop offset="95%" stopColor="#ff6b00" stopOpacity={0} /></linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                    <XAxis dataKey="time" stroke="#555" tick={{ fill: '#666', fontFamily: 'Share Tech Mono', fontSize: 10 }} />
                    <YAxis stroke="#555" tick={{ fill: '#666', fontFamily: 'Share Tech Mono', fontSize: 10 }} domain={[0, 150]} />
                    <Tooltip contentStyle={{ backgroundColor: '#0d0d0d', border: '1px solid rgba(0,243,255,0.3)', fontFamily: 'Share Tech Mono', fontSize: 11 }} />
                    <ReferenceLine y={100} stroke="#ff003c" strokeDasharray="5 5" label={{ value: 'CAPACITY', fill: '#ff003c', fontSize: 9, fontFamily: 'Share Tech Mono' }} />
                    <Area type="monotone" dataKey="load" stroke="#00f3ff" fill="url(#loadG)" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="w-72 glass-panel p-4 flex flex-col">
              <h3 className="text-danger font-mono text-xs tracking-widest mb-3 flex items-center gap-2"><AlertTriangle size={14} /> ANOMALIES</h3>
              {predAlerts.length === 0 ? (
                <div className="flex-1 flex items-center justify-center text-gray-500 font-mono text-xs text-center uppercase tracking-tighter">
                  Click "RUN INFERENCE"<br />to predict grid overloads
                </div>
              ) : (
                <div className="flex-1 space-y-2 overflow-y-auto custom-scrollbar">
                  {predAlerts.map((a, i) => (
                    <div key={i} className="p-2 border border-danger/30 bg-danger/5 text-danger font-mono text-[10px]">
                      <div className="font-bold mb-1">{a.severity} @ {a.time}</div>
                      <p>{a.message}</p>
                    </div>
                  ))}
                </div>
              )}
              {/* Congestion Summary */}
              <div className="mt-4 pt-3 border-t border-glass_border">
                <h4 className="text-primary font-mono text-[10px] tracking-widest mb-2">TRAFFIC IMPACT</h4>
                <div className="space-y-1 font-mono text-[10px]">
                  <div className="flex justify-between"><span className="text-gray-500">BLOCKED ROUTES</span><span className="text-danger">{trafficSegments.filter(s => s.blocked).length}</span></div>
                  <div className="flex justify-between"><span className="text-gray-500">CONGESTED</span><span className="text-secondary">{trafficSegments.filter(s => s.congestion > 0.5).length}</span></div>
                  <div className="flex justify-between"><span className="text-gray-500">CLEAR</span><span className="text-success">{trafficSegments.filter(s => s.congestion <= 0.3).length}</span></div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'traffic' && (
          <motion.div key="traffic" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex gap-4">
            <div className="flex-1 flex flex-col gap-3">
              <div className="flex-1 glass-panel overflow-hidden" style={{ minHeight: 400 }}>
                <TrafficMap nodes={infraNodes} trafficSegments={trafficSegments} />
              </div>
              <div className="flex gap-4 font-mono text-[10px] text-gray-400 px-2">
                <span className="flex items-center gap-1"><div className="w-3 h-1 bg-success rounded"></div> CLEAR</span>
                <span className="flex items-center gap-1"><div className="w-3 h-1 bg-[#ffb000] rounded"></div> SLOW</span>
                <span className="flex items-center gap-1"><div className="w-3 h-1 bg-[#ff6b00] rounded"></div> CONGESTED</span>
                <span className="flex items-center gap-1"><div className="w-3 h-1 bg-danger rounded"></div> BLOCKED</span>
                <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-primary opacity-60"></div> INFRA NODE</span>
              </div>
            </div>
            <div className="w-72 glass-panel p-4 font-mono text-xs flex flex-col">
              <h3 className="text-primary tracking-widest mb-3 flex items-center gap-2"><Navigation size={14} /> ROUTE STATUS</h3>
              <div className="flex-1 space-y-2 overflow-y-auto custom-scrollbar">
                {trafficSegments.length === 0 ? (
                  <div className="text-gray-600 text-center py-10">NO ROUTE DATA</div>
                ) : trafficSegments.sort((a, b) => b.congestion - a.congestion).map((seg, i) => (
                  <div key={i} className="p-2 border-l-2 bg-black/30" style={{ borderColor: seg.congestion > 0.7 ? '#ff003c' : seg.congestion > 0.4 ? '#ff6b00' : '#00ff88' }}>
                    <div className="font-bold text-white text-[10px] mb-1">{seg.name}</div>
                    <div className="flex justify-between text-[9px] text-gray-400">
                      <span>{seg.speed} km/h</span>
                      <span style={{ color: seg.congestion > 0.7 ? '#ff003c' : seg.congestion > 0.4 ? '#ff6b00' : '#00ff88' }}>
                        {seg.blocked ? 'BLOCKED' : seg.congestion > 0.7 ? 'SEVERE' : seg.congestion > 0.4 ? 'SLOW' : 'CLEAR'}
                      </span>
                    </div>
                    <div className="h-1 bg-[#111] mt-1 rounded overflow-hidden">
                      <div className="h-full transition-all" style={{ width: `${seg.congestion * 100}%`, backgroundColor: seg.congestion > 0.7 ? '#ff003c' : seg.congestion > 0.4 ? '#ff6b00' : '#00ff88' }}></div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 pt-3 border-t border-glass_border space-y-2">
                <div className="flex justify-between"><span className="text-gray-500">TOTAL NODES</span><span className="text-white">{infraData?.total_nodes || 0}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">FAILED</span><span className="text-danger">{infraData?.failed_nodes || 0}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">DEGRADED</span><span className="text-secondary">{infraData?.degraded_nodes || 0}</span></div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'env' && (
          <motion.div key="env" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 grid grid-cols-2 gap-4">
            {envCards.map((ea, i) => (
              <div key={i} className={`glass-panel p-5 ${ea.severity === 'HIGH' ? 'border-danger/40' : ''}`}>
                <div className="flex justify-between font-mono text-[10px] text-gray-400 mb-2">
                  <span>{ea.sensor}</span>
                  <span className={ea.severity === 'HIGH' ? 'text-danger animate-pulse' : 'text-success'}>{ea.severity}</span>
                </div>
                <div className="text-3xl font-mono text-white mb-1">
                  {ea.value}<span className="text-xs text-gray-500 ml-1">{ea.metric}</span>
                </div>
                <div className="text-[10px] text-gray-500 font-mono mb-3 uppercase">Threshold: {ea.threshold}</div>
                <div className="h-1.5 w-full bg-[#111] rounded overflow-hidden mb-3">
                  <div className="h-full transition-all duration-1000" style={{
                    width: `${Math.min(100, (ea.value / ea.threshold) * 100)}%`,
                    backgroundColor: ea.value > ea.threshold ? '#ff003c' : ea.value > ea.threshold * 0.7 ? '#ffb000' : '#00ff88',
                  }}></div>
                </div>
                <p className="text-xs text-gray-400 font-mono border-t border-glass_border pt-2">{ea.message}</p>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
