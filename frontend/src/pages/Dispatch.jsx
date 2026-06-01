import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApi } from '../hooks/useApi';
import { fetchDispatchUnits, fetchDispatchIncidents, fetchDispatchHospitals, fetchDispatchActivityLog } from '../utils/apiClient';
import { Cross, Flame, Shield, X, Siren, Navigation, Loader2, Clock, MapPin, Activity } from 'lucide-react';
import { MapContainer, TileLayer, Polyline, Marker, Popup, CircleMarker } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const INCIDENT_COLORS = { P1:'#ff003c', P2:'#ff6b00', P3:'#ffb000' };
const DEPT_CONFIG = {
  ambulance: { label: 'MEDICAL', color: '#00ff88', emoji: '🚑' },
  fire_engine: { label: 'FIRE & RESCUE', color: '#ff6b00', emoji: '🚒' },
  police: { label: 'SECURITY', color: '#00f3ff', emoji: '🚓' },
};

// Distinct shaped icons for each unit type
const makeUnitIcon = (type) => {
  const cfg = DEPT_CONFIG[type] || { color: '#ffb000', emoji: '🚐' };
  return L.divIcon({
    className: '',
    html: `<div style="
      width:28px;height:28px;
      background:${cfg.color}20;
      border:2px solid ${cfg.color};
      border-radius:${type === 'ambulance' ? '50%' : type === 'fire_engine' ? '4px' : '50% 50% 0 50%'};
      display:flex;align-items:center;justify-content:center;
      font-size:14px;
      box-shadow:0 0 12px ${cfg.color}80;
      backdrop-filter:blur(4px);
    ">${cfg.emoji}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
};

const hospitalIcon = (full) => L.divIcon({
  className: '',
  html: `<div style="width:16px;height:16px;background:${full?'#ff003c40':'#00ff8830'};border:1px solid ${full?'#ff003c':'#00ff88'};border-radius:2px;display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:bold;color:${full?'#ff003c':'#00ff88'};">H</div>`,
  iconSize: [16, 16], iconAnchor: [8, 8],
});

const incidentIcon = (priority) => L.divIcon({
  className: '',
  html: `<div style="width:14px;height:14px;background:${INCIDENT_COLORS[priority] || '#ffb000'};border:2px solid #fff;transform:rotate(45deg);box-shadow:0 0 15px ${INCIDENT_COLORS[priority]}80;"></div>`,
  iconSize: [14, 14], iconAnchor: [7, 7],
});

function DispatchMap({ units = [], incidents = [], hospitals = [], activeDept, selectedUnit, setSelectedUnit, disasterZones = [] }) {
  const filteredUnits = activeDept === 'all' ? units : units.filter(u => u.type === activeDept);

  return (
    <MapContainer center={[41.020,28.930]} zoom={13} style={{width:'100%',height:'100%',borderRadius:4}} zoomControl={false} attributionControl={false}>
      <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" opacity={0.8} />

      {/* Disaster / hazard zones (avoid areas) */}
      {disasterZones.map((dz, i) => (
        <CircleMarker
          key={`dz-${i}`}
          center={[dz.lat, dz.lon]}
          radius={dz.radius || 20}
          pathOptions={{ fillColor: '#ff003c', fillOpacity: 0.08, color: '#ff003c', weight: 1, opacity: 0.3, dashArray: '4 4' }}
        >
          <Popup>
            <div className="popup-content" style={{ borderTop: '3px solid #ff003c' }}>
              <div className="text-danger font-bold text-xs mb-1">⚠ HAZARD ZONE</div>
              <div className="text-[10px] text-gray-400">{dz.type || 'DISASTER IMPACT AREA'}</div>
              <div className="text-[10px] text-gray-400">Severity: {(dz.severity * 100).toFixed(0)}%</div>
            </div>
          </Popup>
        </CircleMarker>
      ))}

      {/* Incidents */}
      {incidents.map(inc => (
        <Marker key={inc.id} position={[inc.lat, inc.lon]} icon={incidentIcon(inc.priority)}>
          <Popup><div className="popup-content" style={{borderTop:`3px solid ${INCIDENT_COLORS[inc.priority]}`}}>
            <div className="font-bold text-danger mb-1">{inc.id} • {inc.priority}</div>
            <div className="uppercase text-[10px] text-gray-400 mb-2">{(inc.type || '').replace(/_/g,' ')}</div>
            <div className="grid grid-cols-2 gap-2 text-[9px] font-mono">
              <div>SEVERITY: <span className="text-white">{(inc.severity * 100).toFixed(0)}%</span></div>
              <div>UNITS: <span className="text-white">{inc.units_assigned}</span></div>
            </div>
          </div></Popup>
        </Marker>
      ))}

      {/* Hospitals — only on Medical tab */}
      {activeDept === 'ambulance' && hospitals.map(h => (
        <Marker key={h.id} position={[h.lat, h.lon]} icon={hospitalIcon(h.status === 'FULL')}>
          <Popup><div className="popup-content" style={{borderTop:'3px solid #00ff88'}}>
            <div className="font-bold text-success mb-1">{h.name || h.id}</div>
            <div className="text-[10px] text-gray-400">BEDS: {h.beds - h.utilized}/{h.beds}</div>
            <div className={`text-[10px] font-bold ${h.status === 'FULL' ? 'text-danger' : 'text-success'}`}>{h.status}</div>
          </div></Popup>
        </Marker>
      ))}

      {/* Unit Routes — colored by type, dashed for optimal avoidance */}
      {filteredUnits.map(u => {
        if (!u.route || u.route.length < 2) return null;
        const cfg = DEPT_CONFIG[u.type] || { color: '#ffb000' };
        return (
          <Polyline
            key={`r-${u.id}`}
            positions={u.route.map(r => [r.lat || r[0], r.lon || r[1]])}
            pathOptions={{ color: cfg.color, weight: 3, opacity: 0.6, dashArray: '8 5' }}
          />
        );
      })}

      {/* Units with distinct icons */}
      {filteredUnits.map(u => (
        <Marker key={u.id} position={[u.lat, u.lon]} icon={makeUnitIcon(u.type)} eventHandlers={{click:()=>setSelectedUnit(u)}}>
          <Popup><div className="popup-content" style={{borderTop:`3px solid ${(DEPT_CONFIG[u.type]||{color:'#ffb000'}).color}`}}>
            <div className="font-bold" style={{color:(DEPT_CONFIG[u.type]||{color:'#ffb000'}).color}}>{u.callsign}</div>
            <div className="uppercase text-[10px]">{u.status.replace(/_/g,' ')}</div>
            {u.assigned_incident && <div className="text-[9px] text-danger mt-1">→ {u.assigned_incident}</div>}
            {u.route?.length > 0 && <div className="text-[9px] text-gray-500 mt-1">ROUTE: {u.route.length} waypoints</div>}
          </div></Popup>
        </Marker>
      ))}

      <style>{`
        .leaflet-popup-content-wrapper { background: #0a0a0a; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
        .popup-content { font-family: 'Share Tech Mono', monospace; padding: 6px; min-width: 140px; color: #fff; }
      `}</style>
    </MapContainer>
  );
}

export default function Dispatch() {
  const [activeDept, setActiveDept] = useState('all');
  const [selectedUnit, setSelectedUnit] = useState(null);

  const { data: unitData, loading: unitsLoading } = useApi(fetchDispatchUnits, 5000);
  const { data: incData } = useApi(fetchDispatchIncidents, 8000);
  const { data: hospData } = useApi(fetchDispatchHospitals, 15000);
  const { data: logData } = useApi(fetchDispatchActivityLog, 10000);

  const units = unitData?.units || [];
  const incidents = incData?.incidents || [];
  const hospitals = hospData?.hospitals || [];
  const activityLog = logData?.log || [];

  // Generate disaster zones from incidents for avoidance visualization
  const disasterZones = useMemo(() => {
    return incidents
      .filter(inc => inc.priority === 'P1' || inc.severity > 0.6)
      .map(inc => ({
        lat: inc.lat,
        lon: inc.lon,
        radius: inc.severity > 0.8 ? 30 : 18,
        type: (inc.type || 'crisis').replace(/_/g, ' ').toUpperCase(),
        severity: inc.severity || 0.5,
      }));
  }, [incidents]);

  const unitCounts = useMemo(() => {
    const counts = { total: units.length, dispatched: 0, available: 0 };
    for (const u of units) {
      counts[u.type] = (counts[u.type] || 0) + 1;
      if (u.status === 'dispatched' || u.status === 'en_route') counts.dispatched++;
      else counts.available++;
    }
    return counts;
  }, [units]);

  const departments = [
    { id:'all', label:'ALL UNITS', icon:<Siren size={16}/>, color:'#fff', count: units.length },
    { id:'ambulance', label:'MEDICAL', icon:<Cross size={16}/>, color:'#00ff88', count: unitCounts.ambulance || 0 },
    { id:'fire_engine', label:'FIRE & RESCUE', icon:<Flame size={16}/>, color:'#ff6b00', count: unitCounts.fire_engine || 0 },
    { id:'police', label:'SECURITY', icon:<Shield size={16}/>, color:'#00f3ff', count: unitCounts.police || 0 },
  ];

  if (!unitsLoading && !unitData) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={48} className="text-primary animate-spin opacity-30" />
        <div className="font-mono text-sm text-gray-500 tracking-widest uppercase">Initializing Tactical Feed</div>
      </div>
    );
  }

  const activeDeptConfig = departments.find(d => d.id === activeDept);
  const deptColor = activeDeptConfig?.color || '#00f3ff';

  // Filter units for right panel
  const filteredUnits = activeDept === 'all' ? units : units.filter(u => u.type === activeDept);
  const routedUnits = filteredUnits.filter(u => u.route?.length > 1);

  return (
    <div className="h-full flex flex-col gap-4">
      <header className="flex justify-between items-end pb-2 border-b border-glass_border">
        <div>
          <h1 className="text-2xl font-mono text-primary tracking-widest">DISPATCH CENTER</h1>
          <p className="text-gray-500 text-[10px] font-mono uppercase tracking-[0.2em] mt-0.5">Tactical Deployment // Fleet Intelligence</p>
        </div>
        <div className="flex gap-4 font-mono text-[10px] mb-1">
          <div className="flex items-center gap-2 text-success"><Activity size={10} className="animate-pulse"/> {units.length} ACTIVE UNITS</div>
          <div className="flex items-center gap-2 text-primary"><Navigation size={10}/> {routedUnits.length} EN ROUTE</div>
          <div className="flex items-center gap-2 text-danger"><Siren size={10}/> {incidents.length} OPEN INCIDENTS</div>
        </div>
      </header>

      <div className="flex gap-2">
        {departments.map(d => (
          <button key={d.id} onClick={()=>{setActiveDept(d.id);setSelectedUnit(null);}}
            className={`flex items-center gap-2 px-5 py-2.5 font-mono text-[10px] tracking-widest border transition-all ${activeDept===d.id?'border-primary bg-primary/10 text-primary shadow-[0_0_15px_rgba(0,243,255,0.2)]':'border-glass_border text-gray-500 hover:text-white hover:border-white/20'}`}>
            {d.icon} {d.label} <span className="text-gray-600 ml-1">({d.count})</span>
          </button>
        ))}
      </div>

      <div className="flex-1 flex gap-4 overflow-hidden">
        <div className="flex-1 glass-panel overflow-hidden relative">
          <DispatchMap
            units={units}
            incidents={incidents}
            hospitals={hospitals}
            activeDept={activeDept}
            selectedUnit={selectedUnit}
            setSelectedUnit={setSelectedUnit}
            disasterZones={disasterZones}
          />
          <div className="absolute top-4 left-4 z-[400] glass-panel p-2 flex flex-col gap-2">
            <button className="p-1 hover:bg-white/10 rounded"><MapPin size={16} className="text-primary"/></button>
            <button className="p-1 hover:bg-white/10 rounded"><Navigation size={16} className="text-gray-500"/></button>
          </div>
          {/* Legend */}
          <div className="absolute bottom-4 left-4 z-[400] glass-panel p-2 font-mono text-[9px] flex gap-3 text-gray-400 bg-black/80 backdrop-blur-md">
            <span className="flex items-center gap-1">🚑 AMBULANCE</span>
            <span className="flex items-center gap-1">🚒 FIRE</span>
            <span className="flex items-center gap-1">🚓 POLICE</span>
            <span className="flex items-center gap-1">◆ INCIDENT</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-danger/30 border border-danger inline-block"></span> HAZARD</span>
          </div>
        </div>

        <div className="w-80 flex flex-col gap-4 overflow-hidden">
          <AnimatePresence mode="wait">
            {selectedUnit ? (
              <motion.div key="unit" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} exit={{opacity:0}} className="glass-panel p-4 font-mono text-xs border-r-2" style={{borderColor:deptColor}}>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="tracking-widest font-bold" style={{color:deptColor}}>{selectedUnit.callsign}</h3>
                  <button onClick={()=>setSelectedUnit(null)} className="text-gray-500 hover:text-white p-1 hover:bg-white/5 rounded"><X size={14}/></button>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-end border-b border-white/5 pb-2">
                    <span className="text-gray-500">STATUS</span>
                    <span style={{color: selectedUnit.status.includes('dispatch') ? '#ff6b00' : deptColor}} className="font-bold">{selectedUnit.status.replace(/_/g,' ').toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between items-end border-b border-white/5 pb-2">
                    <span className="text-gray-500">TYPE</span>
                    <span className="text-white">{(DEPT_CONFIG[selectedUnit.type]||{label:'UNIT'}).label}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div><div className="text-[9px] text-gray-500">SPEED</div><div className="text-white font-bold">{selectedUnit.speed_kmh} KM/H</div></div>
                    <div><div className="text-[9px] text-gray-500">ETA</div><div className="text-primary font-bold">{selectedUnit.eta_minutes || '—'} MIN</div></div>
                  </div>
                  {selectedUnit.route?.length > 0 && (
                    <div className="p-2 bg-primary/5 border border-primary/20 rounded">
                      <div className="text-[9px] text-primary font-bold mb-1">OPTIMAL ROUTE</div>
                      <div className="text-gray-400 text-[9px]">{selectedUnit.route.length} waypoints • avoids disaster zones</div>
                    </div>
                  )}
                  {selectedUnit.assigned_incident && (
                    <div className="p-2 bg-danger/5 border border-danger/20 rounded">
                      <div className="text-[9px] text-danger font-bold mb-1">ACTIVE MISSION</div>
                      <div className="text-white text-xs mb-1">{selectedUnit.assigned_incident}</div>
                    </div>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div key="empty" initial={{opacity:0}} animate={{opacity:1}} className="glass-panel p-6 font-mono text-xs text-center flex flex-col items-center justify-center border-dashed border-2">
                <Siren size={32} className="mb-3 text-gray-700"/>
                <span className="text-gray-500 tracking-widest uppercase text-[10px]">Select Unit to Monitor Telemetry</span>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex-1 glass-panel flex flex-col overflow-hidden">
            <div className="p-4 border-b border-glass_border flex justify-between items-center">
              <h3 className="text-primary font-mono text-xs tracking-widest flex items-center gap-2"><Clock size={14}/> LIVE ACTIVITY</h3>
              <span className="text-[9px] bg-primary/10 text-primary px-2 py-0.5 rounded">AUTO-UPDATES</span>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
              {activityLog.length === 0 ? (
                <div className="h-full flex items-center justify-center text-gray-700 font-mono text-[10px]">NO RECENT TELEMETRY</div>
              ) : activityLog.map((entry, i) => {
                const entryType = entry.type || '';
                const entryColor = entryType.includes('fire') ? '#ff6b00' : entryType.includes('medical') || entryType.includes('ambulance') ? '#00ff88' : '#00f3ff';
                return (
                  <div key={i} className="p-3 border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors cursor-crosshair">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-primary font-bold text-[10px]">{entry.time}</span>
                      <span className="text-[9px] px-1 bg-white/10 text-gray-400">{entry.action}</span>
                    </div>
                    <div className="text-white text-[10px] mb-1" style={{color: entryColor}}>{entry.vehicle} → {entry.incident}</div>
                    <div className="text-gray-500 text-[9px] uppercase">{entryType.replace(/_/g,' ')} • SEVERITY {entry.severity}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
