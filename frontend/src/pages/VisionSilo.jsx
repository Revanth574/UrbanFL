import { useMemo, useEffect, useRef } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchVisionAssessment } from '../utils/apiClient';
import { Eye, Activity, Loader2 } from 'lucide-react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const DAMAGE_COLORS = {
  destroyed: '#ff003c',
  major_damage: '#ff6b00',
  minor_damage: '#ffb000',
  intact: '#00ff88',
};

const DAMAGE_RADIUS = { destroyed: 10, major_damage: 7, minor_damage: 5, intact: 3 };

/* ── Auto-fit map to damage data ──────────────────────────────────── */
function MapFitter({ cells }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    if (!cells || cells.length === 0 || fitted.current) return;
    const lats = cells.map(c => c.lat);
    const lons = cells.map(c => c.lon);
    if (lats.length === 0) return;
    const bounds = L.latLngBounds(
      [Math.min(...lats) - 0.02, Math.min(...lons) - 0.02],
      [Math.max(...lats) + 0.02, Math.max(...lons) + 0.02]
    );
    map.fitBounds(bounds, { padding: [60, 60], maxZoom: 14, animate: true, duration: 2 });
    fitted.current = true;
  }, [cells, map]);
  return null;
}

export default function VisionSilo() {


  const { data: apiData, loading } = useApi(fetchVisionAssessment, 15000);
  const isLive = apiData !== null;

  const activeStage = apiData?.stage || 'INIT';
  const stats = useMemo(() => ({
    destroyed: isLive ? (apiData?.destroyed || 0) : 0,
    major: isLive ? (apiData?.major_damage || 0) : 0,
    minor: isLive ? (apiData?.minor_damage || 0) : 0,
    totalCells: isLive ? (apiData?.total_cells || 0) : 0,
  }), [apiData, isLive]);

  // All damage cells — always visible
  const allCells = useMemo(() => {
    if (!isLive || !apiData?.cells) return [];
    return apiData.cells.slice(0, 500).map(cell => ({
      lat: cell.lat,
      lon: cell.lon,
      damage_class: cell.damage_class,
      id: cell.id || cell.img_prefix || 'unknown',
    }));
  }, [apiData, isLive]);

  if (!loading && !isLive) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={48} className="text-secondary animate-spin opacity-30" />
        <div className="font-mono text-sm text-gray-500 tracking-widest">AWAITING VISION DATA</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col relative">
      {/* Map — full bleed */}
      <div className="absolute inset-0 z-0">
        <MapContainer
          center={[41.0082, 28.9784]}
          zoom={12}
          style={{ width: '100%', height: '100%', background: '#080810' }}
          zoomControl={false}
          attributionControl={false}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            maxZoom={19}
          />

          {/* All damage markers — always present */}
          {allCells.map((cell, i) => (
            <CircleMarker
              key={`dmg-${i}`}
              center={[cell.lat, cell.lon]}
              radius={DAMAGE_RADIUS[cell.damage_class] || 5}
              pathOptions={{
                fillColor: DAMAGE_COLORS[cell.damage_class] || '#888',
                fillOpacity: cell.damage_class === 'destroyed' ? 0.85 : cell.damage_class === 'major_damage' ? 0.7 : 0.5,
                color: DAMAGE_COLORS[cell.damage_class] || '#888',
                weight: cell.damage_class === 'destroyed' ? 2 : 1,
                opacity: 0.5,
              }}
            >
              <Popup>
                <div style={{
                  fontFamily: "'Share Tech Mono', monospace",
                  fontSize: 11, color: '#fff',
                  background: '#0a0a0a',
                  border: `1px solid ${DAMAGE_COLORS[cell.damage_class]}40`,
                  padding: 10, minWidth: 140,
                }}>
                  <div style={{ color: DAMAGE_COLORS[cell.damage_class], fontWeight: 'bold', marginBottom: 4 }}>
                    {(cell.damage_class || '').replace('_', ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#888', fontSize: 10 }}>ID: {cell.id}</div>
                  <div style={{ color: '#888', fontSize: 10 }}>Coords: {cell.lat.toFixed(4)}, {cell.lon.toFixed(4)}</div>
                </div>
              </Popup>
            </CircleMarker>
          ))}

          <MapFitter cells={allCells} />

          <style>{`
            .leaflet-popup-content-wrapper { background: transparent; padding: 0; box-shadow: none; border: none; }
            .leaflet-popup-tip { display: none; }
          `}</style>
        </MapContainer>
      </div>

      {/* ── Header HUD ─────────────────────────────────────────────── */}
      <div className="absolute top-4 left-4 right-4 z-[500] flex justify-between items-start pointer-events-none">
        <div className="flex items-center gap-3 backdrop-blur-xl bg-black/70 p-3 rounded border border-glass_border shadow-[0_4px_30px_rgba(0,0,0,0.5)] pointer-events-auto">
          <div className="w-10 h-10 rounded bg-secondary/10 border border-secondary/30 flex items-center justify-center text-secondary"><Eye size={20} /></div>
          <div>
            <h1 className="text-xl font-mono text-secondary tracking-widest">GEO-VISION 3D</h1>
            <p className="text-gray-300 text-[10px] font-mono tracking-widest uppercase">REGION: {activeStage} • {stats.totalCells} CELLS ANALYZED</p>
          </div>
        </div>


      </div>

      {/* ── Bottom Impact Assessment ────────────────────────────────── */}
      <div className="absolute bottom-6 left-6 z-[500] pointer-events-none">
        <div className="glass-panel p-5 pointer-events-auto backdrop-blur-xl bg-black/70 w-72 border border-glass_border shadow-[0_8px_32px_rgba(0,0,0,0.6)]">
          <h3 className="font-mono text-[11px] text-gray-400 tracking-widest mb-4 flex items-center gap-2">
            <Activity size={14} className="text-secondary"/> IMPACT ASSESSMENT
          </h3>
          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between border-b border-white/10 pb-2">
              <span className="text-gray-400">TOTAL CELLS</span>
              <span className="text-white text-sm font-bold">{stats.totalCells}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#ff003c] flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-[#ff003c]"></div> DESTROYED</span>
              <span className="text-white font-bold">{stats.destroyed}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#ff6b00] flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-[#ff6b00]"></div> MAJOR DMG</span>
              <span className="text-white font-bold">{stats.major}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#ffb000] flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-[#ffb000]"></div> MINOR DMG</span>
              <span className="text-white font-bold">{stats.minor}</span>
            </div>
          </div>
          <div className="flex h-2 mt-4 gap-0.5 overflow-hidden rounded">
            {stats.totalCells > 0 && <>
              <div className="bg-[#ff003c]" style={{width:`${(stats.destroyed/stats.totalCells)*100}%`}}></div>
              <div className="bg-[#ff6b00]" style={{width:`${(stats.major/stats.totalCells)*100}%`}}></div>
              <div className="bg-[#ffb000]" style={{width:`${(stats.minor/stats.totalCells)*100}%`}}></div>
              <div className="bg-[#00ff88]" style={{width:`${Math.max(0,100-(stats.destroyed+stats.major+stats.minor)/stats.totalCells*100)}%`}}></div>
            </>}
          </div>
        </div>
      </div>
    </div>
  );
}
