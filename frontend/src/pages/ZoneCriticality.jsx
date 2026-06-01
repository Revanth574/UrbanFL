import { useState, useMemo, useRef, useCallback } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchZones } from '../utils/apiClient';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, AlertCircle, X, Eye, Loader2, Shield, Activity, Users, Crosshair } from 'lucide-react';

const DIMENSION_LABELS = {
  casualty_density: 'Casualty Density',
  structural_damage: 'Structural Damage',
  population_density: 'Population Density',
  infrastructure_loss: 'Infrastructure Loss',
  accessibility: 'Accessibility Risk',
  medical_proximity: 'Medical Proximity',
  secondary_hazard: 'Secondary Hazard',
  time_without_contact: 'Time w/o Contact',
  environmental_risk: 'Environmental Risk',
};

const CRITICALITY = {
  P1: { label: 'CRITICAL', color: '#FF0000', bg: 'rgba(255,0,0,0.1)', desc: 'Deploy Immediately' },
  P2: { label: 'SEVERE', color: '#ff6b00', bg: 'rgba(255,107,0,0.1)', desc: 'Deploy <15min' },
  P3: { label: 'HIGH', color: '#FFCC00', bg: 'rgba(255,204,0,0.1)', desc: 'Deploy <1hr' },
  P4: { label: 'MODERATE', color: '#00CC00', bg: 'rgba(0,204,0,0.1)', desc: 'Deploy <4hr' },
  P5: { label: 'LOW', color: '#0066FF', bg: 'rgba(0,102,255,0.1)', desc: 'Non-Urgent' },
};

// xBD imagery for sector cards
const XBD_SECTOR_IMAGES = [
  { pre: '/xbd/guatemala-volcano_00000004_pre_disaster.png', post: '/xbd/guatemala-volcano_00000004_post_disaster.png' },
  { pre: '/xbd/guatemala-volcano_00000012_pre_disaster.png', post: '/xbd/guatemala-volcano_00000012_post_disaster.png' },
  { pre: '/xbd/guatemala-volcano_00000014_pre_disaster.png', post: '/xbd/guatemala-volcano_00000014_post_disaster.png' },
  { pre: '/xbd/guatemala-volcano_00000020_pre_disaster.png', post: '/xbd/guatemala-volcano_00000020_post_disaster.png' },
  { pre: '/xbd/guatemala-volcano_00000022_pre_disaster.png', post: '/xbd/guatemala-volcano_00000022_post_disaster.png' },
  { pre: '/xbd/hurricane-florence_00000006_pre_disaster.png', post: '/xbd/hurricane-florence_00000006_post_disaster.png' },
  { pre: '/xbd/hurricane-florence_00000009_pre_disaster.png', post: '/xbd/hurricane-florence_00000009_post_disaster.png' },
  { pre: '/xbd/hurricane-florence_00000010_pre_disaster.png', post: '/xbd/hurricane-florence_00000010_post_disaster.png' },
  { pre: '/xbd/hurricane-florence_00000011_pre_disaster.png', post: '/xbd/hurricane-florence_00000011_post_disaster.png' },
  { pre: '/xbd/hurricane-florence_00000014_pre_disaster.png', post: '/xbd/hurricane-florence_00000014_post_disaster.png' },
];

/* ── Before/After Image Comparison Slider ───────────────────────────── */
function ImageCompareSlider({ preImg, postImg }) {
  const containerRef = useRef(null);
  const [sliderPos, setSliderPos] = useState(50);
  const isDragging = useRef(false);

  const handleMove = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  }, []);

  const onMouseDown = () => { isDragging.current = true; };
  const onMouseUp = () => { isDragging.current = false; };
  const onMouseMove = (e) => { if (isDragging.current) handleMove(e.clientX); };
  const onTouchMove = (e) => { handleMove(e.touches[0].clientX); };

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden bg-black aspect-video cursor-col-resize select-none border border-glass_border"
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onMouseMove={onMouseMove}
      onTouchStart={onMouseDown}
      onTouchEnd={onMouseUp}
      onTouchMove={onTouchMove}
    >
      {/* POST (bottom layer — visible on the right) */}
      <img src={postImg} alt="Post-disaster" className="absolute inset-0 w-full h-full object-cover" draggable={false} />

      {/* PRE (top layer — clipped on the left) */}
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderPos}%` }}>
        <img src={preImg} alt="Pre-disaster" className="absolute inset-0 w-full h-full object-cover" style={{ minWidth: containerRef.current ? containerRef.current.offsetWidth : '100%' }} draggable={false} />
      </div>

      {/* Slider line */}
      <div className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]" style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}>
        {/* Handle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white/90 rounded-full border-2 border-white shadow-[0_0_15px_rgba(255,255,255,0.5)] flex items-center justify-center">
          <div className="flex gap-0.5">
            <div className="w-0.5 h-3 bg-black/60 rounded"></div>
            <div className="w-0.5 h-3 bg-black/60 rounded"></div>
          </div>
        </div>
      </div>

      {/* Labels */}
      <div className="absolute top-2 left-2 bg-black/80 backdrop-blur-md px-2 py-1 text-[9px] font-mono z-10 border border-white/10 text-primary">PRE-DISASTER</div>
      <div className="absolute top-2 right-2 bg-black/80 backdrop-blur-md px-2 py-1 text-[9px] font-mono z-10 border border-white/10 text-danger">ACTIVE DAMAGE</div>
    </div>
  );
}

export default function ZoneCriticality() {
  const { data: apiZoneData, loading } = useApi(fetchZones, 15000);
  const isLive = apiZoneData !== null;

  const [weights, setWeights] = useState({
    casualty_density: 1.0,
    structural_damage: 1.0,
    population_density: 0.8,
    infrastructure_loss: 0.7,
  });

  // Build zones from API data with dynamic weighting
  const zones = useMemo(() => {
    if (!isLive || !apiZoneData?.zones?.length) return [];

    return apiZoneData.zones.map((z, i) => {
      const dims = z.dimension_scores || {};

      // Calculate weighted score
      let weightedTotal = 0;
      let weightSum = 0;
      Object.entries(weights).forEach(([key, w]) => {
        weightedTotal += (dims[key] || 0) * w;
        weightSum += w;
      });
      const composite = weightSum > 0 ? (weightedTotal / weightSum) : (z.composite_score || 0);

      const level = composite > 0.7 ? 'P1' : composite > 0.5 ? 'P2' : composite > 0.3 ? 'P3' : 'P4';
      const casualties = z.estimated_casualties ?? 0;
      const trapped = z.estimated_trapped ?? 0;

      const cascading = z.cascading_risks || [];

      return {
        id: z.zone_id || `SECTOR-${i}`,
        composite,
        level,
        dims,
        casualties,
        trapped,
        cascading,
        resources: z.resources || { ambulances: 2, fire_engines: 1, police: 1, search_rescue: 1 },
        accessibility: z.accessibility ?? (1 - (dims.accessibility || 0)),
        population: z.population || 0,
        lat: z.lat || 41.05,
        lon: z.lon || 29.00,
        imagery: XBD_SECTOR_IMAGES[i % XBD_SECTOR_IMAGES.length],
      };
    }).sort((a, b) => b.composite - a.composite);
  }, [apiZoneData, isLive, weights]);

  const [expandedZone, setExpandedZone] = useState(null);
  const [filterLevel, setFilterLevel] = useState(null);

  const expanded = expandedZone ? zones.find(z => z.id === expandedZone) : null;

  const handleWeightChange = (dim, val) => {
    setWeights(prev => ({ ...prev, [dim]: parseFloat(val) }));
  };

  // Offline state
  if (!loading && !isLive) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={48} className="text-danger animate-spin opacity-30" />
        <div className="font-mono text-sm text-gray-500 tracking-widest">AWAITING ZONE DATA</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-4">
      <header className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded bg-danger/10 border border-danger/30 flex items-center justify-center text-danger"><Target size={20} /></div>
          <div>
            <h1 className="text-2xl font-mono text-danger tracking-widest">THREAT MATRIX</h1>
            <p className="text-gray-500 text-xs font-mono uppercase">Analytics // 8-Dimensional Zone Criticality • Satellite Imagery</p>
          </div>
        </div>


        <div className="flex gap-2">
          <button onClick={() => setFilterLevel(null)} className={`px-3 py-1 font-mono text-xs border transition-all ${!filterLevel ? 'border-primary text-primary bg-primary/10' : 'border-glass_border text-gray-500'}`}>ALL ({zones.length})</button>
          {Object.entries(CRITICALITY).map(([k, v]) => (
            <button key={k} onClick={() => setFilterLevel(k)} className="px-3 py-1 font-mono text-xs border transition-all" style={filterLevel === k ? { borderColor: v.color, color: v.color, backgroundColor: v.bg } : { borderColor: '#1a1a1a', color: '#666' }}>{v.label} ({zones.filter(z => z.level === k).length})</button>
          ))}
        </div>
      </header>


      <div className="flex-1 flex gap-4 overflow-hidden">
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 relative z-0">
          <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', alignContent: 'start' }}>
            {zones.map((zone, i) => {
              const crit = CRITICALITY[zone.level] || CRITICALITY.P5;
              const show = !filterLevel || zone.level === filterLevel;
              if (!show) return null;

              return (
                <motion.div
                  layout key={zone.id}
                  initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: Math.min(i * 0.01, 0.5) }}
                  onClick={() => { setExpandedZone(zone.id === expandedZone ? null : zone.id); }}
                  className="border cursor-pointer transition-all relative group overflow-hidden"
                  style={{
                    borderColor: expandedZone === zone.id ? crit.color : '#333',
                    height: '220px',
                    boxShadow: expandedZone === zone.id ? `0 0 20px ${crit.color}40` : zone.level === 'P1' ? `0 0 10px ${crit.color}20` : 'none'
                  }}
                >
                  <div className="absolute inset-0 bg-black">
                    <img src={zone.imagery.post} alt="" className="w-full h-full object-cover opacity-30 group-hover:opacity-60 transition-opacity duration-700" loading="lazy" />
                    <div className="absolute inset-0 mix-blend-overlay" style={{ backgroundColor: crit.color, opacity: 0.3 }}></div>
                    <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent opacity-90"></div>
                  </div>

                  <div className="relative h-full p-4 flex flex-col justify-between z-10 pointer-events-none">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-mono text-lg font-bold text-white tracking-widest">{zone.id.replace('zone_', 'SEC-').replace('ZONE_', 'SEC-').toUpperCase()}</div>
                        <div className="font-mono text-[10px] tracking-widest" style={{ color: crit.color }}>{crit.label} • {Math.round(zone.composite * 100)}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-2xl font-bold" style={{ color: crit.color }}>{zone.casualties}</div>
                        <div className="font-mono text-[9px] text-gray-400">CASUALTIES</div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-end">
                        <div className="font-mono text-[10px] text-gray-300">TRAPPED</div>
                        <div className="font-mono text-sm text-yellow-400 font-bold">{zone.trapped}</div>
                      </div>
                      <div className="flex justify-between items-end">
                        <div className="font-mono text-[10px] text-gray-300">ACCESS</div>
                        <div className="font-mono text-sm text-white font-bold">{(zone.accessibility * 100).toFixed(0)}%</div>
                      </div>
                      <div className="h-1 w-full bg-white/10 mt-2 overflow-hidden">
                        <div className="h-full" style={{ width: `${zone.composite * 100}%`, backgroundColor: crit.color }}></div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
          {zones.filter(z => !filterLevel || z.level === filterLevel).length === 0 && (
            <div className="w-full py-20 text-center font-mono text-gray-600 tracking-widest">
              NO SECTORS MATCH CURRENT FILTER
            </div>
          )}
        </div>

        {/* Expanded Detail Panel */}
        <AnimatePresence mode="wait">
          {expanded && (
            <motion.div key={expanded.id} initial={{ opacity: 0, x: 20, width: 0 }} animate={{ opacity: 1, x: 0, width: 380 }} exit={{ opacity: 0, x: 20, width: 0 }} className="glass-panel p-4 overflow-y-auto custom-scrollbar flex flex-col border-l border-glass_border" style={{ minWidth: 380 }}>
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-mono text-sm font-bold tracking-widest text-white">{expanded.id.replace('zone_', 'SEC-').replace('ZONE_', 'SEC-').toUpperCase()}</h3>
                <button onClick={() => setExpandedZone(null)} className="text-gray-500 hover:text-white p-1 bg-white/5 rounded"><X size={16} /></button>
              </div>

              <div className="p-3 border text-center font-mono mb-4" style={{ borderColor: CRITICALITY[expanded.level].color, backgroundColor: CRITICALITY[expanded.level].bg, color: CRITICALITY[expanded.level].color }}>
                <div className="font-bold text-lg mb-1">{CRITICALITY[expanded.level].label}</div>
                <div className="text-xs text-white/70">{CRITICALITY[expanded.level].desc}</div>
                <div className="text-4xl mt-3 font-light text-white">{Math.round(expanded.composite * 100)}<span className="text-lg opacity-50 ml-1">IDX</span></div>
              </div>

              {/* Before/After Satellite Comparison Slider */}
              <div className="mb-5">
                <div className="flex items-center gap-2 mb-2 border-b border-glass_border pb-1">
                  <Eye size={12} className="text-secondary" />
                  <h4 className="font-mono text-[10px] text-secondary tracking-widest">SATELLITE INTEL — DRAG TO COMPARE</h4>
                </div>
                <ImageCompareSlider preImg={expanded.imagery.pre} postImg={expanded.imagery.post} />
              </div>

              {/* Dimension Bars */}
              <div className="space-y-2.5 mb-5">
                <h4 className="font-mono text-[10px] text-gray-400 tracking-widest border-b border-glass_border pb-1">DIMENSION ANALYSIS</h4>
                {Object.entries(expanded.dims).map(([key, val]) => (
                  <div key={key}>
                    <div className="flex justify-between font-mono text-[10px] text-gray-300 mb-1"><span>{DIMENSION_LABELS[key] || key}</span><span>{(val * 100).toFixed(0)}%</span></div>
                    <div className="h-1.5 bg-[#111] overflow-hidden"><div className="h-full transition-all duration-1000 ease-out" style={{ width: `${val * 100}%`, backgroundColor: val > 0.7 ? '#ff003c' : val > 0.4 ? '#ffb000' : '#00f3ff' }}></div></div>
                  </div>
                ))}
              </div>

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-2 mb-5">
                <div className="glass-panel p-3 text-center">
                  <div className="text-xl font-mono font-bold text-danger">{expanded.casualties}</div>
                  <div className="text-[8px] font-mono text-gray-500 tracking-widest">CASUALTIES</div>
                </div>
                <div className="glass-panel p-3 text-center">
                  <div className="text-xl font-mono font-bold text-secondary">{expanded.trapped}</div>
                  <div className="text-[8px] font-mono text-gray-500 tracking-widest">TRAPPED</div>
                </div>
                <div className="glass-panel p-3 text-center">
                  <div className="text-xl font-mono font-bold text-primary">{(expanded.accessibility * 100).toFixed(0)}%</div>
                  <div className="text-[8px] font-mono text-gray-500 tracking-widest">ACCESS</div>
                </div>
              </div>

              {/* Cascading Risks */}
              {expanded.cascading.length > 0 && (
                <div className="mb-5">
                  <h4 className="font-mono text-[10px] text-danger tracking-widest mb-2 border-b border-danger/20 pb-1">CASCADE RISKS</h4>
                  {expanded.cascading.map((risk, i) => (
                    <div key={i} className="p-2 border-l-2 border-danger bg-danger/5 text-danger text-[10px] font-mono mb-2 flex items-start gap-2">
                      <AlertCircle size={12} className="flex-shrink-0 mt-0.5" />{risk}
                    </div>
                  ))}
                </div>
              )}

              {/* Resources */}
              <div className="mt-auto">
                <h4 className="font-mono text-[10px] text-primary tracking-widest mb-2 border-b border-primary/20 pb-1">RESOURCE ALLOCATION</h4>
                <div className="grid grid-cols-2 gap-2 font-mono text-[10px] text-gray-400">
                  {Object.entries(expanded.resources).map(([k, v]) => (
                    <div key={k} className="bg-black/30 border border-glass_border p-2 flex justify-between items-center">
                      <span>{k.replace('_', ' ').toUpperCase()}</span>
                      <span className="text-white font-bold text-sm">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
