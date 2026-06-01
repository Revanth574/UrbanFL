import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, ChevronUp, ChevronDown, Gauge, Zap } from 'lucide-react';

const SPEEDS = [0.5, 1, 2, 5, 10];

export default function SimControls({ systemStatus, onSetSpeed }) {
  const [expanded, setExpanded] = useState(false);

  const speed = systemStatus?.simulation_speed ?? 1;
  const paused = systemStatus?.paused ?? false;
  const eventsRemaining = systemStatus?.events_remaining ?? 0;
  const recycled = systemStatus?.events_recycled ?? 0;
  const tickCount = systemStatus?.tick_count ?? 0;

  const handleSpeed = (s) => {
    onSetSpeed?.(s);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed bottom-6 right-6 z-50 font-mono"
    >
      {/* Expand/collapse */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            className="mb-2 backdrop-blur-xl bg-black/80 border border-glass_border rounded-lg p-4 shadow-[0_8px_32px_rgba(0,0,0,0.6)] w-56"
          >
            <div className="text-[10px] text-gray-400 tracking-widest mb-3">SIMULATION CONTROL</div>

            {/* Speed buttons */}
            <div className="flex gap-1 mb-3">
              {SPEEDS.map(s => (
                <button
                  key={s}
                  onClick={() => handleSpeed(s)}
                  className="flex-1 py-1.5 text-[10px] border transition-all rounded-sm"
                  style={speed === s && !paused ? {
                    borderColor: '#00f3ff',
                    color: '#00f3ff',
                    backgroundColor: 'rgba(0,243,255,0.1)',
                    boxShadow: '0 0 8px rgba(0,243,255,0.2)',
                  } : {
                    borderColor: '#1a1a1a',
                    color: '#666',
                  }}
                >
                  {s}x
                </button>
              ))}
            </div>

            {/* Stats */}
            <div className="space-y-1.5 text-[10px]">
              <div className="flex justify-between text-gray-500">
                <span>EVENTS IN QUEUE</span>
                <span className="text-white">{eventsRemaining}</span>
              </div>
              <div className="flex justify-between text-gray-500">
                <span>RECYCLE CYCLES</span>
                <span className="text-primary">{recycled}</span>
              </div>
              <div className="flex justify-between text-gray-500">
                <span>TICKS</span>
                <span className="text-gray-300">{tickCount}</span>
              </div>
            </div>

            {/* Event queue bar */}
            <div className="mt-3">
              <div className="h-1 bg-[#111] w-full rounded overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-500"
                  style={{ width: `${Math.min(100, (eventsRemaining / 205) * 100)}%` }}
                ></div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main button row */}
      <div className="flex items-center gap-2">
        {/* Expand toggle */}
        <button
          onClick={() => setExpanded(e => !e)}
          className="w-9 h-9 rounded-full border border-glass_border bg-black/80 backdrop-blur-xl flex items-center justify-center text-gray-400 hover:text-primary hover:border-primary/50 transition-all shadow-lg"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>

        {/* Play/Pause */}
        <button
          onClick={() => handleSpeed(paused ? 1 : 0)}
          className="w-12 h-12 rounded-full border flex items-center justify-center transition-all shadow-[0_4px_20px_rgba(0,0,0,0.5)] backdrop-blur-xl"
          style={paused ? {
            borderColor: '#ffb000',
            backgroundColor: 'rgba(255,176,0,0.15)',
            color: '#ffb000',
          } : {
            borderColor: '#00ff88',
            backgroundColor: 'rgba(0,255,136,0.1)',
            color: '#00ff88',
          }}
        >
          {paused ? <Play size={18} fill="currentColor" className="ml-0.5" /> : <Pause size={18} fill="currentColor" />}
        </button>

        {/* Speed indicator */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-glass_border bg-black/80 backdrop-blur-xl shadow-lg">
          <Zap size={10} className={paused ? 'text-secondary' : 'text-primary animate-pulse'} />
          <span className="text-[10px] tracking-widest" style={{ color: paused ? '#ffb000' : '#00f3ff' }}>
            {paused ? 'PAUSED' : `${speed}x`}
          </span>
          {/* Heartbeat dot */}
          {!paused && (
            <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse ml-1"></div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
