import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Radio, AlertCircle, MessageSquare, ShieldCheck, ShieldAlert, Filter, X, Loader2 } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { fetchNLPFeed } from '../utils/apiClient';

const SEVERITY_STYLE = {
  CRITICAL: { border: 'border-danger', text: 'text-danger', bg: 'bg-danger/5', shadow: 'shadow-[0_0_10px_rgba(255,0,60,0.2)]' },
  HIGH: { border: 'border-secondary', text: 'text-secondary', bg: 'bg-secondary/5', shadow: '' },
  MEDIUM: { border: 'border-primary', text: 'text-primary', bg: 'bg-primary/5', shadow: '' },
  LOW: { border: 'border-gray-600', text: 'text-gray-400', bg: 'bg-black/20', shadow: '' },
};

export default function NLPFeed() {
  const { data: feedData, loading, error } = useApi(fetchNLPFeed, 4000);
  const [filter, setFilter] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [earlyWarnings, setEarlyWarnings] = useState([]);

  // All data from API — no demo fallback
  const reports = feedData?.reports || [];
  const stats = feedData?.stats || { total: 0, critical: 0, verified: 0, avg_credibility: 0 };

  // Extract early warnings from API data
  useEffect(() => {
    if (!feedData) return;
    const critical = reports.filter(r => r.severity === 'CRITICAL' && r.verified);
    if (critical.length > 0) {
      setEarlyWarnings(critical.slice(0, 5).map(r => ({
        id: r.id, time: r.time, message: `VERIFIED CRITICAL: ${r.text.slice(0, 60)}...`,
        type: r.type, location: r.location,
      })));
    }
  }, [feedData]);

  const filteredReports = filter ? reports.filter(r => r.severity === filter) : reports;

  // Offline state
  if (!loading && !feedData) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={48} className="text-primary animate-spin opacity-30" />
        <div className="font-mono text-sm text-gray-500 tracking-widest">AWAITING NLP FEED</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-4">
      <header className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded bg-primary/10 border border-primary/30 flex items-center justify-center text-primary"><Radio size={20} className="animate-pulse" /></div>
          <div>
            <h1 className="text-2xl font-mono text-primary tracking-widest">SIGINT // EARLY WARNING</h1>
            <p className="text-gray-500 text-xs font-mono">SILO 1 NLP • CRISIS INTERCEPT & VERIFICATION ENGINE • LIVE FEED</p>
          </div>
        </div>
        <div className="flex gap-2 font-mono text-xs">
          {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(sev => (
            <button key={sev} onClick={() => setFilter(f => f === sev ? null : sev)} className={`px-3 py-1 border transition-all ${filter === sev ? `${SEVERITY_STYLE[sev].border} ${SEVERITY_STYLE[sev].text} ${SEVERITY_STYLE[sev].bg}` : 'border-glass_border text-gray-500 hover:text-white'}`}>{sev}</button>
          ))}
        </div>
      </header>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-3 font-mono text-xs">
        <div className="glass-panel p-3 flex justify-between items-center"><span className="text-gray-500">INTERCEPTS</span><span className="text-primary text-lg">{stats.total}</span></div>
        <div className="glass-panel p-3 flex justify-between items-center border-danger/30"><span className="text-gray-500">CRITICAL</span><span className="text-danger text-lg animate-pulse">{stats.critical}</span></div>
        <div className="glass-panel p-3 flex justify-between items-center"><span className="text-gray-500">VERIFIED</span><span className="text-success text-lg">{stats.verified}</span></div>
        <div className="glass-panel p-3 flex justify-between items-center"><span className="text-gray-500">AVG CREDIBILITY</span><span className="text-primary text-lg">{((stats.avg_credibility || 0) * 100).toFixed(0)}%</span></div>
      </div>

      <div className="flex-1 flex gap-4 overflow-hidden">
        {/* Feed */}
        <div className="flex-1 glass-panel p-4 flex flex-col overflow-hidden">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(0,243,255,0.015)_1px,transparent_1px)] bg-[size:100%_2px] pointer-events-none z-0"></div>
          <div className="flex-1 overflow-y-auto space-y-2 relative z-10 pr-2 custom-scrollbar">
            {loading && reports.length === 0 ? (
              <div className="flex items-center justify-center py-12 text-gray-600 font-mono text-xs">
                <Loader2 size={16} className="animate-spin mr-2" /> LOADING FEED...
              </div>
            ) : (
              <AnimatePresence>
                {filteredReports.map((report) => {
                  const style = SEVERITY_STYLE[report.severity] || SEVERITY_STYLE.LOW;
                  return (
                    <motion.div
                      key={report.id}
                      initial={{ opacity: 0, x: -30, scale: 0.97 }}
                      animate={{ opacity: 1, x: 0, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.3 }}
                      onClick={() => setSelectedReport(report)}
                      className={`p-3 border ${style.border} ${style.bg} ${style.shadow} cursor-pointer hover:bg-white/5 transition-colors`}
                    >
                      <div className="flex items-center gap-3 text-xs font-mono">
                        <span className="text-gray-500 w-16">{report.time}</span>
                        <span className={`${style.text} w-20 font-bold tracking-widest flex items-center gap-1`}>
                          {report.severity === 'CRITICAL' && <AlertCircle size={12} className="animate-pulse" />}
                          {report.severity}
                        </span>
                        <span className="text-gray-400 w-20 flex items-center gap-1"><MessageSquare size={10} />{report.source}</span>
                        <span className="text-gray-500 w-16">[{report.location}]</span>
                        <span className="flex-1 text-gray-300 font-sans truncate">"{report.text}"</span>
                        {report.verified ? <ShieldCheck size={14} className="text-success flex-shrink-0" /> : <ShieldAlert size={14} className="text-gray-600 flex-shrink-0" />}
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            )}
          </div>
        </div>

        {/* Right Panel */}
        <div className="w-72 flex flex-col gap-4">
          {/* Early Warning */}
          <div className="glass-panel p-4 border-danger/30">
            <h3 className="text-danger font-mono text-xs tracking-widest mb-3 flex items-center gap-2"><AlertCircle size={14} /> EARLY WARNINGS</h3>
            {earlyWarnings.length === 0 ? (
              <div className="text-gray-500 font-mono text-xs text-center py-4">MONITORING...<br />NO VERIFIED CRITICAL EVENTS</div>
            ) : (
              <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar">
                {earlyWarnings.map(w => (
                  <div key={w.id} className="p-2 border border-danger/50 bg-danger/10 text-danger text-[10px] font-mono">
                    <div className="flex justify-between font-bold mb-1"><span>{(w.type || '').replace('_', ' ').toUpperCase()}</span><span>{w.time}</span></div>
                    <p className="opacity-80">{w.message}</p>
                    <div className="text-[9px] mt-1 opacity-60">{w.location}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Selected Detail */}
          <AnimatePresence mode="wait">
            {selectedReport ? (
              <motion.div key={selectedReport.id} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} className="glass-panel p-4 flex-1 overflow-y-auto custom-scrollbar font-mono text-xs">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-primary tracking-widest">REPORT #{String(selectedReport.id).slice(-6)}</h3>
                  <button onClick={() => setSelectedReport(null)} className="text-gray-500 hover:text-white"><X size={14} /></button>
                </div>
                <div className={`p-2 border text-center font-bold mb-3 ${(SEVERITY_STYLE[selectedReport.severity] || SEVERITY_STYLE.LOW).border} ${(SEVERITY_STYLE[selectedReport.severity] || SEVERITY_STYLE.LOW).text} ${(SEVERITY_STYLE[selectedReport.severity] || SEVERITY_STYLE.LOW).bg}`}>{selectedReport.severity}</div>
                <p className="text-gray-300 font-sans text-sm mb-3 border-l-2 border-primary/30 pl-3">"{selectedReport.text}"</p>
                <div className="space-y-2 text-gray-400">
                  <div className="flex justify-between">SOURCE<span className="text-white">{selectedReport.source}</span></div>
                  <div className="flex justify-between">TYPE<span className="text-white">{(selectedReport.type || '').replace('_', ' ')}</span></div>
                  <div className="flex justify-between">LOCATION<span className="text-white">{selectedReport.location}</span></div>
                  <div className="flex justify-between">NLP CONF.<span className="text-primary">{selectedReport.nlpConfidence != null ? `${(selectedReport.nlpConfidence * 100).toFixed(0)}%` : '—'}</span></div>
                  <div className="flex justify-between">CREDIBILITY<span className={selectedReport.credibility > 0.75 ? 'text-success' : 'text-secondary'}>{((selectedReport.credibility || 0) * 100).toFixed(0)}%</span></div>
                  <div className="flex justify-between">VERIFIED<span className={selectedReport.verified ? 'text-success' : 'text-gray-500'}>{selectedReport.verified ? 'YES' : 'PENDING'}</span></div>
                  <div className="flex justify-between">ENV MATCH<span className={selectedReport.envMatch ? 'text-success' : 'text-gray-500'}>{selectedReport.envMatch ? 'CONFIRMED' : 'NO MATCH'}</span></div>
                </div>
                {selectedReport.envMatch && (
                  <div className="mt-3 p-2 border border-success/30 bg-success/5 text-success text-[10px]">
                    ✓ Environment sensors corroborate this report (Silo 3 cross-reference positive)
                  </div>
                )}
              </motion.div>
            ) : (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-panel p-4 flex-1 flex flex-col items-center justify-center text-gray-500 font-mono text-xs text-center">
                <Filter size={28} className="mb-2 opacity-20" />
                CLICK ANY INTERCEPT<br />FOR FULL ANALYSIS
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
