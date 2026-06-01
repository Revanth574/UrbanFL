import { NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LayoutDashboard, MessageSquareText, Eye, Activity, ShieldAlert, Crosshair } from 'lucide-react';

const navItems = [
  { path: '/', label: 'COMMAND', icon: <LayoutDashboard size={18} />, desc: 'DASHBOARD' },
  { path: '/nlp', label: 'SIGINT', icon: <MessageSquareText size={18} />, desc: 'NLP FEED' },
  { path: '/vision', label: 'GEO-VIS', icon: <Eye size={18} />, desc: 'DAMAGE MAP' },
  { path: '/infra', label: 'INFRA', icon: <Activity size={18} />, desc: 'GRID • TRAFFIC' },
  { path: '/dispatch', label: 'DISPATCH', icon: <ShieldAlert size={18} />, desc: 'UNIT COORD' },
  { path: '/zones', label: 'THREAT', icon: <Crosshair size={18} />, desc: 'ZONE MATRIX' },
];

export default function Sidebar({ systemStatus }) {
  const location = useLocation();
  const epsilon = systemStatus?.privacy_epsilon != null
    ? systemStatus.privacy_epsilon.toFixed(2)
    : '—';

  return (
    <div className="w-56 h-full flex flex-col z-20 relative" style={{ background: 'rgba(5,5,8,0.95)', borderRight: '1px solid rgba(0,243,255,0.1)' }}>
      {/* Logo */}
      <div className="px-5 pt-6 pb-4 flex flex-col items-center">
        <div className="relative">
          <h1 className="text-2xl font-mono text-primary font-bold tracking-[0.25em] text-glow text-center">
            URBAN<span className="text-white">OS</span>
          </h1>
          <div className="text-[8px] font-mono text-primary/40 tracking-[0.5em] text-center mt-0.5">CITY COMMAND</div>
        </div>
        <div className="w-full h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent mt-3"></div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-1 px-3 mt-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink key={item.path} to={item.path} className="relative group">
              <div className={`flex items-center gap-3 px-3 py-2.5 rounded text-xs font-mono tracking-wider transition-all duration-300 relative overflow-hidden
                ${isActive
                  ? 'text-primary'
                  : 'text-gray-500 hover:text-gray-300'
                }
              `}>
                {isActive && (
                  <motion.div layoutId="activeNav" className="absolute left-0 top-0 bottom-0 w-0.5 bg-primary shadow-[0_0_8px_rgba(0,243,255,0.8)]" />
                )}
                {isActive && (
                  <motion.div layoutId="activeBg" className="absolute inset-0 bg-primary/8 border border-primary/20 rounded" style={{ zIndex: -1 }} />
                )}
                <span className={isActive ? 'text-primary' : 'text-gray-600 group-hover:text-gray-400 transition-colors'}>{item.icon}</span>
                <div className="flex flex-col">
                  <span className="leading-none">{item.label}</span>
                  <span className="text-[8px] text-gray-600 tracking-widest leading-tight mt-0.5">{item.desc}</span>
                </div>
              </div>
            </NavLink>
          );
        })}
      </nav>

      {/* Footer — live epsilon */}
      <div className="px-5 py-4 border-t border-glass_border">
        <div className="flex items-center gap-2 text-[9px] font-mono text-gray-600">
          <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse shadow-[0_0_4px_#00ff88]"></div>
          SECURE UPLINK ACTIVE
        </div>
        <div className="text-[8px] font-mono text-gray-700 mt-1">SYS v9.2.4 • ε={epsilon} DP</div>
      </div>
    </div>
  );
}
