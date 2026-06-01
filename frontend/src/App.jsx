import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';

import { useLiveSocket } from './hooks/useApi';

import Dashboard from './pages/Dashboard';
import NLPFeed from './pages/NLPFeed';
import VisionSilo from './pages/VisionSilo';
import Infrastructure from './pages/Infrastructure';
import Dispatch from './pages/Dispatch';
import ZoneCriticality from './pages/ZoneCriticality';

function AppShell() {
  const { buffer, systemStatus } = useLiveSocket(60);

  return (
    <div className="flex h-screen w-screen bg-background text-white overflow-hidden relative font-sans">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,243,255,0.03),transparent_70%)] pointer-events-none z-0"></div>
      
      {/* CRT Scanline Overlay */}
      <div className="absolute inset-0 z-50 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-20"></div>

      <Sidebar systemStatus={systemStatus} />
      <div className="flex-1 flex flex-col relative z-10 w-full overflow-hidden">
        <Topbar systemStatus={systemStatus} />
        <main className="flex-1 overflow-auto p-8 relative">
           <div className="absolute inset-0 pointer-events-none opacity-[0.03] bg-[linear-gradient(to_right,#00f3ff_1px,transparent_1px),linear-gradient(to_bottom,#00f3ff_1px,transparent_1px)] bg-[size:3rem_3rem]"></div>
          
          <div className="relative z-10 h-full">
            <Routes>
              <Route path="/" element={<Dashboard buffer={buffer} systemStatus={systemStatus} />} />
              <Route path="/nlp" element={<NLPFeed />} />
              <Route path="/vision" element={<VisionSilo />} />
              <Route path="/infra" element={<Infrastructure />} />
              <Route path="/dispatch" element={<Dispatch />} />
              <Route path="/zones" element={<ZoneCriticality />} />
            </Routes>
          </div>
        </main>
      </div>


    </div>
  );
}

export default function App() {
  const basename = import.meta.env.BASE_URL;
  return (
    <BrowserRouter basename={basename}>
      <AppShell />
    </BrowserRouter>
  );
}
