/**
 * UrbanCityOS — API Client
 * =========================
 * Centralized fetch wrappers and WebSocket manager.
 * ALL data comes from the live backend — no demo fallbacks.
 *
 * Environment-aware: auto-connects to cloud backend on GitHub Pages,
 * or uses local proxy in development.
 */

// Cloud backend URL (Render.com deployment)
const RENDER_BACKEND = 'https://urbanfl-api.onrender.com';

// Auto-detect: GitHub Pages → cloud backend, localhost → vite proxy
const IS_GITHUB_PAGES = window.location.hostname.includes('github.io');
const API_ORIGIN = IS_GITHUB_PAGES ? RENDER_BACKEND : '';
const BASE = `${API_ORIGIN}/api/v1`;

// ── Connection State ─────────────────────────────────────────────────
let _isLive = null;

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      _isLive = true;
      return await res.json();
    }
    _isLive = false;
    return null;
  } catch {
    _isLive = false;
    return null;
  }
}

export function isLive() {
  return _isLive === true;
}

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${BASE}${path}`, {
      signal: AbortSignal.timeout(8000),
      ...options,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _isLive = true;
    return await res.json();
  } catch (err) {
    console.warn(`[API] ${path} failed:`, err.message);
    _isLive = false;
    return null;
  }
}

async function apiPost(path, body) {
  return apiFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// ── Dashboard ────────────────────────────────────────────────────────
export const fetchDashboard = () => apiFetch('/dashboard');
export const fetchDashboardHistory = (n = 50) => apiFetch(`/dashboard/history?n=${n}`);

// ── System ───────────────────────────────────────────────────────────
export const fetchSystemStatus = () => apiFetch('/system/status');
export const fetchSystemAlerts = () => apiFetch('/system/alerts');
export const fetchSiloHealthHistory = () => apiFetch('/system/silo-health-history');

// ── Silos ────────────────────────────────────────────────────────────
export const fetchSiloStatus = () => apiFetch('/silos/status');
export const fetchNLPFeed = (limit = 30) => apiFetch(`/silos/nlp/feed?limit=${limit}`);
export const fetchVisionAssessment = () => apiFetch('/silos/vision/assessment');
export const fetchInfraStatus = () => apiFetch('/silos/infra/status');
export const fetchTrafficSegments = () => apiFetch('/silos/infra/traffic');

// ── Dispatch ─────────────────────────────────────────────────────────
export const fetchDispatchUnits = () => apiFetch('/dispatch/units');
export const fetchDispatchIncidents = () => apiFetch('/dispatch/incidents');
export const fetchDispatchHospitals = () => apiFetch('/dispatch/hospitals');
export const fetchDispatchActivityLog = () => apiFetch('/dispatch/activity-log');
export const triggerPipeline = () => apiPost('/dispatch/run', {});

// ── Zones ────────────────────────────────────────────────────────────
export const fetchZones = () => apiFetch('/zones');
export const fetchTriage = () => apiFetch('/zones/triage');
export const fetchZoneDetail = (zoneId) => apiFetch(`/zones/${zoneId}`);

// ── Analytics ────────────────────────────────────────────────────────
export const fetchRiskForecast = () => apiFetch('/analytics/risk-forecast');
export const runScenario = (scenario) => apiPost('/analytics/scenario', scenario);
export const fetchCasualtyEstimate = () => apiFetch('/analytics/casualty-estimate');

// ── Command Center ───────────────────────────────────────────────────
export const fetchSitrep = () => apiFetch('/command/sitrep');
export const fetchMissions = () => apiFetch('/command/missions');
export const createMission = (data) => apiPost('/command/missions', data);
export const fetchAgencies = () => apiFetch('/command/agencies');
export const fetchDecisions = () => apiFetch('/command/decisions');
export const fetchHospitals = () => apiFetch('/command/hospitals');
export const fetchResources = () => apiFetch('/command/resources');

// ── Simulation ───────────────────────────────────────────────────────
export const fetchSimStatus = () => apiFetch('/simulation/status');
export const setSimSpeed = (speed) => apiPost('/simulation/speed', { speed });
export const pauseSim = () => apiPost('/simulation/pause', {});
export const resumeSim = () => apiPost('/simulation/resume', {});


// ══════════════════════════════════════════════════════════════════════
// WebSocket Manager
// ══════════════════════════════════════════════════════════════════════

/**
 * Create a managed WebSocket connection with auto-reconnect.
 * Now handles the richer 'tick' payload format from the backend.
 *
 * @param {function} onTick - Called with each tick payload { snapshot, system_status, recent_reports, dispatch_summary }
 * @param {function} onBackfill - Called with array of historical snapshots
 * @param {function} onStatusChange - Called with 'connected'|'disconnected'|'reconnecting'
 * @param {function} onSystemStatus - Called with system_status dict on pings/backfill
 * @returns {{ close: function, setSpeed: function }}
 */
export function createLiveSocket(onTick, onBackfill, onStatusChange, onSystemStatus) {
  let ws = null;
  let reconnectDelay = 1000;
  let shouldReconnect = true;
  let reconnectTimer = null;

  function getWsUrl() {
    if (IS_GITHUB_PAGES) {
      // Connect to cloud backend via WSS
      return `wss://urbanfl-api.onrender.com/api/v1/ws/live`;
    }
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/api/v1/ws/live`;
  }

  function connect() {
    try {
      ws = new WebSocket(getWsUrl());
    } catch {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      reconnectDelay = 1000;
      onStatusChange?.('connected');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'tick') {
          onTick?.(msg.data);
        } else if (msg.type === 'backfill') {
          onBackfill?.(msg.data);
          if (msg.system_status) onSystemStatus?.(msg.system_status);
        } else if (msg.type === 'ping') {
          if (msg.system_status) onSystemStatus?.(msg.system_status);
        } else if (msg.type === 'speed_changed') {
          // Handled by caller if needed
        }
      } catch (e) {
        console.warn('[WS] Parse error:', e);
      }
    };

    ws.onclose = () => {
      onStatusChange?.('disconnected');
      if (shouldReconnect) scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function scheduleReconnect() {
    onStatusChange?.('reconnecting');
    reconnectTimer = setTimeout(() => {
      reconnectDelay = Math.min(reconnectDelay * 2, 30000);
      connect();
    }, reconnectDelay);
  }

  function close() {
    shouldReconnect = false;
    clearTimeout(reconnectTimer);
    ws?.close();
  }

  function setSpeed(speed) {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'set_speed', speed }));
    }
  }

  connect();
  return { close, setSpeed };
}
