/**
 * UrbanCityOS — React Hooks for API Integration
 * ================================================
 * useApi          — generic polling hook with loading/error/data
 * useLiveSocket   — WebSocket hook for real-time tick payloads
 * useConnectionStatus — LIVE/OFFLINE mode detection
 * useSystemStatus — Global system telemetry via WS
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { checkHealth, createLiveSocket } from '../utils/apiClient';

// ── useApi: Polling Hook ─────────────────────────────────────────────

export function useApi(fetcher, intervalMs = 10000, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const doFetch = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      if (result !== null) {
        setData(result);
        setError(null);
      } else {
        setError('API unreachable');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    doFetch();
    if (intervalMs > 0) {
      const timer = setInterval(doFetch, intervalMs);
      return () => clearInterval(timer);
    }
  }, [doFetch, intervalMs, ...deps]);

  return { data, loading, error, refetch: doFetch };
}


// ── useLiveSocket: WebSocket Hook (new tick format) ──────────────────

/**
 * Real-time WebSocket hook.
 * Returns latest snapshot, rolling buffer, system status, and connection status.
 * Handles the new 'tick' payload with { snapshot, system_status, recent_reports, dispatch_summary }.
 */
export function useLiveSocket(bufferSize = 60) {
  const [latest, setLatest] = useState(null);
  const [buffer, setBuffer] = useState([]);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [systemStatus, setSystemStatus] = useState(null);
  const [dispatchSummary, setDispatchSummary] = useState(null);
  const [latestReports, setLatestReports] = useState([]);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = createLiveSocket(
      // onTick — receives full tick payload
      (tickPayload) => {
        const snapshot = tickPayload?.snapshot || tickPayload;
        setLatest(snapshot);
        setBuffer(prev => {
          const next = [...prev, snapshot];
          return next.length > bufferSize ? next.slice(-bufferSize) : next;
        });
        if (tickPayload?.system_status) {
          setSystemStatus(tickPayload.system_status);
        }
        if (tickPayload?.dispatch_summary) {
          setDispatchSummary(tickPayload.dispatch_summary);
        }
        if (tickPayload?.recent_reports) {
          setLatestReports(tickPayload.recent_reports);
        }
      },
      // onBackfill
      (history) => {
        setBuffer(prev => {
          const combined = [...history, ...prev];
          return combined.length > bufferSize ? combined.slice(-bufferSize) : combined;
        });
        if (history.length > 0) {
          setLatest(history[history.length - 1]);
        }
      },
      // onStatusChange
      (status) => {
        setWsStatus(status);
      },
      // onSystemStatus (from pings/backfill)
      (ss) => {
        setSystemStatus(ss);
      }
    );

    socketRef.current = socket;
    return () => socket.close();
  }, [bufferSize]);

  const setSpeed = useCallback((speed) => {
    socketRef.current?.setSpeed(speed);
  }, []);

  return { latest, buffer, wsStatus, systemStatus, dispatchSummary, latestReports, setSpeed };
}


// ── useConnectionStatus: LIVE/OFFLINE Detection ──────────────────────

export function useConnectionStatus() {
  const [mode, setMode] = useState('CHECKING');
  const [healthData, setHealthData] = useState(null);

  useEffect(() => {
    async function check() {
      const result = await checkHealth();
      if (result) {
        setMode('LIVE');
        setHealthData(result);
      } else {
        setMode('OFFLINE');
        setHealthData(null);
      }
    }

    check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);

  return { mode, healthData };
}
