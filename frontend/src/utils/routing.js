// OSRM routing utility — fetches actual road-following routes
// Uses the free public OSRM demo server

const OSRM_BASE = 'https://router.project-osrm.org/route/v1/driving';

/**
 * Fetch a real road-following route between waypoints using OSRM.
 * @param {Array<[lat, lon]>} waypoints - Array of [lat, lon] pairs
 * @returns {Promise<Array<[lat, lon]>>} - Array of [lat, lon] pairs forming the route
 */
export async function fetchRoute(waypoints) {
  if (!waypoints || waypoints.length < 2) return waypoints || [];
  
  // OSRM expects lon,lat format (not lat,lon)
  const coords = waypoints.map(([lat, lon]) => `${lon},${lat}`).join(';');
  const url = `${OSRM_BASE}/${coords}?overview=full&geometries=geojson`;
  
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`OSRM error: ${res.status}`);
    const data = await res.json();
    
    if (data.code !== 'Ok' || !data.routes || data.routes.length === 0) {
      console.warn('OSRM returned no route, falling back to straight line');
      return waypoints;
    }
    
    // GeoJSON coordinates are [lon, lat], convert to [lat, lon] for Leaflet
    const geom = data.routes[0].geometry.coordinates;
    return geom.map(([lon, lat]) => [lat, lon]);
  } catch (err) {
    console.warn('OSRM routing failed, using straight line:', err.message);
    return waypoints;
  }
}

/**
 * Fetch route and return it along with distance and duration.
 * @param {Array<[lat, lon]>} waypoints
 * @returns {Promise<{route: Array<[lat, lon]>, distance: number, duration: number}>}
 */
export async function fetchRouteWithMeta(waypoints) {
  if (!waypoints || waypoints.length < 2) return { route: waypoints || [], distance: 0, duration: 0 };
  
  const coords = waypoints.map(([lat, lon]) => `${lon},${lat}`).join(';');
  const url = `${OSRM_BASE}/${coords}?overview=full&geometries=geojson`;
  
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`OSRM error: ${res.status}`);
    const data = await res.json();
    
    if (data.code !== 'Ok' || !data.routes || data.routes.length === 0) {
      return { route: waypoints, distance: 0, duration: 0 };
    }
    
    const r = data.routes[0];
    const geom = r.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    return { route: geom, distance: r.distance, duration: r.duration };
  } catch (err) {
    console.warn('OSRM routing failed:', err.message);
    return { route: waypoints, distance: 0, duration: 0 };
  }
}

/**
 * Batch-fetch multiple routes.
 * @param {Array<{from: [lat,lon], to: [lat,lon], via?: [lat,lon]}>} requests
 * @returns {Promise<Array<Array<[lat, lon]>>>}
 */
export async function fetchRoutesBatch(requests) {
  const promises = requests.map(req => {
    const waypoints = req.via ? [req.from, req.via, req.to] : [req.from, req.to];
    return fetchRoute(waypoints);
  });
  return Promise.all(promises);
}
