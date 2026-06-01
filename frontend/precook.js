import fs from 'fs';
import { TRAFFIC_ROADS, DISPATCH_INCIDENTS, DISPATCH_HOSPITALS, UNIT_POSITIONS } from './src/data/istanbulGeo.js';

const OSRM_BASE = 'https://router.project-osrm.org/route/v1/driving';

async function fetchRoute(waypoints) {
  if (!waypoints || waypoints.length < 2) return waypoints;
  const coords = waypoints.map(([lat, lon]) => `${lon},${lat}`).join(';');
  const url = `${OSRM_BASE}/${coords}?overview=full&geometries=geojson`;
  
  try {
    const res = await fetch(url);
    if (!res.ok) {
        console.error(`OSRM error: ${res.status}`);
        return waypoints;
    }
    const data = await res.json();
    if (data.code !== 'Ok' || !data.routes || data.routes.length === 0) return waypoints;
    return data.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
  } catch (err) {
    console.error('OSRM API fail:', err.message);
    return waypoints;
  }
}

async function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  console.log('Pre-cooking TRAFFIC_ROADS...');
  const newRoads = [];
  for (const r of TRAFFIC_ROADS) {
    console.log(`Fetching route: ${r.name}`);
    
    // 1. Initial coords (Start to End directly)
    const initialWaypoints = [r.coords[0], r.coords[r.coords.length-1]];
    const realCoords = await fetchRoute(initialWaypoints);
    await delay(1500); // stay under OSRM limits
    
    // 2. Reroute coords (Start -> Mid detour -> End)
    let realRerouteCoords = r.rerouteCoords;
    if (r.rerouteCoords) {
      const midPoint = r.rerouteCoords[Math.floor(r.rerouteCoords.length / 2)];
      const rerouteWaypoints = [r.coords[0], midPoint, r.coords[r.coords.length-1]];
      realRerouteCoords = await fetchRoute(rerouteWaypoints);
      await delay(1500);
    }
    
    // 3. Alternative reroute (for "Re-optimize")
    // Let's create a third option using a different midpoint
    let realAltRerouteCoords = null;
    if (r.rerouteCoords && r.rerouteCoords.length > 3) {
      const midPoint2 = r.rerouteCoords[Math.floor(r.rerouteCoords.length / 4)]; // different detour
      const altWaypoints = [r.coords[0], midPoint2, r.coords[r.coords.length-1]];
      realAltRerouteCoords = await fetchRoute(altWaypoints);
      await delay(1500);
    }

    newRoads.push({
      ...r,
      coords: realCoords.length > 2 ? realCoords : r.coords,
      rerouteCoords: realRerouteCoords && realRerouteCoords.length > 2 ? realRerouteCoords : realRerouteCoords,
      altRerouteCoords: realAltRerouteCoords && realAltRerouteCoords.length > 2 ? realAltRerouteCoords : r.rerouteCoords
    });
  }

  // Pre-cook Dispatch assignments (since they use rng(seed), we know exact assignments)
  console.log('Pre-cooking DISPATCH routes...');
  const rng = (seed) => { let s=seed; return ()=>{ s=(s*16807)%2147483647; return(s-1)/2147483646; }; };
  
  const HOSPITALS_DATA = DISPATCH_HOSPITALS.map((h,i) => ({ ...h, beds: [450,320,350,280,200][i], utilized: [380,318,210,220,145][i] }));
  
  const cookedUnits = {};
  
  for (const type of ['medical', 'fire', 'police']) {
    let seed = type === 'medical' ? 101 : type === 'fire' ? 202 : 303;
    const rRng = rng(seed);
    const positions = UNIT_POSITIONS[type] || [];
    const thisTypeUnits = [];
    
    for (let i = 0; i < positions.length; i++) {
        const pos = positions[i];
        const inc = DISPATCH_INCIDENTS[Math.floor(rRng() * DISPATCH_INCIDENTS.length)];
        const hosp = type === 'medical' ? HOSPITALS_DATA[Math.floor(rRng() * HOSPITALS_DATA.length)] : null;
        const statusRandom = rRng();
        const status = statusRandom > 0.5 ? 'en_route' : statusRandom > 0.3 ? 'on_scene' : 'returning';
        
        console.log(`Fetching route: ${type} ${i} -> Incident`);
        const realRouteCoords = await fetchRoute([pos, [inc.lat, inc.lon]]);
        await delay(1000);
        
        let realReturnCoords = null;
        if (hosp) {
            console.log(`Fetching route: ${type} ${i} -> Hospital`);
            realReturnCoords = await fetchRoute([[inc.lat, inc.lon], [hosp.lat, hosp.lon]]);
            await delay(1000);
        }
        
        thisTypeUnits.push({
            id: `${type==='medical'?'AMB':type==='fire'?'FIR':'POL'}-${String(i+1).padStart(2,'0')}`,
            routeCoords: realRouteCoords,
            returnCoords: realReturnCoords
        });
        
        // Burn RNG cycles to match Dispatch.jsx calls:
        rRng(); // target ETA
        rRng(); // target Speed
        if (type === 'medical') rRng(); // stabilized
        if (type === 'fire') rRng(); // water
        if (type === 'police') rRng(); // sector
    }
    cookedUnits[type] = thisTypeUnits;
  }

  // Generate a file
  let outputFile = `export const PRECOOKED_ROADS = ${JSON.stringify(newRoads, null, 2)};\n`;
  outputFile += `export const PRECOOKED_UNITS = ${JSON.stringify(cookedUnits, null, 2)};\n`;
  
  fs.writeFileSync('src/data/osrmPrecooked.js', outputFile);
  console.log('Successfully wrote src/data/osrmPrecooked.js');
}

main();
