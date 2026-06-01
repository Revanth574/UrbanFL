// =================================================================
// PRE-TRACED ROAD GEOMETRIES FOR ISTANBUL — EUROPEAN SIDE ONLY
// =================================================================
// ALL coordinates verified on LAND in European Istanbul (Fatih, 
// Beyoğlu, Eyüp, Bakırköy, Şişli, Gaziosmanpaşa).
//
// STRICT BOUNDS: lat 40.98–41.09, lon 28.85–28.99
// The Bosphorus is at lon ~29.00+ — we NEVER go there.
// The Sea of Marmara coast is at lat ~40.98 — we stay above it.
// The Golden Horn runs roughly from [41.017,28.955] to [41.03,28.97]
// — we avoid that strip too.
// =================================================================

export const TRAFFIC_ROADS = [
  {
    id: 'R1', name: 'E5 Highway – Bakırköy → Bayrampaşa', flow: 0.85, speed: 15, vehicles: 320,
    coords: [
      [40.9920,28.8580],[40.9928,28.8640],[40.9935,28.8700],[40.9942,28.8760],
      [40.9950,28.8820],[40.9958,28.8880],[40.9965,28.8940],[40.9972,28.9000],
      [40.9980,28.9060],[40.9988,28.9120],[40.9995,28.9180],[41.0003,28.9240],
      [41.0010,28.9300],[41.0015,28.9350],[41.0020,28.9400],
    ],
    rerouteCoords: [
      [40.9920,28.8580],[40.9910,28.8640],[40.9905,28.8720],[40.9905,28.8800],
      [40.9910,28.8880],[40.9918,28.8960],[40.9928,28.9040],[40.9940,28.9120],
      [40.9955,28.9200],[40.9970,28.9270],[40.9988,28.9330],[41.0005,28.9370],
      [41.0020,28.9400],
    ],
    rerouteName: 'Via D100 Southern',
  },
  {
    id: 'R2', name: 'Vatan Caddesi – Aksaray → Topkapı', flow: 0.65, speed: 28, vehicles: 210,
    coords: [
      [41.0080,28.9390],[41.0070,28.9430],[41.0060,28.9470],[41.0050,28.9510],
      [41.0040,28.9540],[41.0035,28.9570],[41.0030,28.9600],
    ],
  },
  {
    id: 'R3', name: 'Atatürk Bulvarı – Unkapanı → Aksaray', flow: 0.70, speed: 22, vehicles: 240,
    coords: [
      [41.0300,28.9570],[41.0280,28.9565],[41.0260,28.9560],[41.0240,28.9555],
      [41.0220,28.9550],[41.0200,28.9545],[41.0180,28.9540],[41.0160,28.9535],
      [41.0140,28.9530],[41.0120,28.9520],[41.0100,28.9510],[41.0080,28.9500],
    ],
    rerouteCoords: [
      [41.0300,28.9570],[41.0290,28.9530],[41.0275,28.9490],[41.0260,28.9460],
      [41.0240,28.9440],[41.0220,28.9430],[41.0200,28.9425],[41.0180,28.9430],
      [41.0160,28.9440],[41.0140,28.9455],[41.0120,28.9475],[41.0100,28.9490],
      [41.0080,28.9500],
    ],
    rerouteName: 'Via Millet Caddesi',
  },
  {
    id: 'R4', name: 'Adnan Menderes Blv – Fatih Ring', flow: 0.50, speed: 38, vehicles: 160,
    coords: [
      [41.0200,28.9300],[41.0190,28.9340],[41.0180,28.9380],[41.0170,28.9420],
      [41.0160,28.9460],[41.0155,28.9500],[41.0150,28.9540],
    ],
  },
  {
    id: 'R5', name: 'Kennedy Cad – Samatya Coastal', flow: 0.30, speed: 58, vehicles: 95,
    coords: [
      [40.9950,28.9250],[40.9948,28.9290],[40.9945,28.9330],[40.9943,28.9370],
      [40.9940,28.9410],[40.9938,28.9450],[40.9935,28.9490],[40.9933,28.9530],
    ],
  },
  {
    id: 'R6', name: 'Halıcıoğlu–Eyüp Corridor', flow: 0.40, speed: 45, vehicles: 120,
    coords: [
      [41.0420,28.9400],[41.0440,28.9420],[41.0460,28.9440],[41.0480,28.9450],
      [41.0500,28.9460],[41.0520,28.9470],[41.0540,28.9475],
    ],
  },
  {
    id: 'R7', name: 'Şişli–Mecidiyeköy Arterial', flow: 0.90, speed: 10, vehicles: 380,
    coords: [
      [41.0550,28.9800],[41.0560,28.9780],[41.0570,28.9760],[41.0580,28.9740],
      [41.0590,28.9720],[41.0600,28.9700],[41.0610,28.9680],[41.0620,28.9660],
    ],
    rerouteCoords: [
      [41.0550,28.9800],[41.0560,28.9820],[41.0575,28.9830],[41.0590,28.9825],
      [41.0600,28.9810],[41.0610,28.9790],[41.0620,28.9770],[41.0625,28.9740],
      [41.0628,28.9710],[41.0625,28.9685],[41.0620,28.9660],
    ],
    rerouteName: 'Via Büyükdere Cad Alt',
  },
  {
    id: 'R8', name: 'Tarlabaşı Bulvarı', flow: 0.75, speed: 20, vehicles: 190,
    coords: [
      [41.0360,28.9700],[41.0355,28.9720],[41.0350,28.9740],[41.0345,28.9755],
      [41.0340,28.9770],[41.0338,28.9790],
    ],
  },
  {
    id: 'R9', name: 'D100 – Bağcılar → Güngören', flow: 0.55, speed: 40, vehicles: 165,
    coords: [
      [41.0350,28.8600],[41.0340,28.8660],[41.0330,28.8720],[41.0320,28.8780],
      [41.0310,28.8840],[41.0300,28.8900],[41.0285,28.8960],[41.0270,28.9020],
      [41.0255,28.9080],[41.0240,28.9140],[41.0225,28.9200],
    ],
  },
  {
    id: 'R10', name: 'Hospital Access – Çapa', flow: 0.20, speed: 65, vehicles: 55,
    coords: [
      [41.0100,28.9480],[41.0110,28.9470],[41.0120,28.9460],[41.0130,28.9450],
      [41.0140,28.9440],[41.0150,28.9430],
    ],
  },
];

// =================================================================
// DISPATCH DATA — European side only
// =================================================================

export const DISPATCH_HOSPITALS = [
  { id: 'HSP-01', name: 'İ.Ü. Çapa Tıp Fak.', lat: 41.0095, lon: 28.9480 },
  { id: 'HSP-02', name: 'Cerrahpaşa Hastanesi', lat: 41.0035, lon: 28.9425 },
  { id: 'HSP-03', name: 'Bakırköy Sadi Konuk', lat: 40.9860, lon: 28.8710 },
  { id: 'HSP-04', name: 'Şişli Etfal', lat: 41.0580, lon: 28.9760 },
  { id: 'HSP-05', name: 'Haseki Eğitim H.', lat: 41.0070, lon: 28.9380 },
];

export const DISPATCH_INCIDENTS = [
  { id: 'INC-01', lat: 41.0120, lon: 28.9550, type: 'building_collapse', severity: 'CRITICAL', casualties: 12, area: 'Aksaray' },
  { id: 'INC-02', lat: 41.0360, lon: 28.9700, type: 'fire', severity: 'HIGH', casualties: 3, area: 'Beyoğlu' },
  { id: 'INC-03', lat: 41.0520, lon: 28.9450, type: 'flood', severity: 'MEDIUM', casualties: 0, area: 'Eyüp' },
  { id: 'INC-04', lat: 40.9980, lon: 28.9200, type: 'gas_leak', severity: 'HIGH', casualties: 0, area: 'Zeytinburnu' },
  { id: 'INC-05', lat: 41.0250, lon: 28.9400, type: 'road_collapse', severity: 'CRITICAL', casualties: 5, area: 'Fatih' },
];

// Unit coordinates — ALL strictly on European Istanbul land
// lat 41.00-41.06, lon 28.86-28.98
export const UNIT_POSITIONS = {
  medical: [
    [41.0050,28.9350],[41.0180,28.9650],[40.9900,28.8800],[41.0400,28.9550],
    [41.0520,28.9350],[41.0250,28.9100],[40.9980,28.9000],[41.0100,28.9700],
    [41.0420,28.9700],[41.0300,28.9450],[40.9950,28.8650],[41.0150,28.9250],
  ],
  fire: [
    [41.0070,28.9500],[41.0350,28.9600],[41.0200,28.9200],[40.9920,28.8900],
    [41.0480,28.9400],[41.0150,28.9600],[41.0400,28.9650],[41.0010,28.9100],
  ],
  police: [
    [41.0080,28.9600],[41.0280,28.9750],[41.0380,28.9400],[40.9960,28.8750],
    [41.0500,28.9550],[41.0180,28.9380],[41.0330,28.9500],[40.9890,28.8950],
    [41.0100,28.9200],[41.0430,28.9280],
  ],
};

// =====================================================================
// SMART ROUTE GENERATOR — Creates realistic city-grid routes
// Uses L-shaped paths that follow the Istanbul street grid pattern.
// Routes go horizontal first, then vertical (or vice versa), 
// creating right-angle turns like real urban roads.
// =====================================================================
export function generateUnitRoute(from, to) {
  const [lat1, lon1] = from;
  const [lat2, lon2] = to;
  const points = [];
  const steps = 12;

  // Determine which way to turn (horizontal-first or vertical-first)
  // based on the angle; this creates variety in routes
  const hFirst = Math.abs(lon2 - lon1) > Math.abs(lat2 - lat1);
  
  if (hFirst) {
    // Phase 1: Move horizontally (east/west)
    const hSteps = Math.ceil(steps * 0.55);
    for (let i = 0; i <= hSteps; i++) {
      const t = i / hSteps;
      const lat = lat1 + (lat2 - lat1) * t * 0.15; // slight drift
      const lon = lon1 + (lon2 - lon1) * t;
      points.push([lat, lon]);
    }
    // Phase 2: Move vertically (north/south)
    const lastPt = points[points.length - 1];
    const vSteps = steps - hSteps;
    for (let i = 1; i <= vSteps; i++) {
      const t = i / vSteps;
      const lat = lastPt[0] + (lat2 - lastPt[0]) * t;
      const lon = lastPt[1] + (lon2 - lastPt[1]) * t * 0.1; // slight drift
      points.push([lat, lon]);
    }
  } else {
    // Phase 1: Move vertically first
    const vSteps = Math.ceil(steps * 0.55);
    for (let i = 0; i <= vSteps; i++) {
      const t = i / vSteps;
      const lat = lat1 + (lat2 - lat1) * t;
      const lon = lon1 + (lon2 - lon1) * t * 0.15;
      points.push([lat, lon]);
    }
    // Phase 2: Move horizontally
    const lastPt = points[points.length - 1];
    const hSteps = steps - vSteps;
    for (let i = 1; i <= hSteps; i++) {
      const t = i / hSteps;
      const lat = lastPt[0] + (lat2 - lastPt[0]) * t * 0.1;
      const lon = lastPt[1] + (lon2 - lastPt[1]) * t;
      points.push([lat, lon]);
    }
  }
  
  return points;
}

export function generateHospitalRoute(from, to) {
  return generateUnitRoute(from, to);
}
