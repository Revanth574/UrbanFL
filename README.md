# 🏙️ UrbanFL — Federated Learning for Urban Emergency Management

[![Deploy](https://github.com/Revanth574/UrbanFL/actions/workflows/deploy.yml/badge.svg)](https://github.com/Revanth574/UrbanFL/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-GitHub_Pages-blue)](https://revanth574.github.io/UrbanFL/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Privacy-preserving federated learning pipeline for real-time urban situational awareness, inter-agency coordination, and automated emergency response dispatch.**

🔗 **Live Dashboard:** [https://revanth574.github.io/UrbanFL/](https://revanth574.github.io/UrbanFL/)

---

## 🚀 Overview

**UrbanFL (UrbanCityOS)** is a full-scale monitoring system that handles active urban crises by integrating heterogeneous, multi-modal real-world datasets across multiple data silos using **federated learning**. It features a command center dashboard with real-time 3D geospatial visualization, AI-driven risk assessment, and automated dispatch coordination.

A core innovation is **Privacy-Preserving Geospatial Analysis** — securely processing sensitive city data within a federated framework without centralizing raw observations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   UrbanCityOS                        │
├──────────┬──────────┬──────────┬────────────────────┤
│  Silo 1  │  Silo 2  │  Silo 3  │   Command Center   │
│  NLP     │  Vision  │  Infra   │   (Orchestrator)   │
│  Crisis  │  Damage  │  Routing │                    │
│  Text    │  Sat.Img │  Network │   FL Aggregation   │
└────┬─────┴────┬─────┴────┬─────┴─────────┬──────────┘
     │          │          │               │
     └──────────┴──────────┘               │
              Federated Learning           │
              (FedHetero + DualAgg)        │
                                           ▼
                              ┌─────────────────────┐
                              │  React Dashboard    │
                              │  Real-Time WebSocket│
                              │  3D Map + Analytics │
                              └─────────────────────┘
```

### Data Silos

| Silo | Domain | Model | Data Sources |
|------|--------|-------|-------------|
| **Silo 1** | NLP — Crisis Classification | `CrisisClassifier` (XLM-RoBERTa) | CrisisNLP, HumAID, public reports |
| **Silo 2** | Vision — Damage Detection | `SiameseChangeDetector` | xBD Satellite Imagery |
| **Silo 3** | Infrastructure — Geospatial Routing | BPR Traffic Model + OSRM | Istanbul road networks, OSM |

### FL Innovations
- **Heterogeneous Federated Learning** (`FedHetero`) — dynamic client disparity handling
- **Dual Aggregation** (`DualAggregator`) — advanced aggregation beyond FedAvg
- **Differential Privacy** — Gaussian mechanism with tracked privacy budget (ε)
- **Concept Drift Detection** — adapts to evolving distributions during crises

---

## 🖥️ Tech Stack

### Backend
- **Python 3.10+** — FastAPI, PyTorch, NetworkX
- **Real-Time Streaming** — WebSocket-based tick system
- **Geospatial** — OSRM API routing, Istanbul land-constrained simulation

### Frontend
- **React 19** + **Vite 8** — High-performance SPA
- **MapLibre GL** + **Leaflet** — 3D geospatial visualization
- **Recharts** — FL convergence & zone analytics
- **Framer Motion** — Premium micro-animations
- **TailwindCSS** — Utility-first styling

---

## 📂 Project Structure

```
UrbanFL/
├── api/                    # FastAPI backend server
│   ├── routers/            # REST + WebSocket endpoints
│   └── state.py            # Shared simulation state
├── src/                    # Core simulation & FL engine
│   ├── federated/          # FedHetero, DualAggregator, Privacy
│   ├── silo1/              # NLP crisis classifier
│   ├── silo2/              # Vision damage detector
│   ├── silo3/              # Infrastructure routing
│   ├── coordination/       # Command center logic
│   ├── dashboard/          # Situation monitor
│   └── orchestrator.py     # Agent-based execution graph
├── frontend/               # React dashboard
│   └── src/
│       ├── components/     # Sidebar, Topbar, panels
│       ├── pages/          # Dashboard, Dispatch, Vision, NLP, etc.
│       ├── hooks/          # Real-time WebSocket hooks
│       └── utils/          # API client, OSRM routing
├── config/                 # YAML simulation configs
├── data/                   # Geospatial & crisis datasets
├── paper/                  # Research paper drafts
└── experiments/            # Testing & benchmarks
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Clone & Setup Backend
```bash
git clone https://github.com/Revanth574/UrbanFL.git
cd UrbanFL
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Setup Frontend
```bash
cd frontend
npm install
```

### 3. Run Everything
```bash
# Option A: Use the start script (Windows)
start.bat

# Option B: Manual
# Terminal 1 — Backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

### 4. Open Dashboard
Navigate to `http://localhost:5173` — the frontend proxies API requests to the backend automatically.

---

## 🌐 Deployment

The frontend is automatically deployed to **GitHub Pages** on every push to `main` via GitHub Actions.

**Live URL:** [https://revanth574.github.io/UrbanFL/](https://revanth574.github.io/UrbanFL/)

> **Note:** The GitHub Pages deployment serves the static frontend only. For full functionality (real-time data, WebSocket streaming), run the backend locally.

---

## 📊 Datasets

| Dataset | Type | Size | Included |
|---------|------|------|----------|
| CrisisNLP v2 | Crisis text classification | ~50MB | ✅ |
| Istanbul Geospatial | Roads, hospitals, shelters, traffic | ~12MB | ✅ |
| xBD Sample Images | Pre/post disaster satellite | ~2MB | ✅ (samples) |
| xBD Full Dataset | Satellite imagery corpus | ~91GB | ❌ (download separately) |
| Istanbul Roads Network | `.gpickle` graph | ~56MB | ❌ (download separately) |
| Palu Network | `.graphml` graph | ~18MB | ❌ (download separately) |

Large files excluded from Git are listed in `.gitignore`. Download them separately for full simulation.

---

## 🔬 Research

This project explores novel approaches in:
- **Privacy-preserving geospatial analysis** in federated settings
- **Heterogeneous FL** across multi-modal data silos
- **Real-time emergency dispatch optimization** under disaster constraints
- **Concept drift adaptation** in crisis-evolving data distributions

See the `paper/` directory for research documentation.

---

## 📄 License

This project is licensed under the MIT License.
