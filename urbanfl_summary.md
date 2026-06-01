# UrbanFL: Privacy-Preserving City Monitoring & Emergency Management (UrbanCityOS)

## Project Overview
**UrbanFL** is an advanced, privacy-preserving federated learning (FL) pipeline designed for urban situational awareness, inter-agency coordination, and automated emergency response dispatch. It has evolved into **UrbanCityOS**, a full-scale monitoring system handling active crises by integrating heterogeneous, multi-modal real-world datasets across multiple "silos".

A core innovation is **Privacy-Preserving Geospatial Analysis**, setting a new standard for securely processing sensitive city data within a federated framework without centralizing raw observations.

## Architectural Silos (Multi-Modal Inputs)

The platform is split into three main operational data silos. The orchestrator integrates their localized knowledge using a federated learning approach:

### 1. Silo 1: NLP - Crisis Classification
*   **Data Sources:** CrisisNLP, public crisis reports, and HumAID data.
*   **Model:** `CrisisClassifier` based on **XLM-RoBERTa**.
*   **Innovations & Optimizations:** 
    *   Optimized for low-VRAM (4GB RTX 3050) using Automatic Mixed Precision (AMP), gradient accumulation, and a frozen base encoder (training only the classification heads).
    *   Multi-task output: Predicts both the **incident class** (Cross Entropy) and the **severity** (MSE).
    
### 2. Silo 2: Vision - Damage & Change Detection
*   **Data Sources:** xBD Satellite Imagery (Pre- and Post-disaster).
*   **Model:** `SiameseChangeDetector`.
*   **Innovations & Optimizations:**
    *   Generates pixel-wise binary change masks alongside disaster type classification and severity estimation.
    *   Uses a combined loss function of `BCEWithLogitsLoss` + `Dice Loss` to maximize Intersection-over-Union (IoU).

### 3. Silo 3: Infrastructure & Geospatial Routing
*   **Data Sources:** Road networks, seismic/flood risk zones, OpenStreetMap.
*   **Scope Constraint:** Strictly constrained to **European Istanbul**.
*   **Innovations & Optimizations:**
    *   Integrates **OSRM API** routing to replace synthetic path generation, allowing for realistic, high-fidelity, grid-following route geometries (eliminating straight-line/water-crossing artifacts).
    *   Calibrates Bureau of Public Roads (BPR) parameters across distinct road types for accurate traffic and dispatch simulation under disaster conditions.

## Federated Learning & Orchestration innovations
*   **Orchestration Graph (`src/orchestrator.py`):** An agent-based execution graph connects the data silos with command center modules for automated resource dispatch and situational awareness.
*   **Privacy Guarantees (`src/federated/privacy_analysis.py`, `guardrails.py`):** Aggregates real gradients using a Differential Privacy (DP) Gaussian mechanism, strictly tracking the privacy budget (Epsilon $\epsilon$).
*   **Heterogeneous Setup (`src/federated/fedhetero.py`):** Handles client disparity dynamically.
*   **Dual Aggregation (`src/federated/dual_aggregator.py`):** Explores advanced aggregation geometries beyond standard FedAvg.
*   **Concept Drift (`src/federated/concept_drift.py`):** Adapts to rapidly changing incoming distributions over time as a crisis evolves in real-world scenarios.

## Command Center Frontend (Dashboard)
The operational view is powered by a modern, high-performance web interface allowing commanders to visualize the output of all FL models simultaneously:
*   **Core Stack:** React (v19) via Vite, JavaScript (ES Modules).
*   **Styling & UI:** TailwindCSS, `clsx`, `tailwind-merge`.
*   **Geospatial Visualization:** `Leaflet` & `react-leaflet` to visualize incident markers, European Istanbul land-based routing, and emergency dispatch unit movements.
*   **Analytics & Motion:** `Recharts` for charting FL convergence/zone metrics, and `Framer Motion` for dynamic, premium micro-animations/transitions.

## Recent Focus & Adjustments
*   Migrated from synthetic/fallback data to using **100% real, downloaded, and cached datasets** across all 12 dataset streams via `MultiHazardDataAdapter`.
*   Evaluated **FastAPI** to replace underlying backend web frameworks for high-performance asynchronous API requests from the frontend and between FL clients.
*   Ensured rigid land-based routing rules preventing dispatch units from traveling over water bodies (e.g., straight lines across the Bosphorus) by integrating OSRM and precise OpenStreetMap land polygon data.
