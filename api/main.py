"""
UrbanCityOS — FastAPI Application Entry Point
==============================================
Lifespan boot, CORS, router mounting, sys.path fix.
Run with:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# ── Fix sys.path so `from src.*` imports work ────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.state import app_state

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-35s %(levelname)-6s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urbanfl.api")


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Boot all backend modules on startup, tear down on shutdown."""
    logger.info("FastAPI lifespan — STARTUP")
    app_state.boot()
    await app_state.start_simulation()
    yield
    logger.info("FastAPI lifespan — SHUTDOWN")
    await app_state.stop_simulation()


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="UrbanCityOS API",
    description="Full-stack API for the UrbanFL emergency management platform",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:5174",   # Vite dev fallback
        "http://localhost:4173",   # Vite preview
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:4173",
        "https://revanth574.github.io",  # GitHub Pages
        os.environ.get("RENDER_EXTERNAL_URL", ""),  # Render self-reference
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
import traceback
from fastapi import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Global error on {request.url.path}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "traceback": tb, "error": str(exc)},
    )



# ── Static Files ─────────────────────────────────────────────────────
XBD_DIR = os.path.join(PROJECT_ROOT, "data", "real", "xbd", "geotiffs")
if os.path.exists(XBD_DIR):
    app.mount("/api/v1/vision/files", StaticFiles(directory=XBD_DIR), name="vision_files")

# ── Import and mount routers ────────────────────────────────────────
from api.routers import dashboard, silos, dispatch, zones, analytics, command, ws, simulation, system

app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])
app.include_router(silos.router,     prefix="/api/v1", tags=["Silos"])
app.include_router(dispatch.router,  prefix="/api/v1", tags=["Dispatch"])
app.include_router(zones.router,     prefix="/api/v1", tags=["Zones"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(command.router,   prefix="/api/v1", tags=["Command"])
app.include_router(ws.router,        prefix="/api/v1", tags=["WebSocket"])
app.include_router(simulation.router, prefix="/api/v1", tags=["Simulation"])
app.include_router(system.router,    prefix="/api/v1", tags=["System"])


# ── Health check ─────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["System"])
async def health():
    """Health check endpoint — also used by frontend to detect LIVE vs DEMO mode."""
    return {
        "status": "ok",
        "uptime_seconds": round(app_state.get_uptime_seconds(), 1),
        "simulation_speed": app_state.speed,
        "ws_clients": len(app_state._ws_clients),
        "snapshots_in_history": len(app_state._snapshot_history),
    }
