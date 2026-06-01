"""WebSocket router — real-time live feed."""

import json
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.state import app_state

router = APIRouter()
logger = logging.getLogger("urbanfl.api.ws")


@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Real-time situation feed via WebSocket.
    On connect: sends last 10 snapshots for immediate chart backfill.
    Then: receives pushed tick payloads from the simulation loop.

    Each tick payload contains:
      - snapshot: SituationSnapshot dict
      - system_status: DEFCON, FL round, epsilon, etc.
      - recent_reports: last 3 new NLP reports
      - dispatch_summary: unit counts, incident count
    """
    await ws.accept()
    app_state.register_ws(ws)
    logger.info(f"WebSocket client connected (total: {len(app_state._ws_clients)})")

    try:
        # Send backfill (last 10 snapshots) + current system status
        history = app_state.get_snapshot_history(10)
        await ws.send_text(json.dumps({
            "type": "backfill",
            "data": history,
            "system_status": app_state.get_system_status(),
        }))

        # Keep the connection alive — the simulation loop pushes tick payloads
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                msg = json.loads(data)

                # Handle client commands
                if msg.get("type") == "set_speed":
                    app_state.speed = float(msg.get("speed", 1))
                    await ws.send_text(json.dumps({
                        "type": "speed_changed",
                        "speed": app_state.speed,
                    }))

            except asyncio.TimeoutError:
                # Send keepalive ping with system status
                await ws.send_text(json.dumps({
                    "type": "ping",
                    "system_status": app_state.get_system_status(),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state.unregister_ws(ws)
        logger.info(f"WebSocket client disconnected (remaining: {len(app_state._ws_clients)})")
