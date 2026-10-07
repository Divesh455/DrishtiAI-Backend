"""
ws_screening.py
===============
WebSocket route for real-time screening progress events.

Frontend connects to:
    ws://127.0.0.1:8000/ws/screening/{screening_id}

and receives JSON progress events like:
    {
        "event": "screening_progress",
        "screening_id": "DR-20261006-1234",
        "step": 2,
        "total_steps": 5,
        "percent": 40,
        "message": "Running DR Classification & Senanur Safety Check...",
        "status": "in_progress"
    }
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.ws_manager import ws_manager

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/screening/{screening_id}")
async def screening_progress_ws(
    screening_id: str,
    websocket: WebSocket,
):
    """
    WebSocket endpoint for real-time screening pipeline progress.

    Frontend connects before or right after submitting
    POST /api/v1/screening/screen and listens for step-by-step
    progress events while the AI pipeline is running.

    Progress Events (JSON):
        step         : Current step number (1-5)
        total_steps  : Total steps (always 5)
        percent      : Completion percentage (0-100)
        message      : Human-readable status message
        status       : "in_progress" | "completed" | "failed"
    """
    await ws_manager.connect(screening_id, websocket)
    try:
        while True:
            # Drain any queued progress events and send to frontend
            await ws_manager.drain_and_broadcast(screening_id)

            # Also listen for any ping/pong or disconnect from the client
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.2,
                )
            except asyncio.TimeoutError:
                pass  # No client message; keep looping and draining

    except WebSocketDisconnect:
        ws_manager.disconnect(screening_id, websocket)
