"""
ws_manager.py
=============
WebSocket connection manager and progress event emitter for DrishtiAI.

Usage
-----
Import `ws_manager` singleton anywhere in the backend and call:

    ws_manager.emit(screening_id, step, percent, message, status)

The message is delivered to all WebSocket clients listening on
    /ws/screening/{screening_id}
"""

import asyncio
import json
from typing import Dict, List

from fastapi import WebSocket


class ScreeningProgressManager:
    """
    Manages active WebSocket connections per screening_id and emits
    real-time progress events to the frontend during AI pipeline execution.
    """

    def __init__(self):
        # screening_id -> list of active WebSocket connections
        self._connections: Dict[str, List[WebSocket]] = {}

        # screening_id -> asyncio event loop queue for thread-safe emit from sync code
        self._queues: Dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def connect(self, screening_id: str, websocket: WebSocket):
        await websocket.accept()
        if screening_id not in self._connections:
            self._connections[screening_id] = []
        self._connections[screening_id].append(websocket)

        # Create an event queue for this screening_id if not present
        if screening_id not in self._queues:
            self._queues[screening_id] = asyncio.Queue()

        print(f"[WS] Client connected to screening {screening_id}")

    def disconnect(self, screening_id: str, websocket: WebSocket):
        conns = self._connections.get(screening_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self._connections.pop(screening_id, None)
            self._queues.pop(screening_id, None)
        print(f"[WS] Client disconnected from screening {screening_id}")

    # ------------------------------------------------------------------
    # Emit from async context (inside async route/task)
    # ------------------------------------------------------------------

    async def broadcast(self, screening_id: str, payload: dict):
        """Send progress payload to all connected clients for this screening."""
        message = json.dumps(payload)
        disconnected = []
        for ws in self._connections.get(screening_id, []):
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(screening_id, ws)

    # ------------------------------------------------------------------
    # Emit from sync context (inside sync screening_service.py)
    # ------------------------------------------------------------------

    def emit(
        self,
        screening_id: str,
        step: int,
        total_steps: int,
        percent: int,
        message: str,
        status: str = "in_progress",  # in_progress | completed | failed
    ):
        """
        Thread-safe progress emit from synchronous code.
        Puts a progress event onto the screening's asyncio queue.
        """
        if not screening_id:
            return

        payload = {
            "event": "screening_progress",
            "screening_id": screening_id,
            "step": step,
            "total_steps": total_steps,
            "percent": percent,
            "message": message,
            "status": status,
        }

        queue = self._queues.get(screening_id)
        if queue:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Drain queue and broadcast (called by the WS route's reader loop)
    # ------------------------------------------------------------------

    async def drain_and_broadcast(self, screening_id: str):
        """
        Drains all pending progress events from the queue and broadcasts them.
        Called periodically inside the WebSocket route receive loop.
        """
        queue = self._queues.get(screening_id)
        if not queue:
            return
        while not queue.empty():
            try:
                payload = queue.get_nowait()
                await self.broadcast(screening_id, payload)
            except asyncio.QueueEmpty:
                break


# Global singleton instance used across all modules
ws_manager = ScreeningProgressManager()
