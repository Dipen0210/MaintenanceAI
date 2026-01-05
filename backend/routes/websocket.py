"""
WebSocket Routes for Real-time Updates.
Provides live streaming of machine status updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import asyncio
import json

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Remove dead connections
                self.disconnect(connection)
    
    async def send_personal(self, websocket: WebSocket, message: Dict):
        """Send message to a specific client."""
        await websocket.send_json(message)


manager = ConnectionManager()


@router.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time machine status updates.
    
    Clients connect here to receive live updates about:
    - Machine health changes
    - New anomaly detections
    - RUL predictions
    - Maintenance alerts
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client (e.g., subscription requests)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get('type')
                
                if message_type == 'subscribe':
                    # Client subscribing to specific machine updates
                    machine_id = message.get('machine_id')
                    await manager.send_personal(websocket, {
                        'type': 'subscribed',
                        'machine_id': machine_id,
                        'message': f'Subscribed to updates for {machine_id}'
                    })
                
                elif message_type == 'ping':
                    await manager.send_personal(websocket, {'type': 'pong'})
                
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {
                    'type': 'error',
                    'message': 'Invalid JSON'
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_update(update_type: str, data: Dict):
    """
    Broadcast an update to all connected clients.
    
    Args:
        update_type: Type of update (e.g., 'anomaly', 'fault', 'rul', 'alert')
        data: Update data to send
    """
    message = {
        'type': update_type,
        'data': data
    }
    await manager.broadcast(message)
