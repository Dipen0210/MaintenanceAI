"""
WebSocket Routes for Real-time Updates.
Provides live streaming of machine status updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Set
import asyncio
import json
from datetime import datetime

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}  # machine_id subscriptions
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        print(f"📡 WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        print(f"📡 WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn)
    
    async def broadcast_machine_update(self, machine_data: Dict):
        """Broadcast a machine status update."""
        message = {
            'type': 'machine_update',
            'timestamp': datetime.now().isoformat(),
            'data': machine_data
        }
        await self.broadcast(message)
    
    async def broadcast_plant_summary(self, summary: Dict):
        """Broadcast plant summary update."""
        message = {
            'type': 'plant_summary',
            'timestamp': datetime.now().isoformat(),
            'data': summary
        }
        await self.broadcast(message)
    
    async def broadcast_alert(self, machine_id: str, alert_type: str, details: str):
        """Broadcast a critical alert."""
        message = {
            'type': 'alert',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'machine_id': machine_id,
                'alert_type': alert_type,  # 'critical', 'warning', 'fault'
                'details': details
            }
        }
        await self.broadcast(message)
    
    async def send_personal(self, websocket: WebSocket, message: Dict):
        """Send message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)
    
    def get_connection_count(self) -> int:
        return len(self.active_connections)


# Global manager instance
manager = ConnectionManager()


@router.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time machine status updates.
    
    Clients receive:
    - machine_update: Individual machine status changes
    - plant_summary: Overall plant health updates
    - alert: Critical alerts for immediate attention
    
    Clients can send:
    - {"type": "ping"} -> Receives {"type": "pong"}
    - {"type": "subscribe", "machine_id": "XXX"} -> Subscribe to specific machine
    """
    await manager.connect(websocket)
    
    # Send welcome message
    await manager.send_personal(websocket, {
        'type': 'connected',
        'message': 'Connected to Predictive Maintenance WebSocket',
        'timestamp': datetime.now().isoformat()
    })
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get('type')
                
                if message_type == 'subscribe':
                    machine_id = message.get('machine_id')
                    if machine_id:
                        manager.subscriptions[websocket].add(machine_id)
                        await manager.send_personal(websocket, {
                            'type': 'subscribed',
                            'machine_id': machine_id
                        })
                
                elif message_type == 'unsubscribe':
                    machine_id = message.get('machine_id')
                    if machine_id and machine_id in manager.subscriptions.get(websocket, set()):
                        manager.subscriptions[websocket].discard(machine_id)
                        await manager.send_personal(websocket, {
                            'type': 'unsubscribed',
                            'machine_id': machine_id
                        })
                
                elif message_type == 'ping':
                    await manager.send_personal(websocket, {
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    })
                
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {
                    'type': 'error',
                    'message': 'Invalid JSON'
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Export functions for use by other modules
async def broadcast_machine_update(machine_data: Dict):
    """Broadcast machine update to all connected clients."""
    await manager.broadcast_machine_update(machine_data)


async def broadcast_plant_summary(summary: Dict):
    """Broadcast plant summary to all clients."""
    await manager.broadcast_plant_summary(summary)


async def broadcast_alert(machine_id: str, alert_type: str, details: str):
    """Broadcast alert to all clients."""
    await manager.broadcast_alert(machine_id, alert_type, details)


def get_ws_connection_count() -> int:
    """Get number of active WebSocket connections."""
    return manager.get_connection_count()
