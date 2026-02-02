"""
WebSocket Manager for real-time training updates.
"""

from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Supports multiple rooms for different training sessions.
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, room: str = "default"):
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            room: Room/session identifier
        """
        await websocket.accept()
        async with self.lock:
            if room not in self.active_connections:
                self.active_connections[room] = []
            self.active_connections[room].append(websocket)
        logger.info(f"Client connected to room: {room}")
    
    async def disconnect(self, websocket: WebSocket, room: str = "default"):
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to remove
            room: Room the connection belongs to
        """
        async with self.lock:
            if room in self.active_connections:
                if websocket in self.active_connections[room]:
                    self.active_connections[room].remove(websocket)
                if not self.active_connections[room]:
                    del self.active_connections[room]
        logger.info(f"Client disconnected from room: {room}")
    
    async def disconnect_all(self):
        """Disconnect all connections."""
        async with self.lock:
            for room, connections in self.active_connections.items():
                for ws in connections:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            self.active_connections.clear()
    
    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send a message to a specific connection.
        
        Args:
            websocket: Target WebSocket
            message: Message to send
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def broadcast(self, message: Dict[str, Any], room: str = "default"):
        """
        Broadcast a message to all connections in a room.
        
        Args:
            message: Message to broadcast
            room: Target room
        """
        if room not in self.active_connections:
            return
        
        dead_connections = []
        for connection in self.active_connections[room]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        
        # Clean up dead connections
        for dead in dead_connections:
            await self.disconnect(dead, room)
    
    async def broadcast_training_update(
        self,
        room: str,
        step: int,
        episode: int,
        reward: float,
        metrics: Dict[str, float],
        state: Optional[Dict] = None
    ):
        """
        Broadcast a training update.
        
        Args:
            room: Training session room
            step: Current timestep
            episode: Current episode
            reward: Current reward
            metrics: Training metrics
            state: Optional environment state for visualization
        """
        message = {
            "type": "training_update",
            "data": {
                "step": step,
                "episode": episode,
                "reward": reward,
                "metrics": metrics,
                "state": state,
            }
        }
        await self.broadcast(message, room)
    
    async def broadcast_episode_complete(
        self,
        room: str,
        episode: int,
        total_reward: float,
        length: int,
        metrics: Dict[str, float]
    ):
        """
        Broadcast episode completion.
        
        Args:
            room: Training session room
            episode: Completed episode number
            total_reward: Episode total reward
            length: Episode length
            metrics: Episode metrics
        """
        message = {
            "type": "episode_complete",
            "data": {
                "episode": episode,
                "total_reward": total_reward,
                "length": length,
                "metrics": metrics,
            }
        }
        await self.broadcast(message, room)
    
    async def broadcast_training_complete(
        self,
        room: str,
        total_steps: int,
        total_episodes: int,
        final_metrics: Dict[str, float]
    ):
        """
        Broadcast training completion.
        
        Args:
            room: Training session room
            total_steps: Total steps trained
            total_episodes: Total episodes
            final_metrics: Final training metrics
        """
        message = {
            "type": "training_complete",
            "data": {
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "metrics": final_metrics,
            }
        }
        await self.broadcast(message, room)


# Global connection manager instance
manager = ConnectionManager()

