"""Training routes for RL-GYM API."""
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import asyncio

from api.websocket import WebSocketManager
from training import TrainingManager, TrainingSession
from config import Config

router = APIRouter(prefix="/training", tags=["training"])

# Global training manager
training_manager = TrainingManager()

# WebSocket manager for live updates
ws_manager = WebSocketManager()


class TrainingConfig(BaseModel):
    """Training configuration request model."""
    environment: str = "gridworld"
    algorithm: str = "dqn"
    env_config: Optional[Dict[str, Any]] = None
    agent_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    natural_language_goal: Optional[str] = None
    dataset_id: Optional[str] = None


class TrainingResponse(BaseModel):
    """Training response model."""
    session_id: str
    status: str
    message: str


@router.post("/start", response_model=TrainingResponse)
async def start_training(config: TrainingConfig):
    """Start a new training session."""
    try:
        # Get environment config
        env_config = config.env_config or {}
        env_config["env_type"] = config.environment
        
        # Get agent config
        agent_config = config.agent_config or {}
        agent_config["algorithm"] = config.algorithm
        
        # Get training config
        training_config = config.training_config or {}
        
        # Add natural language goal if provided
        if config.natural_language_goal:
            env_config["natural_language_goal"] = config.natural_language_goal
        
        # Add dataset if provided
        if config.dataset_id:
            training_config["dataset_id"] = config.dataset_id
        
        # Create session through manager
        session_id = training_manager.create_session(
            env_config=env_config,
            agent_config=agent_config,
            training_config=training_config
        )
        
        # Start training asynchronously
        asyncio.create_task(_run_training(session_id))
        
        return TrainingResponse(
            session_id=session_id,
            status="started",
            message=f"Training session {session_id} started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_training(session_id: str):
    """Run training in background and broadcast updates."""
    try:
        async for update in training_manager.run_session(session_id):
            # Broadcast update to all connected WebSocket clients
            await ws_manager.broadcast({
                "type": "training_update",
                "session_id": session_id,
                **update
            })
    except Exception as e:
        await ws_manager.broadcast({
            "type": "training_error",
            "session_id": session_id,
            "error": str(e)
        })


@router.post("/stop/{session_id}")
async def stop_training(session_id: str):
    """Stop a training session."""
    try:
        training_manager.stop_session(session_id)
        return {"status": "stopped", "session_id": session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/pause/{session_id}")
async def pause_training(session_id: str):
    """Pause a training session."""
    try:
        training_manager.pause_session(session_id)
        return {"status": "paused", "session_id": session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/resume/{session_id}")
async def resume_training(session_id: str):
    """Resume a paused training session."""
    try:
        training_manager.resume_session(session_id)
        return {"status": "resumed", "session_id": session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/status/{session_id}")
async def get_training_status(session_id: str):
    """Get the status of a training session."""
    try:
        session = training_manager.get_session(session_id)
        return {
            "session_id": session_id,
            "status": session.status.value,
            "current_step": session.current_step,
            "current_episode": session.current_episode,
            "metrics": session.metrics_history[-10:] if session.metrics_history else []
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions")
async def list_sessions():
    """List all training sessions."""
    sessions = []
    for session_id in training_manager.sessions:
        session = training_manager.get_session(session_id)
        sessions.append({
            "session_id": session_id,
            "status": session.status.value,
            "current_step": session.current_step,
            "current_episode": session.current_episode
        })
    return {"sessions": sessions}


@router.get("/metrics/{session_id}")
async def get_metrics(session_id: str, limit: int = 100):
    """Get training metrics for a session."""
    try:
        session = training_manager.get_session(session_id)
        metrics = session.metrics_history[-limit:] if session.metrics_history else []
        return {
            "session_id": session_id,
            "metrics": metrics,
            "total_steps": session.current_step,
            "total_episodes": session.current_episode
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a training session."""
    try:
        training_manager.cleanup_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.websocket("/ws")
async def training_websocket(websocket: WebSocket):
    """WebSocket endpoint for live training updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_json()
            
            # Handle client messages
            if data.get("type") == "subscribe":
                session_id = data.get("session_id")
                if session_id:
                    await websocket.send_json({
                        "type": "subscribed",
                        "session_id": session_id
                    })
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@router.get("/algorithms")
async def list_algorithms():
    """List available RL algorithms and their configurations."""
    return {
        "algorithms": [
            {
                "id": "dqn",
                "name": "Deep Q-Network (DQN)",
                "description": "Value-based algorithm with experience replay",
                "parameters": {
                    "learning_rate": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.1},
                    "gamma": {"type": "float", "default": 0.99, "min": 0.9, "max": 0.999},
                    "epsilon_start": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
                    "epsilon_end": {"type": "float", "default": 0.01, "min": 0.01, "max": 0.5},
                    "epsilon_decay": {"type": "float", "default": 0.995, "min": 0.9, "max": 0.999},
                    "buffer_size": {"type": "int", "default": 10000, "min": 1000, "max": 1000000},
                    "batch_size": {"type": "int", "default": 64, "min": 16, "max": 512},
                    "target_update_freq": {"type": "int", "default": 100, "min": 10, "max": 1000}
                }
            },
            {
                "id": "ppo",
                "name": "Proximal Policy Optimization (PPO)",
                "description": "Policy gradient with clipped surrogate objective",
                "parameters": {
                    "learning_rate": {"type": "float", "default": 0.0003, "min": 0.0001, "max": 0.01},
                    "gamma": {"type": "float", "default": 0.99, "min": 0.9, "max": 0.999},
                    "clip_epsilon": {"type": "float", "default": 0.2, "min": 0.1, "max": 0.4},
                    "value_coef": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0},
                    "entropy_coef": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1},
                    "n_steps": {"type": "int", "default": 2048, "min": 128, "max": 4096},
                    "n_epochs": {"type": "int", "default": 10, "min": 1, "max": 20},
                    "batch_size": {"type": "int", "default": 64, "min": 16, "max": 512}
                }
            },
            {
                "id": "sac",
                "name": "Soft Actor-Critic (SAC)",
                "description": "Off-policy actor-critic with entropy regularization",
                "parameters": {
                    "learning_rate": {"type": "float", "default": 0.0003, "min": 0.0001, "max": 0.01},
                    "gamma": {"type": "float", "default": 0.99, "min": 0.9, "max": 0.999},
                    "tau": {"type": "float", "default": 0.005, "min": 0.001, "max": 0.1},
                    "alpha": {"type": "float", "default": 0.2, "min": 0.01, "max": 1.0},
                    "auto_alpha": {"type": "bool", "default": True},
                    "buffer_size": {"type": "int", "default": 100000, "min": 10000, "max": 1000000},
                    "batch_size": {"type": "int", "default": 256, "min": 64, "max": 1024}
                }
            },
            {
                "id": "a2c",
                "name": "Advantage Actor-Critic (A2C)",
                "description": "Synchronous actor-critic algorithm",
                "parameters": {
                    "learning_rate": {"type": "float", "default": 0.0007, "min": 0.0001, "max": 0.01},
                    "gamma": {"type": "float", "default": 0.99, "min": 0.9, "max": 0.999},
                    "value_coef": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0},
                    "entropy_coef": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1},
                    "n_steps": {"type": "int", "default": 5, "min": 1, "max": 20},
                    "max_grad_norm": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0}
                }
            }
        ]
    }
