"""
Training API Routes

Endpoints for starting, monitoring, and controlling training sessions.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel, Field
import uuid
import asyncio
import logging

from ..websocket import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Active training sessions
training_sessions: Dict[str, Dict[str, Any]] = {}


class TrainingConfig(BaseModel):
    """Training configuration model."""
    
    algorithm: str = Field(default="PPO", description="RL algorithm to use")
    environment: str = Field(default="navigation", description="Environment name")
    environment_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Hyperparameters
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    batch_size: int = Field(default=64, ge=1)
    n_steps: int = Field(default=2048, ge=1)
    n_epochs: int = Field(default=10, ge=1)
    
    # Training settings
    total_timesteps: int = Field(default=100000, ge=1)
    eval_frequency: int = Field(default=1000, ge=1)
    save_frequency: int = Field(default=10000, ge=1)
    
    # Goal (optional natural language)
    goal_text: Optional[str] = None
    
    # Visualization
    render_frequency: int = Field(default=100, ge=1)


class TrainingSession(BaseModel):
    """Training session response model."""
    
    session_id: str
    status: str
    algorithm: str
    environment: str
    current_step: int = 0
    current_episode: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)


@router.post("/start", response_model=TrainingSession)
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """
    Start a new training session.
    
    Creates a new training session with the specified configuration
    and begins training in the background.
    """
    session_id = str(uuid.uuid4())[:8]
    
    # Create session record
    session = {
        "id": session_id,
        "status": "initializing",
        "config": config.dict(),
        "current_step": 0,
        "current_episode": 0,
        "metrics": {},
        "history": {
            "rewards": [],
            "lengths": [],
            "losses": [],
        }
    }
    training_sessions[session_id] = session
    
    # Start training in background
    background_tasks.add_task(run_training, session_id, config)
    
    logger.info(f"Started training session: {session_id}")
    
    return TrainingSession(
        session_id=session_id,
        status="initializing",
        algorithm=config.algorithm,
        environment=config.environment,
    )


async def run_training(session_id: str, config: TrainingConfig):
    """
    Run training in background.
    
    This function simulates the training loop and sends updates via WebSocket.
    In production, this would use the actual RL algorithms.
    """
    import numpy as np
    
    session = training_sessions.get(session_id)
    if not session:
        return
    
    try:
        session["status"] = "running"
        
        # Simulate training loop
        total_steps = config.total_timesteps
        episode = 0
        episode_reward = 0
        episode_length = 0
        
        for step in range(total_steps):
            session["current_step"] = step
            
            # Simulate step reward (with gradual improvement)
            progress = step / total_steps
            base_reward = -0.1 + progress * 0.3
            reward = base_reward + np.random.normal(0, 0.1)
            
            episode_reward += reward
            episode_length += 1
            
            # Simulate episode end
            if np.random.random() < 0.01 or episode_length > 500:
                episode += 1
                session["current_episode"] = episode
                
                # Store history
                session["history"]["rewards"].append(episode_reward)
                session["history"]["lengths"].append(episode_length)
                
                # Calculate metrics
                recent_rewards = session["history"]["rewards"][-100:]
                metrics = {
                    "mean_reward": float(np.mean(recent_rewards)),
                    "max_reward": float(np.max(recent_rewards)),
                    "episode_length": float(np.mean(session["history"]["lengths"][-100:])),
                    "loss": max(0.01, 1.0 - progress),
                }
                session["metrics"] = metrics
                
                # Broadcast episode completion
                await manager.broadcast_episode_complete(
                    session_id,
                    episode,
                    episode_reward,
                    episode_length,
                    metrics
                )
                
                episode_reward = 0
                episode_length = 0
            
            # Send periodic updates
            if step % config.render_frequency == 0:
                # Simulate environment state for visualization
                state = {
                    "position": [
                        100 + 600 * progress + np.random.normal(0, 10),
                        300 + np.random.normal(0, 50)
                    ],
                    "velocity": [np.random.normal(50, 10), np.random.normal(0, 20)],
                }
                
                await manager.broadcast_training_update(
                    session_id,
                    step,
                    episode,
                    reward,
                    session["metrics"],
                    state
                )
            
            # Small delay to prevent CPU overuse
            if step % 100 == 0:
                await asyncio.sleep(0.01)
        
        # Training complete
        session["status"] = "completed"
        await manager.broadcast_training_complete(
            session_id,
            total_steps,
            episode,
            session["metrics"]
        )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        session["status"] = "error"
        session["error"] = str(e)


@router.get("/{session_id}", response_model=TrainingSession)
async def get_training_status(session_id: str):
    """Get the status of a training session."""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = training_sessions[session_id]
    return TrainingSession(
        session_id=session_id,
        status=session["status"],
        algorithm=session["config"]["algorithm"],
        environment=session["config"]["environment"],
        current_step=session["current_step"],
        current_episode=session["current_episode"],
        metrics=session["metrics"],
    )


@router.post("/{session_id}/pause")
async def pause_training(session_id: str):
    """Pause a running training session."""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = training_sessions[session_id]
    if session["status"] == "running":
        session["status"] = "paused"
    
    return {"status": session["status"]}


@router.post("/{session_id}/resume")
async def resume_training(session_id: str):
    """Resume a paused training session."""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = training_sessions[session_id]
    if session["status"] == "paused":
        session["status"] = "running"
    
    return {"status": session["status"]}


@router.post("/{session_id}/stop")
async def stop_training(session_id: str):
    """Stop a training session."""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = training_sessions[session_id]
    session["status"] = "stopped"
    
    return {"status": "stopped"}


@router.get("/{session_id}/history")
async def get_training_history(session_id: str):
    """Get training history for a session."""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = training_sessions[session_id]
    return session["history"]


@router.get("/")
async def list_sessions():
    """List all training sessions."""
    return [
        {
            "session_id": sid,
            "status": session["status"],
            "algorithm": session["config"]["algorithm"],
            "environment": session["config"]["environment"],
            "current_step": session["current_step"],
        }
        for sid, session in training_sessions.items()
    ]


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time training updates.
    
    Connect to receive live updates during training.
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Handle commands from client
            if data == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket, session_id)

