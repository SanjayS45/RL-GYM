"""
Training Manager
Manages multiple training sessions.
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from .session import TrainingSession, SessionConfig


class TrainingManager:
    """
    Manages multiple concurrent training sessions.
    
    Features:
    - Session creation and management
    - Resource allocation
    - Session queuing
    - Broadcast updates
    """
    
    def __init__(self, max_sessions: int = 5):
        """
        Initialize training manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
        """
        self.max_sessions = max_sessions
        self.sessions: Dict[str, TrainingSession] = {}
        self.active_sessions: Dict[str, asyncio.Task] = {}
        self.session_queue: List[TrainingSession] = []
        self._update_callbacks: Dict[str, callable] = {}
    
    def create_session(
        self,
        config: SessionConfig,
        on_update: Optional[callable] = None
    ) -> TrainingSession:
        """
        Create a new training session.
        
        Args:
            config: Session configuration
            on_update: Update callback
            
        Returns:
            Created session
        """
        session = TrainingSession(config, on_update)
        self.sessions[session.id] = session
        
        if on_update:
            self._update_callbacks[session.id] = on_update
        
        return session
    
    async def start_session(self, session_id: str) -> bool:
        """
        Start a training session.
        
        Args:
            session_id: Session ID to start
            
        Returns:
            True if started, False if queued
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        
        # Check capacity
        if len(self.active_sessions) >= self.max_sessions:
            # Queue the session
            self.session_queue.append(session)
            session.status = "queued"
            return False
        
        # Start the session
        task = asyncio.create_task(self._run_session(session))
        self.active_sessions[session_id] = task
        return True
    
    async def _run_session(self, session: TrainingSession):
        """Run a session and handle completion."""
        try:
            await session.start()
        finally:
            # Remove from active sessions
            if session.id in self.active_sessions:
                del self.active_sessions[session.id]
            
            # Start next queued session
            await self._process_queue()
    
    async def _process_queue(self):
        """Process queued sessions."""
        while (len(self.active_sessions) < self.max_sessions and 
               len(self.session_queue) > 0):
            next_session = self.session_queue.pop(0)
            task = asyncio.create_task(self._run_session(next_session))
            self.active_sessions[next_session.id] = task
    
    def pause_session(self, session_id: str):
        """Pause a running session."""
        if session_id in self.sessions:
            self.sessions[session_id].pause()
    
    def resume_session(self, session_id: str):
        """Resume a paused session."""
        if session_id in self.sessions:
            self.sessions[session_id].resume()
    
    def stop_session(self, session_id: str):
        """Stop a running session."""
        if session_id in self.sessions:
            self.sessions[session_id].stop()
        
        # Remove from queue if queued
        self.session_queue = [
            s for s in self.session_queue if s.id != session_id
        ]
    
    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state."""
        session = self.get_session(session_id)
        return session.get_state() if session else None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return [
            {
                "id": s.id,
                "status": s.status,
                "algorithm": s.config.algorithm,
                "environment": s.config.environment,
                "created_at": s.created_at.isoformat(),
            }
            for s in self.sessions.values()
        ]
    
    def get_active_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)
    
    def get_queued_count(self) -> int:
        """Get count of queued sessions."""
        return len(self.session_queue)
    
    def remove_session(self, session_id: str):
        """Remove a completed/stopped session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.status in ["completed", "stopped", "error"]:
                del self.sessions[session_id]
                if session_id in self._update_callbacks:
                    del self._update_callbacks[session_id]
    
    def cleanup_completed(self):
        """Remove all completed sessions."""
        to_remove = [
            sid for sid, s in self.sessions.items()
            if s.status in ["completed", "stopped", "error"]
        ]
        for sid in to_remove:
            self.remove_session(sid)


# Global manager instance
_manager: Optional[TrainingManager] = None


def get_manager() -> TrainingManager:
    """Get the global training manager."""
    global _manager
    if _manager is None:
        _manager = TrainingManager()
    return _manager

