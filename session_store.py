#!/usr/bin/env python3
"""
Simple file-based session store for steering functionality

This allows sessions to be shared between processes
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SessionStore:
    """Simple file-based session store"""

    def __init__(self, store_file: str = "steering_sessions.json"):
        self.store_file = Path(store_file)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.last_load_time = 0
        self.load_sessions()

    def load_sessions(self):
        """Load sessions from file"""
        try:
            if self.store_file.exists():
                with open(self.store_file, "r") as f:
                    data = json.load(f)
                    self.sessions = data.get("sessions", {})
                    self.last_load_time = time.time()
                    logger.info(f"[SESSION_STORE] Loaded {len(self.sessions)} sessions")
        except Exception as e:
            logger.warning(f"[SESSION_STORE] Error loading sessions: {e}")
            self.sessions = {}

    def save_sessions(self):
        """Save sessions to file"""
        try:
            data = {"sessions": self.sessions, "timestamp": time.time()}
            with open(self.store_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"[SESSION_STORE] Saved {len(self.sessions)} sessions")
        except Exception as e:
            logger.error(f"[SESSION_STORE] Error saving sessions: {e}")

    def add_session(self, session_id: str, session_info: Dict[str, Any]):
        """Add a session to the store"""
        # Convert state object to serializable format
        serializable_info = self._make_serializable(session_info)
        self.sessions[session_id] = serializable_info
        self.save_sessions()
        logger.info(f"[SESSION_STORE] Added session {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session from the store"""
        # Reload if file has been updated
        if self.store_file.exists():
            file_mtime = self.store_file.stat().st_mtime
            if file_mtime > self.last_load_time:
                self.load_sessions()

        return self.sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all sessions"""
        # Reload if file has been updated
        if self.store_file.exists():
            file_mtime = self.store_file.stat().st_mtime
            if file_mtime > self.last_load_time:
                self.load_sessions()

        return self.sessions

    def remove_session(self, session_id: str):
        """Remove a session from the store"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions()
            logger.info(f"[SESSION_STORE] Removed session {session_id}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable"""
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    try:
                        result[key] = self._make_serializable(value)
                    except:
                        result[key] = str(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)


# Global session store instance
session_store = SessionStore()
