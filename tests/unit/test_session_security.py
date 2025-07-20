"""Security-critical tests for session management following TDD principles.

This test suite covers session security:
- Session creation and validation
- Session expiration
- Session fixation prevention
- Concurrent session limits
- Session storage security
"""

import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import pytest


class SessionStore:
    """Secure session storage implementation."""
    
    def __init__(self, max_sessions_per_user: int = 5):
        self._sessions: Dict[str, Dict] = {}
        self._user_sessions: Dict[str, list] = {}
        self.max_sessions_per_user = max_sessions_per_user
        self.session_timeout = timedelta(minutes=30)
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new session for a user.
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Session ID
            
        Raises:
            ValueError: If user has too many active sessions
        """
        # Check concurrent session limit
        self._cleanup_expired_sessions()
        user_sessions = self._user_sessions.get(user_id, [])
        
        if len(user_sessions) >= self.max_sessions_per_user:
            # Optionally, remove oldest session
            oldest_session_id = user_sessions[0]
            self.invalidate_session(oldest_session_id)
        
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store session data
        session_data = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_accessed': datetime.utcnow(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'data': {}
        }
        
        self._sessions[session_id] = session_data
        
        # Track user sessions
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = []
        self._user_sessions[user_id].append(session_id)
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Optional[Dict]:
        """Validate and retrieve session data.
        
        Args:
            session_id: Session ID to validate
            ip_address: Optional IP address to verify
            
        Returns:
            Session data if valid, None otherwise
        """
        session = self._sessions.get(session_id)
        
        if not session:
            return None
        
        # Check expiration
        if datetime.utcnow() - session['last_accessed'] > self.session_timeout:
            self.invalidate_session(session_id)
            return None
        
        # Verify IP address if provided (optional IP pinning)
        if ip_address and session['ip_address'] != ip_address:
            # Log potential session hijacking attempt
            return None
        
        # Update last accessed time
        session['last_accessed'] = datetime.utcnow()
        
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session.
        
        Args:
            session_id: Session ID to invalidate
            
        Returns:
            True if session was invalidated, False if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Remove from user sessions
        user_id = session['user_id']
        if user_id in self._user_sessions:
            self._user_sessions[user_id] = [
                sid for sid in self._user_sessions[user_id] 
                if sid != session_id
            ]
            if not self._user_sessions[user_id]:
                del self._user_sessions[user_id]
        
        # Remove session
        del self._sessions[session_id]
        return True
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of sessions invalidated
        """
        session_ids = self._user_sessions.get(user_id, [])[:]
        count = 0
        
        for session_id in session_ids:
            if self.invalidate_session(session_id):
                count += 1
        
        return count
    
    def regenerate_session_id(self, old_session_id: str) -> Optional[str]:
        """Regenerate session ID to prevent fixation attacks.
        
        Args:
            old_session_id: Current session ID
            
        Returns:
            New session ID if successful, None otherwise
        """
        session = self._sessions.get(old_session_id)
        if not session:
            return None
        
        # Generate new ID
        new_session_id = secrets.token_urlsafe(32)
        
        # Move session data
        self._sessions[new_session_id] = session
        del self._sessions[old_session_id]
        
        # Update user session tracking
        user_id = session['user_id']
        if user_id in self._user_sessions:
            self._user_sessions[user_id] = [
                new_session_id if sid == old_session_id else sid
                for sid in self._user_sessions[user_id]
            ]
        
        return new_session_id
    
    def get_user_sessions(self, user_id: str) -> list:
        """Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session data
        """
        self._cleanup_expired_sessions()
        session_ids = self._user_sessions.get(user_id, [])
        
        sessions = []
        for session_id in session_ids:
            session = self._sessions.get(session_id)
            if session:
                sessions.append({
                    'session_id': session_id,
                    'created_at': session['created_at'],
                    'last_accessed': session['last_accessed'],
                    'ip_address': session['ip_address']
                })
        
        return sessions
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if current_time - session['last_accessed'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)


class TestSessionCreation:
    """Test secure session creation."""
    
    def test_create_session_generates_secure_id(self):
        """Test that session IDs are cryptographically secure."""
        # Arrange
        store = SessionStore()
        
        # Act
        session_id = store.create_session(
            user_id="user-123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Assert
        assert session_id is not None
        assert len(session_id) >= 32  # Sufficient entropy
        assert isinstance(session_id, str)
        # Should be URL-safe
        assert all(c.isalnum() or c in '-_' for c in session_id)
    
    def test_create_multiple_sessions_unique_ids(self):
        """Test that each session gets a unique ID."""
        # Arrange
        store = SessionStore()
        session_ids = []
        
        # Act
        for i in range(100):
            session_id = store.create_session(
                user_id=f"user-{i}",
                ip_address="192.168.1.1",
                user_agent="Test"
            )
            session_ids.append(session_id)
        
        # Assert
        assert len(set(session_ids)) == 100  # All unique
    
    def test_concurrent_session_limit_enforcement(self):
        """Test that concurrent session limits are enforced."""
        # Arrange
        store = SessionStore(max_sessions_per_user=3)
        user_id = "user-123"
        
        # Act - Create max sessions
        sessions = []
        for i in range(3):
            session_id = store.create_session(
                user_id=user_id,
                ip_address=f"192.168.1.{i}",
                user_agent="Test"
            )
            sessions.append(session_id)
        
        # Create one more (should remove oldest)
        new_session = store.create_session(
            user_id=user_id,
            ip_address="192.168.1.99",
            user_agent="Test"
        )
        
        # Assert
        assert new_session is not None
        # Oldest session should be invalidated
        assert store.validate_session(sessions[0]) is None
        # Newer sessions should still be valid
        assert store.validate_session(sessions[1]) is not None
        assert store.validate_session(sessions[2]) is not None


class TestSessionValidation:
    """Test session validation and security checks."""
    
    def test_validate_valid_session(self):
        """Test validating a valid session."""
        # Arrange
        store = SessionStore()
        session_id = store.create_session(
            user_id="user-123",
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        # Act
        session = store.validate_session(session_id)
        
        # Assert
        assert session is not None
        assert session['user_id'] == "user-123"
        assert session['ip_address'] == "192.168.1.1"
    
    def test_validate_invalid_session(self):
        """Test that invalid session IDs return None."""
        # Arrange
        store = SessionStore()
        
        # Act
        session = store.validate_session("invalid-session-id")
        
        # Assert
        assert session is None
    
    def test_validate_expired_session(self):
        """Test that expired sessions are invalidated."""
        # Arrange
        store = SessionStore()
        store.session_timeout = timedelta(seconds=1)  # Short timeout for testing
        
        session_id = store.create_session(
            user_id="user-123",
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Act
        session = store.validate_session(session_id)
        
        # Assert
        assert session is None
    
    def test_ip_address_validation(self):
        """Test IP address validation for session hijacking prevention."""
        # Arrange
        store = SessionStore()
        session_id = store.create_session(
            user_id="user-123",
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        # Act - Validate with correct IP
        session = store.validate_session(session_id, ip_address="192.168.1.1")
        assert session is not None
        
        # Act - Validate with different IP
        session = store.validate_session(session_id, ip_address="10.0.0.1")
        assert session is None  # Should reject different IP


class TestSessionSecurity:
    """Test session security features."""
    
    def test_session_fixation_prevention(self):
        """Test regenerating session ID to prevent fixation attacks."""
        # Arrange
        store = SessionStore()
        old_session_id = store.create_session(
            user_id="user-123",
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        # Act
        new_session_id = store.regenerate_session_id(old_session_id)
        
        # Assert
        assert new_session_id is not None
        assert new_session_id != old_session_id
        # Old session should be invalid
        assert store.validate_session(old_session_id) is None
        # New session should be valid with same data
        session = store.validate_session(new_session_id)
        assert session is not None
        assert session['user_id'] == "user-123"
    
    def test_invalidate_user_sessions(self):
        """Test invalidating all sessions for a user."""
        # Arrange
        store = SessionStore()
        user_id = "user-123"
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session_id = store.create_session(
                user_id=user_id,
                ip_address=f"192.168.1.{i}",
                user_agent="Test"
            )
            sessions.append(session_id)
        
        # Act
        count = store.invalidate_user_sessions(user_id)
        
        # Assert
        assert count == 3
        # All sessions should be invalid
        for session_id in sessions:
            assert store.validate_session(session_id) is None
    
    def test_get_user_sessions(self):
        """Test retrieving all active sessions for a user."""
        # Arrange
        store = SessionStore()
        user_id = "user-123"
        
        # Create sessions from different devices
        session1 = store.create_session(
            user_id=user_id,
            ip_address="192.168.1.1",
            user_agent="Desktop"
        )
        session2 = store.create_session(
            user_id=user_id,
            ip_address="192.168.1.2",
            user_agent="Mobile"
        )
        
        # Act
        sessions = store.get_user_sessions(user_id)
        
        # Assert
        assert len(sessions) == 2
        assert any(s['session_id'] == session1 for s in sessions)
        assert any(s['session_id'] == session2 for s in sessions)
        # Should include metadata
        assert all('ip_address' in s for s in sessions)
        assert all('created_at' in s for s in sessions)
    
    def test_session_timeout_updates(self):
        """Test that session timeout is updated on access."""
        # Arrange
        store = SessionStore()
        session_id = store.create_session(
            user_id="user-123",
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        # Get initial last accessed time
        session1 = store.validate_session(session_id)
        last_accessed1 = session1['last_accessed']
        
        # Wait a bit
        time.sleep(0.1)
        
        # Act - Access session again
        session2 = store.validate_session(session_id)
        last_accessed2 = session2['last_accessed']
        
        # Assert - Last accessed should be updated
        assert last_accessed2 > last_accessed1