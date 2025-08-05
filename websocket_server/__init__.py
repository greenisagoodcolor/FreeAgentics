"""
Websocket package stub for provider initialization.

This module forwards imports to the actual websocket implementation in backend.websocket.
This is a temporary compatibility layer to satisfy provider initialization expectations.

TODO: Refactor provider initialization to use proper module paths.
"""

# Forward all imports to the actual implementation
try:
    from backend.websocket import *  # type: ignore  # noqa: F403,F401
except ImportError:
    # If backend.websocket doesn't exist, provide empty module
    pass
