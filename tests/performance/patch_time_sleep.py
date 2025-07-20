"""Monkey patch time.sleep for performance tests to use real CPU work."""

import time
import numpy as np

# Store the original sleep function
_original_sleep = time.sleep


def patched_sleep(duration):
    """Replace time.sleep with actual CPU work."""
    if duration <= 0:
        return
    
    start_time = time.time()
    
    # For very short sleeps (< 1ms), do minimal work
    if duration < 0.001:
        _ = sum(i**2 for i in range(100))
    else:
        # Perform CPU work that roughly matches the duration
        while time.time() - start_time < duration:
            # Adaptive work based on remaining time
            remaining = duration - (time.time() - start_time)
            if remaining < 0.001:
                break
            
            # Scale work to remaining time
            if remaining > 0.01:
                # Heavier work for longer durations
                size = min(50, int(remaining * 1000))
                if size > 0:
                    matrix = np.random.rand(size, size)
                    _ = matrix.sum()
            else:
                # Light work for short durations
                _ = sum(i**2 for i in range(1000))


def enable_patch():
    """Enable the time.sleep patch."""
    time.sleep = patched_sleep


def disable_patch():
    """Disable the time.sleep patch."""
    time.sleep = _original_sleep


# Auto-patch when imported from performance or benchmark directories
import inspect
import os

frame = inspect.currentframe()
if frame and frame.f_back:
    caller_file = frame.f_back.f_globals.get('__file__', '')
    if 'performance' in caller_file or 'benchmark' in caller_file:
        enable_patch()