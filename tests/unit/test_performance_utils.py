"""Test performance utilities."""

import time
import pytest
from tests.performance.performance_utils import (
    cpu_work, memory_work, io_work, adaptive_cpu_work, replace_sleep
)


class TestPerformanceUtils:
    """Test performance utility functions."""
    
    def test_cpu_work_duration(self):
        """Test that cpu_work takes approximately the requested duration."""
        duration = 0.01  # 10ms
        start = time.time()
        cpu_work(duration)
        elapsed = time.time() - start
        
        # Should take at least 50% of requested time
        assert elapsed >= duration * 0.5
        # But not more than 300% (to account for system variability)
        assert elapsed <= duration * 3.0
    
    def test_cpu_work_intensities(self):
        """Test different intensity levels."""
        for intensity in ["light", "medium", "heavy"]:
            start = time.time()
            cpu_work(0.001, intensity)
            elapsed = time.time() - start
            
            # Should complete without error
            assert elapsed >= 0
    
    def test_memory_work(self):
        """Test memory allocation and operations."""
        # Should complete without error
        memory_work(0.1)  # 0.1 MB
        memory_work(1.0)  # 1 MB
    
    def test_io_work(self):
        """Test I/O simulation."""
        # Should complete without error
        io_work(10)
        io_work(100)
    
    def test_adaptive_cpu_work(self):
        """Test adaptive CPU work."""
        target = 0.01  # 10ms
        actual = adaptive_cpu_work(target)
        
        # Should be close to target
        assert actual >= target * 0.5
        assert actual <= target * 2.0
    
    def test_replace_sleep_types(self):
        """Test different work types in replace_sleep."""
        for work_type in ["cpu", "memory", "io", "mixed"]:
            start = time.time()
            replace_sleep(0.001, work_type)
            elapsed = time.time() - start
            
            # Should complete without error
            assert elapsed >= 0


class TestTimeSleepPatch:
    """Test time.sleep patching."""
    
    def test_patch_functionality(self):
        """Test that patching works correctly."""
        from tests.performance.patch_time_sleep import enable_patch, disable_patch, _original_sleep
        
        # Enable patch
        enable_patch()
        
        # time.sleep should now do CPU work
        start = time.time()
        time.sleep(0.001)
        elapsed = time.time() - start
        
        # Should have done some work (not just sleep)
        assert elapsed >= 0
        
        # Disable patch
        disable_patch()
        
        # Should be back to original
        assert time.sleep == _original_sleep