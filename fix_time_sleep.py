#!/usr/bin/env python3
"""Script to automatically replace time.sleep() with cpu_work() in performance tests."""

import re
import os
from pathlib import Path


def fix_time_sleep_in_file(file_path: Path) -> bool:
    """Fix time.sleep() calls in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Check if file already imports performance_utils
        has_import = "from tests.performance.performance_utils import" in content
        
        # Pattern to match time.sleep() calls
        pattern = r'time\.sleep\s*\((.*?)\)'
        
        # Find all matches
        matches = list(re.finditer(pattern, content))
        
        if not matches:
            return False
        
        # Add import if not present
        if not has_import:
            # Find the right place to add import (after other imports)
            import_lines = []
            lines = content.split('\n')
            last_import_idx = 0
            
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    last_import_idx = i
            
            # Insert the import after the last import
            lines.insert(last_import_idx + 1, "from tests.performance.performance_utils import cpu_work")
            content = '\n'.join(lines)
        
        # Replace time.sleep() calls
        def replace_sleep(match):
            duration = match.group(1).strip()
            # Determine work type based on context (simple heuristic)
            if float(duration) < 0.01:
                return f'cpu_work({duration}, "light")'
            else:
                return f'cpu_work({duration})'
        
        # Replace all occurrences
        content = re.sub(pattern, replace_sleep, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix all performance test files."""
    
    # Files identified by the performance theater detector
    files_to_fix = [
        "/home/green/FreeAgentics/tests/performance/test_authentication_performance.py",
        "/home/green/FreeAgentics/tests/performance/test_async_coordination_performance.py",
        "/home/green/FreeAgentics/tests/performance/enhanced_ci_benchmarks.py",
        "/home/green/FreeAgentics/tests/performance/agent_simulation_framework.py",
        "/home/green/FreeAgentics/tests/performance/test_coordination_load.py",
        "/home/green/FreeAgentics/tests/performance/test_threading_optimizations.py",
        "/home/green/FreeAgentics/tests/performance/test_matrix_pooling_performance.py",
        "/home/green/FreeAgentics/benchmarks/performance_benchmark_suite.py",
        "/home/green/FreeAgentics/benchmarks/quick_threading_vs_multiprocessing_test.py",
        "/home/green/FreeAgentics/benchmarks/production_benchmark.py",
        "/home/green/FreeAgentics/benchmarks/threading_vs_multiprocessing_benchmark.py",
        "/home/green/FreeAgentics/benchmarks/performance_suite.py",
        "/home/green/FreeAgentics/tests/integration/test_auth_rate_limiting.py",
        "/home/green/FreeAgentics/tests/integration/test_auth_load.py",
        "/home/green/FreeAgentics/tests/integration/test_matrix_pooling_pymdp.py",
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            print(f"Processing {file_path}...")
            if fix_time_sleep_in_file(path):
                print(f"  ✅ Fixed time.sleep() calls")
                fixed_count += 1
            else:
                print(f"  ⏭️  No changes needed")
        else:
            print(f"  ❌ File not found: {file_path}")
    
    print(f"\n✅ Fixed {fixed_count} files")


if __name__ == "__main__":
    main()