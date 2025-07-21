#!/usr/bin/env python3
"""Fix syntax errors from bad line breaks."""

import re
from pathlib import Path

def fix_syntax_errors():
    """Fix specific syntax errors."""
    
    # Fix enhanced_agent_coordinator.py
    file = Path("agents/enhanced_agent_coordinator.py")
    if file.exists():
        content = file.read_text()
        # Fix line 68
        content = re.sub(
            r'logger\.info\(\s*f"Agent Coordinator initialized - PyMDP:\s*\{PYMDP_AVAILABLE\},\s*Performance Mode',
            'logger.info(f"Agent Coordinator initialized - PyMDP: {PYMDP_AVAILABLE}, Performance Mode',
            content
        )
        file.write_text(content)
        print(f"Fixed: {file}")
    
    # Fix memory_optimization files
    files_to_fix = {
        "agents/memory_optimization/agent_memory_optimizer.py": [
            (r'"""Use pickle for now\.  In production, use msgpack or protobuf \-\s*only used with trusted data', 
             '"""Use pickle for now.  In production, use msgpack or protobuf - only used with trusted data')
        ],
        "agents/memory_optimization/efficient_structures.py": [
            (r'if use_memory_mapping or\s*$', 'if use_memory_mapping or ')
        ],
        "agents/memory_optimization/enhanced_memory_profiler.py": [
            (r'logger\.info\(\s*f"Memory snapshot saved:', 
             'logger.info(f"Memory snapshot saved:')
        ],
        "agents/memory_optimization/gc_tuning.py": [
            (r'logger\.info\(\s*f"Monitored GC for \{monitor_time\}s\.  Total collections:\s*\{collections\},\s*Total\s*duration\)', 
             'logger.info(f"Monitored GC for {monitor_time}s.  Total collections: {collections}, Total duration")')
        ],
        "agents/memory_optimization/lifecycle_manager.py": [
            (r'if profile\.state == AgentLifecycleState\.ACTIVE and\s*$',
             'if profile.state == AgentLifecycleState.ACTIVE and ')
        ]
    }
    
    for filepath, patterns in files_to_fix.items():
        file = Path(filepath)
        if file.exists():
            content = file.read_text()
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            file.write_text(content)
            print(f"Fixed: {file}")

if __name__ == "__main__":
    fix_syntax_errors()