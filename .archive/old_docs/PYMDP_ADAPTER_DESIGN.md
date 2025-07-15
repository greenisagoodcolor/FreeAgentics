# PyMDP API Compatibility Adapter Design

## Overview

The PyMDP API Compatibility Adapter (`agents/pymdp_adapter.py`) provides a thin translation layer between PyMDP's actual API behavior and the expected API behavior in the FreeAgentics codebase. This adapter follows strict principles with ZERO fallbacks - operations either work correctly or raise exceptions.

## Critical Mission

This adapter is critical for the VC demo. It must handle PyMDP API correctly without any graceful degradation or mock operations.

## Design Principles

### 1. ZERO Fallbacks
- No graceful degradation
- No mock operations  
- No default values
- Operations work or raise clear exceptions

### 2. Strict Type Checking
- All inputs validated with `isinstance()` checks
- All outputs converted to exact expected types
- No implicit type coercion

### 3. Real PyMDP Operations Only
- Works only with actual PyMDP agent instances
- No simulation or approximation
- Direct pass-through to PyMDP with type adaptation

### 4. Clear Error Propagation
- PyMDP errors are propagated, not masked
- Additional validation errors are specific and descriptive
- Stack traces preserved for debugging

## Key Functionality

### sample_action() Adapter

Handles the tuple/int return value issue from `base_agent.py:397`:

```python
# PyMDP returns: numpy.ndarray[float64] with shape (1,)
# Adapter returns: exact int type

action_result = pymdp_agent.sample_action()  # Returns np.array([2.0])
action_int = adapter.sample_action(pymdp_agent)  # Returns 2 (int)
```

### infer_states() Adapter

Handles various observation formats and PyMDP's return type variations:

```python
# Accepts: int, List[int], or numpy array observations
# Returns: List[NDArray[np.floating]] 

beliefs = adapter.infer_states(agent, 0)  # int observation
beliefs = adapter.infer_states(agent, [0, 1])  # list observation
beliefs = adapter.infer_states(agent, np.array([0]))  # array observation
```

### infer_policies() Adapter

Validates PyMDP's policy inference return format:

```python
# Returns: Tuple[NDArray[np.floating], NDArray[np.floating]]
q_pi, G = adapter.infer_policies(agent)
```

### validate_agent_state()

Ensures PyMDP agent has required attributes:

```python
# Checks for A, B matrices (required)
# Returns True if valid, raises RuntimeError if not
is_valid = adapter.validate_agent_state(agent)
```

### safe_array_conversion()

Strict array-to-scalar conversion utility:

```python
# Converts numpy arrays/scalars to exact Python types
int_val = adapter.safe_array_conversion(np.array([5.9]), int)  # Returns 5
float_val = adapter.safe_array_conversion(np.int64(7), float)  # Returns 7.0
```

## Usage Example

```python
from agents.pymdp_adapter import PyMDPCompatibilityAdapter
from pymdp.agent import Agent as PyMDPAgent
from pymdp import utils

# Create adapter
adapter = PyMDPCompatibilityAdapter()

# Create PyMDP agent
A = utils.random_A_matrix([3], [3])
B = utils.random_B_matrix([3], [3])
agent = PyMDPAgent(A=A, B=B)

# Full workflow with adapter
observation = 0

# Step 1: Update beliefs
beliefs = adapter.infer_states(agent, observation)

# Step 2: Infer policies  
q_pi, G = adapter.infer_policies(agent)

# Step 3: Sample action (returns exact int)
action = adapter.sample_action(agent)  # type: int
```

## Error Handling

The adapter provides clear, specific errors:

```python
# Wrong agent type
adapter.sample_action("not an agent")
# TypeError: Expected PyMDPAgent, got <class 'str'>

# Invalid return format
# RuntimeError: PyMDP sample_action() returned shape (2,), expected (1,)

# Missing initialization
# AttributeError: 'Agent' object has no attribute 'q_pi'
```

## Testing

Comprehensive test coverage in `tests/unit/test_pymdp_adapter_strict.py`:

- Unit tests for each adapter method
- Integration tests with real PyMDP operations
- Error handling validation
- Performance benchmarks
- Design decision documentation

All tests follow TDD principles with RED-GREEN-REFACTOR cycle.

## Implementation Notes

1. **No Fallbacks**: If PyMDP changes its API, the adapter will fail loudly rather than silently returning incorrect results.

2. **Type Strictness**: The adapter enforces exact types (e.g., `int` not `np.int64`) to ensure compatibility with type-sensitive code.

3. **Minimal Overhead**: The adapter adds minimal performance overhead - primarily type checking and conversion.

4. **Future Compatibility**: New PyMDP API changes can be handled by updating the adapter methods while maintaining the same interface.

## Conclusion

This adapter serves as a critical compatibility layer for the VC demo, ensuring PyMDP operations work correctly with strict type safety and zero tolerance for errors. It follows TDD principles and provides comprehensive test coverage to validate all functionality.