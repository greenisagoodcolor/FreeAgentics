# PyTorch vs PyMDP: Architectural Separation

## Overview

FreeAgentics uses a **hybrid architecture** that cleanly separates PyTorch and PyMDP responsibilities to avoid redundancy and conflicts while leveraging the strengths of each library.

## Architectural Principles

### 1. **Core Active Inference: PyMDP Only**
- **Mathematical Operations**: All core Active Inference math uses PyMDP
- **Belief Updates**: PyMDP's validated categorical distributions
- **Policy Selection**: PyMDP's proven algorithms
- **Free Energy Computation**: PyMDP's reference implementations

### 2. **Neural Components: PyTorch Only**
- **Graph Neural Networks (GNN)**: PyTorch Geometric for graph processing
- **Continuous State Models**: Neural network-based generative models
- **Parameter Learning**: Gradient-based optimization
- **GPU Acceleration**: When neural components require it

### 3. **Clean Interfaces**
- **Conversion Layer**: Minimal, well-defined conversion between NumPy ↔ PyTorch
- **Optional Dependencies**: PyTorch is optional unless neural features are needed
- **Graceful Degradation**: System works without PyTorch for basic Active Inference

## Component Mapping

| Component | Library | Justification |
|-----------|---------|---------------|
| **Core Inference** | PyMDP | Validated mathematical implementations |
| **Discrete Models** | PyMDP | Reference implementation for categorical distributions |
| **Belief Updates** | PyMDP | Proven Bayesian inference algorithms |
| **Policy Selection** | PyMDP | Validated expected free energy computation |
| **GNN Layers** | PyTorch | Specialized graph neural network operations |
| **Continuous Models** | PyTorch | Neural network-based generative models |
| **GPU Acceleration** | PyTorch | When needed for neural components |
| **Gradient Learning** | PyTorch | Parameter optimization for neural models |

## Installation Options

### Basic Installation (PyMDP only)
```bash
pip install freeagentics
```
- Includes core Active Inference functionality
- No PyTorch dependency
- Works for most use cases

### Neural Features Installation
```bash
pip install freeagentics[neural]
```
- Includes PyTorch and PyTorch Geometric
- Enables GNN and continuous model features
- Required for neural network-based components

## Code Organization

### Core Active Inference (PyMDP)
```
inference/engine/
├── active_inference.py          # PyMDP-based inference algorithms
├── pymdp_policy_selector.py     # Pure PyMDP policy selection
├── pymdp_generative_model.py    # PyMDP matrix operations
└── belief_update.py             # PyMDP belief updates
```

### Neural Components (PyTorch)
```
inference/gnn/                   # Graph Neural Networks
├── layers.py                    # PyTorch Geometric layers
├── active_inference.py          # GNN-enhanced Active Inference
└── model_mapper.py              # Neural model mapping

inference/engine/
├── generative_model.py          # Neural generative models
├── parameter_learning.py        # Gradient-based learning
└── computational_optimization.py # GPU acceleration
```

### Conversion Layer
```
inference/engine/
└── pymdp_integration.py         # Clean NumPy ↔ PyTorch conversion
```

## Testing Strategy

### Separate Test Suites
- **Core Tests**: Run without PyTorch dependency
- **Neural Tests**: Require PyTorch installation
- **Integration Tests**: Test conversion layer

### Coverage Analysis
- **Core Coverage**: Measured without PyTorch conflicts
- **Neural Coverage**: Measured separately with PyTorch
- **Combined Coverage**: Aggregated results

## Migration Path

### Phase 1: Immediate (Current)
- Make PyTorch optional dependency
- Fix import conflicts with graceful degradation
- Ensure core functionality works PyMDP-only

### Phase 2: Refactoring
- Move neural-specific code to separate modules
- Clean up conversion layer
- Optimize PyMDP-only paths

### Phase 3: Optimization
- Remove unnecessary PyTorch dependencies from core
- Optimize pure PyMDP performance
- Add neural features as plugins

## Benefits

1. **No Version Conflicts**: Core system avoids PyTorch compatibility issues
2. **Reduced Complexity**: Clear separation of mathematical vs neural operations
3. **Better Testing**: Can test core functionality without PyTorch
4. **Performance**: PyMDP optimizations for core inference
5. **Modularity**: Neural features are optional additions
6. **Maintenance**: Easier to maintain separate concerns

## Implementation Guidelines

### For Core Active Inference
- Use PyMDP for all mathematical operations
- Import PyTorch only with graceful degradation
- Prefer NumPy arrays for data exchange
- Use PyMDP's validated algorithms

### For Neural Components
- Use PyTorch for neural network operations
- Implement clean conversion to/from NumPy
- Make GPU usage optional
- Provide fallback to CPU/NumPy implementations

### For Integration
- Minimize conversion overhead
- Validate matrix conventions match
- Handle tensor/array type checking
- Provide clear error messages for missing dependencies 