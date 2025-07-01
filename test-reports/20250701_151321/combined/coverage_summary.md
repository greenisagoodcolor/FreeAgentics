# FreeAgentics Comprehensive Coverage Report

**Generated:** Tue Jul  1 15:13:45 CEST 2025
**Timestamp:** 20250701_151321

## Coverage Summary

### Backend Coverage (Python)

#### Core Components (No ML Dependencies)
- **Agents Base Modules**: Resource business model, agent factory, data models
- **Communication Systems**: LLM integration, agent communication
- **Coalition Systems**: Formation, contracts, resource sharing
- **Knowledge Systems**: Knowledge graph, reasoning
- **API Layer**: REST endpoints, validation

#### PyMDP Integration
- **Status**: Tested with coverage
- **Components**: Generative models, policy selection, integration layer

#### PyTorch Components  
- **Status**: Tested without coverage (compatibility issues)
- **Components**: Active inference engine, neural generative models

#### Graph Neural Networks
- **Status**: Tested without coverage (compatibility issues)
- **Components**: GNN layers, batch processing, feature extraction

### Frontend Coverage (TypeScript/JavaScript)

- **Status**: Tested with coverage
- **Components**: React components, hooks, services, utilities

## Architecture Notes

### Dependency Separation
- **Core Systems**: Use pure Python/NumPy - always testable
- **PyMDP Integration**: Mathematical Active Inference - testable when available  
- **PyTorch Components**: Neural networks and GPU acceleration - tested separately
- **GNN Systems**: Graph processing - requires PyTorch Geometric

### Testing Strategy
1. **Core functionality** always tested with full coverage
2. **PyMDP components** tested with coverage when available
3. **PyTorch components** tested functionally (no coverage due to compatibility)
4. **GNN components** tested functionally when PyTorch Geometric available

This approach ensures:
- ✅ No technical debt in testing infrastructure
- ✅ Comprehensive coverage of testable components  
- ✅ Graceful degradation when dependencies unavailable
- ✅ Clear separation of concerns between libraries

