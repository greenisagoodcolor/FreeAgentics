"""Memory optimization modules for agents."""

from .agent_memory_optimizer import (
    AgentMemoryOptimizer,
    CompressedHistory,
    OptimizedAgentMemory,
    SharedAgentParameters,
    SharedComputationPool,
    SharedObservationBuffer,
    get_agent_optimizer,
)
from .belief_compression import BeliefCompressor, CompressedBeliefPool, SparseBeliefState
from .efficient_structures import (
    CompactActionHistory,
    CompactKnowledgeGraph,
    EfficientTemporalSequence,
    LazyBeliefArray,
    MemoryMappedBuffer,
    MemoryStats,
    benchmark_data_structures,
    create_efficient_belief_buffer,
)
from .gc_tuning import (
    AdaptiveGCTuner,
    GCContextManager,
    GCStats,
    get_gc_tuner,
    optimize_gc_for_agents,
)
from .lifecycle_manager import (
    AgentLifecycleState,
    AgentMemoryLifecycleManager,
    AgentMemoryProfile,
    MemoryUsageSnapshot,
    cleanup_agent_memory,
    get_global_lifecycle_manager,
    get_memory_statistics,
    managed_agent_memory,
    register_agent_memory,
    update_agent_memory_usage,
)
from .matrix_pooling import (
    MatrixOperationPool,
    MatrixPool,
    PooledMatrix,
    get_global_pool,
    pooled_dot,
    pooled_einsum,
    pooled_matmul,
    pooled_matrix,
)
from .memory_profiler import (
    AdvancedMemoryProfiler,
    AllocationPattern,
    MemorySnapshot,
    get_memory_profiler,
)

__all__ = [
    # Belief compression
    "BeliefCompressor",
    "SparseBeliefState",
    "CompressedBeliefPool",
    # Matrix pooling
    "MatrixOperationPool",
    "MatrixPool",
    "PooledMatrix",
    "get_global_pool",
    "pooled_dot",
    "pooled_einsum",
    "pooled_matmul",
    "pooled_matrix",
    # Lifecycle management
    "AgentLifecycleState",
    "AgentMemoryProfile",
    "AgentMemoryLifecycleManager",
    "MemoryUsageSnapshot",
    "get_global_lifecycle_manager",
    "managed_agent_memory",
    "register_agent_memory",
    "update_agent_memory_usage",
    "cleanup_agent_memory",
    "get_memory_statistics",
    # Efficient data structures
    "MemoryStats",
    "LazyBeliefArray",
    "MemoryMappedBuffer",
    "CompactActionHistory",
    "EfficientTemporalSequence",
    "CompactKnowledgeGraph",
    "create_efficient_belief_buffer",
    "benchmark_data_structures",
    # GC tuning
    "AdaptiveGCTuner",
    "GCStats",
    "GCContextManager",
    "get_gc_tuner",
    "optimize_gc_for_agents",
    # Memory profiling
    "AdvancedMemoryProfiler",
    "MemorySnapshot",
    "AllocationPattern",
    "get_memory_profiler",
    # Agent memory optimization
    "AgentMemoryOptimizer",
    "OptimizedAgentMemory",
    "SharedAgentParameters",
    "SharedObservationBuffer",
    "CompressedHistory",
    "SharedComputationPool",
    "get_agent_optimizer",
]
