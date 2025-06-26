from .active_inference import (
    BeliefPropagation,
    ExpectationMaximization,
    GradientDescentInference,
    InferenceAlgorithm,
    InferenceConfig,
    NaturalGradientInference,
    ParticleFilterInference,
    VariationalMessagePassing,
    create_inference_algorithm,
)
from .belief_state import (
    BeliefState,
    BeliefStateConfig,
    ContinuousBeliefState,
    DiscreteBeliefState,
    create_belief_state,
    create_continuous_belief_state,
    create_discrete_belief_state,
)
from .belief_update import (  # GNNBeliefUpdater,
    AttentionGraphBeliefUpdater,
    BeliefUpdateConfig,
    DirectGraphObservationModel,
    HierarchicalBeliefUpdater,
    LearnedGraphObservationModel,
    create_belief_updater,
)
from .generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    FactorizedGenerativeModel,
    GenerativeModel,
    HierarchicalGenerativeModel,
    ModelDimensions,
    ModelParameters,
    create_generative_model,
)
from .graphnn_integration import (
    DirectGraphMapper,
    GNNActiveInferenceAdapter,
    GraphFeatureAggregator,
    GraphNNIntegrationConfig,
    HierarchicalGraphIntegration,
    LearnedGraphMapper,
    create_gnn_adapter,
)
from .policy_selection import (
    ContinuousExpectedFreeEnergy,
    HierarchicalPolicySelector,
    Policy,
    PolicyConfig,
    PolicySelector,
    SophisticatedInference,
    create_policy_selector,
)
from .precision import (
    AdaptivePrecisionController,
    GradientPrecisionOptimizer,
    HierarchicalPrecisionOptimizer,
    MetaLearningPrecisionOptimizer,
    PrecisionConfig,
    PrecisionOptimizer,
    create_precision_optimizer,
)

# Import new pymdp-based implementations
from .pymdp_generative_model import (
    PyMDPGenerativeModel,
    PyMDPGenerativeModelAdapter,
    create_pymdp_generative_model,
)
from .temporal_planning import (
    AdaptiveHorizonPlanner,
    AStarPlanner,
    BeamSearchPlanner,
    MonteCarloTreeSearch,
    PlanningConfig,
    TemporalPlanner,
    TrajectorySampling,
    TreeNode,
    create_temporal_planner,
)

# from .pymdp_policy_selection import (
#     PyMDPPolicySelector,
#     PyMDPPolicyAdapter,
#     create_pymdp_policy_selector,
#     replace_discrete_expected_free_energy,
# )

# Backward compatibility alias - DiscreteExpectedFreeEnergy is now replaced by PyMDPPolicySelector
# This maintains compatibility while using the new, bug-free implementation
# DiscreteExpectedFreeEnergy = PyMDPPolicyAdapter
__all__ = [
    # Generative Models
    "ModelDimensions",
    "ModelParameters",
    "GenerativeModel",
    "DiscreteGenerativeModel",
    "ContinuousGenerativeModel",
    "HierarchicalGenerativeModel",
    "FactorizedGenerativeModel",
    "create_generative_model",
    # pymdp Generative Models
    "PyMDPGenerativeModel",
    "PyMDPGenerativeModelAdapter",
    "create_pymdp_generative_model",
    # Belief States
    "BeliefState",
    "BeliefStateConfig",
    "DiscreteBeliefState",
    "ContinuousBeliefState",
    "create_discrete_belief_state",
    "create_continuous_belief_state",
    "create_belief_state",
    # Inference
    "InferenceConfig",
    "InferenceAlgorithm",
    "VariationalMessagePassing",
    "BeliefPropagation",
    "GradientDescentInference",
    "NaturalGradientInference",
    "ExpectationMaximization",
    "ParticleFilterInference",
    "create_inference_algorithm",
    # Precision
    "PrecisionConfig",
    "PrecisionOptimizer",
    "GradientPrecisionOptimizer",
    "HierarchicalPrecisionOptimizer",
    "MetaLearningPrecisionOptimizer",
    "AdaptivePrecisionController",
    "create_precision_optimizer",
    # Policy Selection
    "PolicyConfig",
    "Policy",
    "PolicySelector",
    "DiscreteExpectedFreeEnergy",  # Backward compatibility
    "ContinuousExpectedFreeEnergy",
    "HierarchicalPolicySelector",
    "SophisticatedInference",
    "create_policy_selector",
    # pymdp Policy Selection
    # "PyMDPPolicySelector",
    # "PyMDPPolicyAdapter",
    # "create_pymdp_policy_selector",
    # "replace_discrete_expected_free_energy",
    # Temporal Planning
    "PlanningConfig",
    "TreeNode",
    "TemporalPlanner",
    "MonteCarloTreeSearch",
    "BeamSearchPlanner",
    "AStarPlanner",
    "TrajectorySampling",
    "AdaptiveHorizonPlanner",
    "create_temporal_planner",
    # GNN Integration
    "GraphNNIntegrationConfig",
    "DirectGraphMapper",
    "LearnedGraphMapper",
    "GNNActiveInferenceAdapter",
    "GraphFeatureAggregator",
    "HierarchicalGraphIntegration",
    "create_gnn_adapter",
    # Belief Update
    "BeliefUpdateConfig",
    "DirectGraphObservationModel",
    "LearnedGraphObservationModel",
    # "GNNBeliefUpdater",
    "AttentionGraphBeliefUpdater",
    "HierarchicalBeliefUpdater",
    "create_belief_updater",
]
