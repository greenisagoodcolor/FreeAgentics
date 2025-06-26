# Active inference module for agents
from .generative_model import (
    ContinuousGenerativeModel,
    DiscreteGenerativeModel,
    FactorizedGenerativeModel,
    HierarchicalGenerativeModel,
    ModelDimensions,
    ModelParameters,
    create_generative_model,
)
from .precision import (
    AdaptivePrecisionController,
    GradientPrecisionOptimizer,
    HierarchicalPrecisionOptimizer,
    MetaLearningPrecisionOptimizer,
    PrecisionConfig,
    create_precision_optimizer,
)

__all__ = [
    # Generative models
    "ModelDimensions",
    "ModelParameters",
    "DiscreteGenerativeModel",
    "ContinuousGenerativeModel",
    "HierarchicalGenerativeModel",
    "FactorizedGenerativeModel",
    "create_generative_model",
    # Precision optimization
    "PrecisionConfig",
    "GradientPrecisionOptimizer",
    "HierarchicalPrecisionOptimizer",
    "MetaLearningPrecisionOptimizer",
    "AdaptivePrecisionController",
    "create_precision_optimizer",
]
