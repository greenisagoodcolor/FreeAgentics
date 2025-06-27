."""

Inference Algorithms Module
Provides backward compatibility for test imports.
"""

from ..engine.active_inference import (
    BeliefPropagation,
    GradientDescentInference,
    InferenceConfig,
    ParticleFilterInference,
    VariationalMessagePassing,
    create_inference_algorithm,
)

# Re-export variational message passing module
from . import variational_message_passing

__all__ = [
    "InferenceConfig",
    "VariationalMessagePassing",
    "BeliefPropagation",
    "GradientDescentInference",
    "ParticleFilterInference",
    "create_inference_algorithm",
    "variational_message_passing",
]
