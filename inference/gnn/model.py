"""Base model class for GNN implementations."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class GMNModel:
    """Graph Machine Network model representation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the GMN model."""
        self.config = config
        self.architecture = config.get("architecture", "GCN")
        self.layers = config.get("layers", [])
        self.hyperparameters = config.get("hyperparameters", {})
        self.metadata = config.get("metadata", {})
        self._model = None
        self._device = None

    def build(self):
        """Build the model architecture."""
        # This would be implemented with actual PyTorch Geometric models
        logger.info(f"Building {self.architecture} model with {len(self.layers)} layers")

    def forward(self, x, edge_index):
        """Forward pass through the model."""
        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")
        # Actual implementation would use the PyTorch model
        return x

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "architecture": self.architecture,
            "layers": self.layers,
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata,
        }


def create_gnn_model(config: Dict[str, Any]) -> GMNModel:
    """Create a GNN model for test compatibility.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        GMNModel instance
    """
    logger.info(f"Creating GNN model with config: {config}")
    return GMNModel(config)
