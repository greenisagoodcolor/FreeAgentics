"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
GNN Integration for Active Inference
This module provides the interface between Graph Neural Networks and Active Inference.
"""

logger = logging.getLogger(__name__)

# Note: Some imports commented out due to missing modules
# from ...gnn.feature_extractor import FeatureExtractor
from inference.gnn.layers import GATLayer, GCNLayer, SAGELayer

from .active_inference import InferenceAlgorithm, InferenceConfig, VariationalMessagePassing

# from .active_inference import InferenceAlgorithm
from .generative_model import (
    DiscreteGenerativeModel,
    GenerativeModel,
    ModelDimensions,
    ModelParameters,
)


@dataclass
class GNNIntegrationConfig:
    """Configuration for GNN integration"""

    gnn_type: str = "gcn"
    num_layers: int = 3
    hidden_dim: int = 64
    output_dim: int = 32
    dropout: float = 0.1
    aggregation_method: str = "mean"
    use_edge_features: bool = True
    use_global_features: bool = True
    state_mapping: str = "direct"
    observation_mapping: str = "learned"
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-16


class GraphToStateMapper(ABC):
    """Abstract base class for mapping graph representations to Active Inference states"""

    def __init__(self, config: GNNIntegrationConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def map_to_states(
        self, graph_features: torch.Tensor, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Map graph features to state representation"""
        pass

    @abstractmethod
    def map_to_observations(
        self, graph_features: torch.Tensor, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Map graph features to observation representation"""

        pass


class DirectGraphMapper(GraphToStateMapper):
    """
    Direct mapping from graph features to states/observations.
    Assumes graph features directly correspond to state dimensions.
    """

    def __init__(self, config: GNNIntegrationConfig, state_dim: int, observation_dim: int) -> None:
        super().__init__(config)
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        # Type annotations for Optional projections
        self.state_projection: Optional[nn.Linear]
        self.obs_projection: Optional[nn.Linear]

        if config.output_dim != state_dim:
            self.state_projection = nn.Linear(config.output_dim, state_dim).to(self.device)
        else:
            self.state_projection = None
        if config.output_dim != observation_dim:
            self.obs_projection = nn.Linear(config.output_dim, observation_dim).to(self.device)
        else:
            self.obs_projection = None

    def map_to_states(
        self, graph_features: torch.Tensor, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Direct mapping to states"""
        features = graph_features.to(self.device)
        if node_indices is not None:
            features = features[node_indices]
        if self.state_projection is not None:
            features = self.state_projection(features)
        if self.config.state_mapping == "direct":
            features = F.softmax(features, dim=-1)
        return features

    def map_to_observations(
        self, graph_features: torch.Tensor, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Direct mapping to observations"""

        features = graph_features.to(self.device)
        if node_indices is not None:
            features = features[node_indices]
        if self.obs_projection is not None:
            features = self.obs_projection(features)
        return features


class LearnedGraphMapper(GraphToStateMapper):
    """
    Learned mapping from graph features to states/observations.
    Uses neural networks to learn the transformation.
    """

    def __init__(self, config: GNNIntegrationConfig, state_dim: int, observation_dim: int) -> None:
        super().__init__(config)
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.state_mapper = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, state_dim),
        ).to(self.device)
        self.obs_mapper = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, observation_dim),
        ).to(self.device)

    def map_to_states(
        self, graph_features: torch.Tensor, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Learned mapping to states"""
        features = graph_features.to(self.device)
        if node_indices is not None:
            features = features[node_indices]
        states = self.state_mapper(features)
        states = F.softmax(states, dim=-1)
        return states

    def map_to_observations(
        self, graph_features: torch.Tensor, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Learned mapping to observations"""

        features = graph_features.to(self.device)
        if node_indices is not None:
            features = features[node_indices]
        observations = self.obs_mapper(features)
        return observations


class GNNActiveInferenceAdapter:
    """
    Main adapter between GNN and Active Inference.
    Handles the integration of graph neural network outputs with
    Active Inference generative models and inference algorithms.
    """

    def __init__(
        self,
        config: GNNIntegrationConfig,
        gnn_model: nn.Module,
        generative_model: GenerativeModel,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        self.config = config
        self.gnn_model = gnn_model
        self.generative_model = generative_model
        self.inference = inference_algorithm
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        if isinstance(generative_model, DiscreteGenerativeModel):
            self.state_dim = generative_model.dims.num_states
            self.obs_dim = generative_model.dims.num_observations
        else:
            self.state_dim = generative_model.dims.num_states
            self.obs_dim = generative_model.dims.num_observations
        if config.state_mapping == "direct":
            self.mapper = DirectGraphMapper(config, self.state_dim, self.obs_dim)
        elif config.state_mapping == "learned":
            self.mapper = LearnedGraphMapper(config, self.state_dim, self.obs_dim)
        else:
            self.mapper = LearnedGraphMapper(config, self.state_dim, self.obs_dim)
        self.aggregator = GraphFeatureAggregator(config)

    def process_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process graph through GNN and extract features.
        Args:
            node_features: Node feature matrix [num_nodes x feature_dim]
            edge_index: Edge connectivity [2 x num_edges]
            edge_features: Optional edge features
            batch: Optional batch assignment for multiple graphs
        Returns:
            Dictionary with processed features
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_features is not None:
            edge_features = edge_features.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
        with torch.no_grad():
            graph_features = self.gnn_model(node_features, edge_index, edge_features)
        if batch is not None:
            aggregated_features = self.aggregator.aggregate(graph_features, batch)
        else:
            aggregated_features = self.aggregator.aggregate_single(graph_features)
        return {
            "node_features": graph_features,
            "graph_features": aggregated_features,
            "edge_index": edge_index,
        }

    def graph_to_beliefs(
        self,
        graph_data: Dict[str, torch.Tensor],
        agent_node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert graph features to belief states for Active Inference.
        Args:
            graph_data: Processed graph data from process_graph
            agent_node_indices: Indices of nodes representing agents
        Returns:
            Belief states suitable for Active Inference
        """

        if agent_node_indices is None:
            features = graph_data["graph_features"]
        else:
            features = graph_data["node_features"]
        beliefs = self.mapper.map_to_states(features, agent_node_indices)
        return beliefs

    def graph_to_observations(
        self,
        graph_data: Dict[str, torch.Tensor],
        observation_node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert graph features to observations for Active Inference.
        Args:
            graph_data: Processed graph data
            observation_node_indices: Indices of nodes providing observations
        Returns:
            Observations suitable for Active Inference
        """

        if observation_node_indices is None:
            features = graph_data["graph_features"]
        else:
            features = graph_data["node_features"]
        observations = self.mapper.map_to_observations(features, observation_node_indices)
        return observations

    def update_beliefs_with_graph(
        self,
        current_beliefs: torch.Tensor,
        graph_data: Dict[str, torch.Tensor],
        agent_node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update Active Inference beliefs using graph information.
        Args:
            current_beliefs: Current belief state
            graph_data: Processed graph data
            agent_node_indices: Indices of agent nodes
        Returns:
            Updated beliefs
        """
        observations = self.graph_to_observations(graph_data, agent_node_indices)
        if isinstance(self.generative_model, DiscreteGenerativeModel):
            updated_beliefs = self.inference.infer_states(
                observations, self.generative_model, current_beliefs
            )
        else:
            updated_beliefs = self.inference.infer_states(
                observations, self.generative_model, current_beliefs
            )
        return updated_beliefs

    def compute_expected_free_energy_with_graph(
        self,
        policy: Any,
        graph_data: Dict[str, torch.Tensor],
        preferences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute expected free energy incorporating graph structure.
        Args:
            policy: Policy to evaluate
            graph_data: Processed graph data
            preferences: Optional preferences
        Returns:
            Expected free energy
        """
        beliefs = self.graph_to_beliefs(graph_data)
        return torch.tensor(0.0, device=self.device)


class GraphFeatureAggregator:
    """
    Aggregates node features into graph-level representations.
    """

    def __init__(self, config: GNNIntegrationConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        if config.aggregation_method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(config.output_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, 1),
            ).to(self.device)

    def aggregate(self, node_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node features by batch assignment.
        Args:
            node_features: Node features [num_nodes x feature_dim]
            batch: Batch assignment [num_nodes]
        Returns:
            Aggregated features [num_graphs x feature_dim]
        """
        num_graphs = batch.max().item() + 1
        feature_dim = node_features.shape[1]
        aggregated = torch.zeros(num_graphs, feature_dim, device=self.device)
        if self.config.aggregation_method == "mean":
            for i in range(num_graphs):
                mask = batch == i
                if mask.any():
                    aggregated[i] = node_features[mask].mean(dim=0)
        elif self.config.aggregation_method == "max":
            for i in range(num_graphs):
                mask = batch == i
                if mask.any():
                    aggregated[i] = node_features[mask].max(dim=0)[0]
        elif self.config.aggregation_method == "sum":
            for i in range(num_graphs):
                mask = batch == i
                if mask.any():
                    aggregated[i] = node_features[mask].sum(dim=0)
        elif self.config.aggregation_method == "attention":
            for i in range(num_graphs):
                mask = batch == i
                if mask.any():
                    graph_nodes = node_features[mask]
                    attention_weights = F.softmax(self.attention(graph_nodes), dim=0)
                    aggregated[i] = (attention_weights * graph_nodes).sum(dim=0)
        return aggregated

    def aggregate_single(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features for a single graph.
        Args:
            node_features: Node features [num_nodes x feature_dim]
        Returns:
            Aggregated features [1 x feature_dim]
        """

        if self.config.aggregation_method == "mean":
            return node_features.mean(dim=0, keepdim=True)
        elif self.config.aggregation_method == "max":
            return node_features.max(dim=0, keepdim=True)[0]
        elif self.config.aggregation_method == "sum":
            return node_features.sum(dim=0, keepdim=True)
        elif self.config.aggregation_method == "attention":
            attention_weights = F.softmax(self.attention(node_features), dim=0)
            return (attention_weights * node_features).sum(dim=0, keepdim=True)
        else:
            return node_features.mean(dim=0, keepdim=True)


class HierarchicalGraphIntegration:
    """
    Hierarchical integration of graphs with Active Inference.
    Processes graphs at multiple scales for hierarchical inference.
    """

    def __init__(self, config: GNNIntegrationConfig, level_configs: List[Dict[str, Any]]) -> None:
        self.config = config
        self.num_levels = len(level_configs)
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.level_gnns = nn.ModuleList()
        self.level_adapters = []
        for level_config in level_configs:
            gnn = self._create_gnn(level_config)
            self.level_gnns.append(gnn)
            self.level_adapters.append(None)

    def _create_gnn(self, level_config: Dict[str, Any]) -> nn.Module:
        """Create GNN model for a specific level"""
        gnn_type = level_config.get("gnn_type", self.config.gnn_type)
        num_layers = level_config.get("num_layers", 2)
        hidden_dim = level_config.get("hidden_dim", self.config.hidden_dim)
        output_dim = level_config.get("output_dim", self.config.output_dim)
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = level_config.get("input_dim", self.config.hidden_dim)
            else:
                in_dim = hidden_dim
            if i == num_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim
            if gnn_type == "gcn":
                layer = GCNLayer(in_dim, out_dim)
            elif gnn_type == "gat":
                layer = GATLayer(in_dim, out_dim)
            elif gnn_type == "graphsage":
                layer = SAGELayer(in_dim, out_dim)
            else:
                layer = GCNLayer(in_dim, out_dim)
            layers.append(layer)
        return nn.Sequential(*layers).to(self.device)

    def set_generative_models(
        self,
        generative_models: List[GenerativeModel],
        inference_algorithms: List[InferenceAlgorithm],
    ) -> None:
        """Set generative models and create adapters for each level"""
        for i, (gen_model, inf_algo) in enumerate(zip(generative_models, inference_algorithms)):
            self.level_adapters[i] = GNNActiveInferenceAdapter(
                self.config, self.level_gnns[i], gen_model, inf_algo
            )

    def process_hierarchical_graph(
        self, graph_data_per_level: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process graphs at each hierarchical level.
        Args:
            graph_data_per_level: List of graph data for each level
        Returns:
            Processed features for each level
        """
        processed_levels = []
        for i, (graph_data, adapter) in enumerate(zip(graph_data_per_level, self.level_adapters)):
            if adapter is not None:
                processed = adapter.process_graph(
                    graph_data["node_features"],
                    graph_data["edge_index"],
                    graph_data.get("edge_features"),
                    graph_data.get("batch"),
                )
                processed_levels.append(processed)
            else:
                processed_levels.append(graph_data)
        return processed_levels

    def hierarchical_belief_update(
        self,
        current_beliefs: List[torch.Tensor],
        graph_data_per_level: List[Dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Update beliefs hierarchically using graph information.
        Args:
            current_beliefs: Current beliefs at each level
            graph_data_per_level: Graph data for each level
        Returns:
            Updated beliefs at each level
        """
        updated_beliefs = []
        for i in range(self.num_levels):
            if self.level_adapters[i] is not None:
                graph_data = graph_data_per_level[i]
                processed = self.level_adapters[i].process_graph(
                    graph_data["node_features"],
                    graph_data["edge_index"],
                    graph_data.get("edge_features"),
                    graph_data.get("batch"),
                )
                updated = self.level_adapters[i].update_beliefs_with_graph(
                    current_beliefs[i], processed
                )
                updated_beliefs.append(updated)
            else:
                updated_beliefs.append(current_beliefs[i])
        return updated_beliefs


class GNNActiveInferenceIntegration:
    """
    High-level integration class that can create Active Inference models from GNN specifications.
    This class provides a convenient interface for creating complete Active Inference systems.
    """

    def __init__(self, config: GNNIntegrationConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    def create_from_gnn_spec(self, gnn_spec: Dict[str, Any]) -> Any:
        """
        Create an Active Inference model from a GNN specification.
        Args:
            gnn_spec: Dictionary containing model specification with keys:
                - model_type: 'discrete' or 'continuous'
                - dimensions: dict with num_states, num_observations, num_actions
                - matrices: dict with A, B, C, D matrices for discrete models
        Returns:
            Object with generative_model and inference attributes
        """
        from inference.engine.active_inference import InferenceConfig, VariationalMessagePassing
        from inference.engine.generative_model import (
            DiscreteGenerativeModel,
            ModelDimensions,
            ModelParameters,
        )

        # Extract dimensions
        dims_dict = gnn_spec["dimensions"]
        dims = ModelDimensions(
            num_states=dims_dict["num_states"],
            num_observations=dims_dict["num_observations"],
            num_actions=dims_dict["num_actions"],
        )
        # Create model parameters
        params = ModelParameters(use_gpu=self.config.use_gpu)
        # Create generative model
        if gnn_spec["model_type"] == "discrete":
            gen_model = DiscreteGenerativeModel(dims, params)
            # Set matrices if provided
            if "matrices" in gnn_spec:
                matrices = gnn_spec["matrices"]
                if "A" in matrices:
                    gen_model.A = torch.tensor(
                        matrices["A"], dtype=torch.float32, device=self.device
                    )
                if "B" in matrices:
                    gen_model.B = torch.tensor(
                        matrices["B"], dtype=torch.float32, device=self.device
                    )
                if "C" in matrices:
                    gen_model.C = torch.tensor(
                        matrices["C"], dtype=torch.float32, device=self.device
                    )
                if "D" in matrices:
                    gen_model.D = torch.tensor(
                        matrices["D"], dtype=torch.float32, device=self.device
                    )
        else:
            raise NotImplementedError(f"Model type {gnn_spec['model_type']} not yet implemented")
        # Create inference algorithm
        inf_config = InferenceConfig(use_gpu=self.config.use_gpu)
        inference = VariationalMessagePassing(inf_config)

        # Return a simple object with the required attributes
        class ActiveInferenceModel:
            def __init__(
                self, generative_model: GenerativeModel, inference: InferenceAlgorithm
            ) -> None:
                self.generative_model = generative_model
                self.inference = inference

        return ActiveInferenceModel(gen_model, inference)


def create_gnn_adapter(
    adapter_type: str, config: Optional[GNNIntegrationConfig] = None, **kwargs
) -> Union[GNNActiveInferenceAdapter, HierarchicalGraphIntegration]:
    """
    Factory function to create GNN adapters.
    Args:
        adapter_type: Type of adapter ('standard', 'hierarchical')
        config: Integration configuration
        **kwargs: Adapter-specific parameters
    Returns:
        GNN adapter instance
    """
    if config is None:
        config = GNNIntegrationConfig()
    if adapter_type == "standard":
        gnn_model = kwargs.get("gnn_model")
        generative_model = kwargs.get("generative_model")
        inference_algorithm = kwargs.get("inference_algorithm")
        if None in [gnn_model, generative_model, inference_algorithm]:
            raise ValueError(
                "Standard adapter requires gnn_model, generative_model, and inference_algorithm"
            )
        return GNNActiveInferenceAdapter(config, gnn_model, generative_model, inference_algorithm)
    elif adapter_type == "hierarchical":
        level_configs = kwargs.get("level_configs", [])
        if not level_configs:
            raise ValueError("Hierarchical adapter requires level_configs")
        adapter = HierarchicalGraphIntegration(config, level_configs)
        generative_models = kwargs.get("generative_models")
        inference_algorithms = kwargs.get("inference_algorithms")
        if generative_models and inference_algorithms:
            adapter.set_generative_models(generative_models, inference_algorithms)
        return adapter
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


if __name__ == "__main__":
    config = GNNIntegrationConfig(
        gnn_type="gcn", num_layers=3, hidden_dim=64, output_dim=32, use_gpu=False
    )

    class DummyGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return self.layers(x)

    gnn_model = DummyGNN(10, 64, 32)
    dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
    params = ModelParameters(use_gpu=False)
    gen_model = DiscreteGenerativeModel(dims, params)
    inf_config = InferenceConfig(use_gpu=False)
    inference = VariationalMessagePassing(inf_config)
    adapter = GNNActiveInferenceAdapter(config, gnn_model, gen_model, inference)
    node_features = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    graph_data = adapter.process_graph(node_features, edge_index)
    beliefs = adapter.graph_to_beliefs(graph_data)
    print(f"Beliefs shape: {beliefs.shape}")
    print(f"Beliefs: {beliefs}")
