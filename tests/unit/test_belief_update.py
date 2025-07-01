"""
Tests for belief update algorithms.

This module tests belief update functionality with graceful degradation
for PyTorch dependencies.
"""

import numpy as np
import pytest

# Graceful degradation for PyTorch imports
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TORCH_AVAILABLE = False
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)

# Import FreeAgentics modules with graceful degradation
try:
    from inference.engine.belief_update import BeliefUpdateC, TemporalBeliefUpdate
    BELIEF_MODULES_AVAILABLE = True
except ImportError as e:
    BELIEF_MODULES_AVAILABLE = False
    pytest.skip(f"Belief modules not available: {e}", allow_module_level=True)

from typing import Tuple
import torch.nn as nn

from inference.engine.belief_update import (
    AttentionGraphBeliefUpdater,
    BeliefUpdateConfig,
    DirectBeliefUpdater,
    DirectGraphObservationModel,
    GraphNNBeliefUpdater,
    HierarchicalBeliefUpdater,
    LearnedGraphObservationModel,
    create_belief_updater,
)
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)


class TestBeliefUpdateConfig:
    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = BeliefUpdateConfig()
        assert config.update_method == "variational"
        assert config.learning_rate == 0.01
        assert config.num_iterations == 10
        assert config.use_gpu

    def test_custom_config(self) -> None:
        """Test custom configuration"""
        config = BeliefUpdateConfig(
            update_method="gradient",
            learning_rate=0.1,
            use_gpu=False,
        )
        assert config.update_method == "gradient"
        assert config.learning_rate == 0.1
        assert not config.use_gpu


class TestDirectGraphObservationModel:
    def test_initialization(self) -> None:
        """Test model initialization"""
        config = BeliefUpdateConfig(use_gpu=False)
        model = DirectGraphObservationModel(config)
        assert model.config == config

    def test_forward(self) -> None:
        """Test forward pass"""
        config = BeliefUpdateConfig(use_gpu=False)
        model = DirectGraphObservationModel(config)
        graph_features = torch.randn(3, 8)
        output = model.forward(graph_features)
        assert output.shape == (3, 8)
        assert torch.equal(output, graph_features)


class TestLearnedGraphObservationModel:
    def test_initialization(self) -> None:
        """Test learned model initialization"""
        config = BeliefUpdateConfig(use_gpu=False)
        model = LearnedGraphObservationModel(config, input_dim=8, output_dim=4)
        assert model.config == config
        assert isinstance(model.network, nn.Sequential)

    def test_forward(self) -> None:
        """Test learned forward pass"""
        config = BeliefUpdateConfig(use_gpu=False)
        model = LearnedGraphObservationModel(config, input_dim=8, output_dim=4)
        graph_features = torch.randn(3, 8)
        output = model.forward(graph_features)
        assert output.shape == (3, 4)


class TestGraphNNBeliefUpdater:
    @pytest.fixture
    def setup_updater(
        self,
    ) -> Tuple[GraphNNBeliefUpdater, BeliefUpdateConfig, DiscreteGenerativeModel]:
        """Setup belief updater with components"""
        config = BeliefUpdateConfig(use_gpu=False)
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)
        updater = GraphNNBeliefUpdater(config)
        return updater, config, gen_model

    def test_initialization(
        self,
        setup_updater: Tuple[GraphNNBeliefUpdater, BeliefUpdateConfig, DiscreteGenerativeModel],
    ) -> None:
        """Test updater initialization"""
        updater, config, gen_model = setup_updater
        assert updater.config == config
        assert updater.device is not None

    def test_update_beliefs(
        self,
        setup_updater: Tuple[GraphNNBeliefUpdater, BeliefUpdateConfig, DiscreteGenerativeModel],
    ) -> None:
        """Test belief update"""
        updater, config, gen_model = setup_updater
        current_beliefs = torch.softmax(torch.randn(2, 4), dim=-1)
        observations = torch.randn(2, 3)
        updated_beliefs = updater.update_beliefs(current_beliefs, observations, gen_model)
        assert updated_beliefs.shape == (2, 4)


class TestAttentionGraphBeliefUpdater:
    @pytest.fixture
    def setup_attention_updater(self) -> AttentionGraphBeliefUpdater:
        """Setup attention-based updater"""
        config = BeliefUpdateConfig(use_gpu=False)
        updater = AttentionGraphBeliefUpdater(config)
        return updater

    def test_initialization(self, setup_attention_updater: AttentionGraphBeliefUpdater) -> None:
        """Test attention updater initialization"""
        updater = setup_attention_updater
        assert updater.config is not None
        assert updater.device is not None

    def test_update_beliefs(self, setup_attention_updater: AttentionGraphBeliefUpdater) -> None:
        """Test belief update with attention"""
        updater = setup_attention_updater
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)
        current_beliefs = torch.softmax(torch.randn(2, 4), dim=-1)
        observations = torch.randn(2, 3)
        updated_beliefs = updater.update_beliefs(current_beliefs, observations, gen_model)
        assert updated_beliefs.shape == (2, 4)


class TestHierarchicalBeliefUpdater:
    @pytest.fixture
    def setup_hierarchical_updater(self) -> HierarchicalBeliefUpdater:
        """Setup hierarchical updater"""
        config = BeliefUpdateConfig(use_gpu=False)
        updater = HierarchicalBeliefUpdater(config)
        return updater

    def test_initialization(self, setup_hierarchical_updater: HierarchicalBeliefUpdater) -> None:
        """Test hierarchical updater initialization"""
        updater = setup_hierarchical_updater
        assert updater.config is not None
        assert updater.device is not None

    def test_update_beliefs(self, setup_hierarchical_updater: HierarchicalBeliefUpdater) -> None:
        """Test hierarchical belief update"""
        updater = setup_hierarchical_updater
        dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
        params = ModelParameters(use_gpu=False)
        gen_model = DiscreteGenerativeModel(dims, params)
        current_beliefs = torch.softmax(torch.randn(2, 4), dim=-1)
        observations = torch.randn(2, 3)
        updated_beliefs = updater.update_beliefs(current_beliefs, observations, gen_model)
        assert updated_beliefs.shape == (2, 4)


class TestFactoryFunction:
    def test_create_graphnn_updater(self) -> None:
        """Test creating GraphNN updater"""
        config = BeliefUpdateConfig(use_gpu=False)
        updater = create_belief_updater("graphnn", config)
        assert isinstance(updater, GraphNNBeliefUpdater)

    def test_create_attention_updater(self) -> None:
        """Test creating attention updater"""
        config = BeliefUpdateConfig(use_gpu=False)
        updater = create_belief_updater("attention", config)
        assert isinstance(updater, AttentionGraphBeliefUpdater)

    def test_create_hierarchical_updater(self) -> None:
        """Test creating hierarchical updater"""
        config = BeliefUpdateConfig(use_gpu=False)
        updater = create_belief_updater("hierarchical", config)
        assert isinstance(updater, HierarchicalBeliefUpdater)

    def test_default_updater(self) -> None:
        """Test default updater creation"""
        config = BeliefUpdateConfig(use_gpu=False)
        updater = create_belief_updater("unknown", config)
        assert isinstance(updater, DirectBeliefUpdater)
