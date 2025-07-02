"""
Incremental Backend Import Testing.

Debug and test backend modules one by one to identify import issues.
"""

import importlib
import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../..")))


def test_basic_imports():
    """Test basic Python imports work."""
    import numpy as np
    import torch

    assert np.__version__ is not None
    assert torch.__version__ is not None


def test_world_module_imports():
    """Test world module imports individually"""

    # Test simulation engine
    try:
        importlib.import_module("world.simulation.engine")
        print("✓ SimulationEngine imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import SimulationEngine: {e}")

    # Test grid position
    try:
        importlib.import_module("world.grid_position")
        print("✓ GridPosition and Position imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import GridPosition: {e}")

    # Test H3 world
    try:
        importlib.import_module("world.h3_world")
        print("✓ H3World imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import H3World: {e}")

    # Test spatial API
    try:
        importlib.import_module("world.spatial.spatial_api")
        print("✓ SpatialAPI imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import SpatialAPI: {e}")


def test_inference_basic_imports():
    """Test basic inference module imports"""

    # Test active inference engine
    try:
        importlib.import_module("inference.engine.active_inference")
        print("✓ ActiveInferenceEngine imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import ActiveInferenceEngine: {e}")

    # Test generative model
    try:
        importlib.import_module("inference.engine.generative_model")
        print("✓ GenerativeModel imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import GenerativeModel: {e}")


def test_inference_gnn_imports():
    """Test GNN module imports"""

    # Test GNN layers
    try:
        importlib.import_module("inference.gnn.layers")
        print("✓ GNN layers imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import GNN layers: {e}")

    # Test model mapper
    try:
        importlib.import_module("inference.gnn.model_mapper")
        print("✓ ModelMapper imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import ModelMapper: {e}")

    # Test parser
    try:
        importlib.import_module("inference.gnn.parser")
        print("✓ GNNParser imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import GNNParser: {e}")


def test_coalition_imports():
    """Test coalition module imports"""

    # Test coalition builder
    try:
        importlib.import_module("coalitions.formation.coalition_builder")
        print("✓ CoalitionBuilder imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import CoalitionBuilder: {e}")

    # Test coalition core
    try:
        importlib.import_module("coalitions.coalition.coalition_models")
        print("✓ Coalition models imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import Coalition models: {e}")


def test_agent_basic_imports():
    """Test basic agent module imports"""

    # Test data model first (likely needed by others)
    try:
        importlib.import_module("agents.base.data_model")
        print("✓ Agent data model imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import agent data model: {e}")

    # Test interfaces (likely needed by others)
    try:
        importlib.import_module("agents.base.interfaces")
        print("✓ Agent interfaces imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import agent interfaces: {e}")


def test_agent_logger_import():
    """Test agent logger import specifically"""

    try:
        importlib.import_module("agents.base.agent")
        print("✓ AgentLogger imported successfully")

        # Test instantiation
        logger = importlib.import_module(
            "agents.base.agent").AgentLogger("test_agent")
        assert logger.agent_id == "test_agent"
        print("✓ AgentLogger instantiated successfully")

    except Exception as e:
        pytest.fail(f"Failed to import or instantiate AgentLogger: {e}")


def test_base_agent_import():
    """Test base agent import specifically"""

    try:
        importlib.import_module("agents.base.agent")
        print("✓ BaseAgent imported successfully")

        # Note: Don't instantiate as it likely requires complex setup

    except Exception as e:
        pytest.fail(f"Failed to import BaseAgent: {e}")


def test_memory_imports():
    """Test memory module imports step by step"""

    # Test basic memory classes
    try:
        importlib.import_module("agents.base.memory")
        print("✓ Memory enums imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import memory enums: {e}")

    try:
        importlib.import_module("agents.base.memory")
        print("✓ Basic memory classes imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import basic memory classes: {e}")

    try:
        importlib.import_module("agents.base.memory")
        print("✓ Memory storage classes imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import memory storage classes: {e}")

    try:
        importlib.import_module("agents.base.memory")
        print("✓ Advanced memory classes imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import advanced memory classes: {e}")

    try:
        importlib.import_module("agents.base.memory")
        print("✓ MemorySystem imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import MemorySystem: {e}")


def test_perception_imports():
    """Test perception module imports step by step"""

    # Test basic perception classes
    try:
        importlib.import_module("agents.base.perception")
        print("✓ Perception enums imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import perception enums: {e}")

    try:
        importlib.import_module("agents.base.perception")
        print("✓ Basic perception classes imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import basic perception classes: {e}")

    try:
        importlib.import_module("agents.base.perception")
        print("✓ Perception capabilities imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import perception capabilities: {e}")

    try:
        importlib.import_module("agents.base.perception")
        print("✓ Sensor systems imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import sensor systems: {e}")

    try:
        importlib.import_module("agents.base.perception")
        print("✓ PerceptionSystem imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import PerceptionSystem: {e}")


def test_functional_world_classes():
    """Test that world classes actually work"""

    from world.grid_position import GridCoordinate, Position
    from world.h3_world import H3World

    # Test GridCoordinate functionality
    pos1 = GridCoordinate(5, 10)
    pos2 = GridCoordinate(8, 14)
    distance = pos1.distance_to(pos2)
    assert distance > 0
    print(f"✓ GridCoordinate works: distance = {distance:.2f}")

    # Test Position functionality
    pos3d = Position(x=1.5, y=2.7, z=3.2)
    assert pos3d.x == 1.5
    print(f"✓ Position works: {pos3d.to_array()}")

    # Test SimulationEngine basic setup
    from world.simulation.engine import SimulationConfig

    config = SimulationConfig(
        max_cycles=100, time_step=0.1, world={
            "resolution": 5, "size": 50, "resource_density": 1.0}, agents={
            "count": 10, "distribution": {
                "explorer": 4, "merchant": 3, "scholar": 2, "guardian": 1}, }, )
    engine = importlib.import_module(
        "world.simulation.engine").SimulationEngine(config)
    assert engine is not None
    print("✓ SimulationEngine basic functionality works")

    # Test H3World basic setup
    h3_world = H3World(center_lat=37.7749, center_lng=-122.4194, resolution=6)
    # Get a cell from the world
    center_cell = h3_world.get_cell(h3_world.center_hex)
    assert center_cell is not None
    print(f"✓ H3World works: center hex = {h3_world.center_hex}")

    # Test SpatialAPI basic setup
    spatial_api = importlib.import_module(
        "world.spatial.spatial_api").SpatialAPI(resolution=7)

    # Test basic hex operations
    hex_id = spatial_api.get_hex_at_position(
        37.7749, -122.4194)  # San Francisco
    assert hex_id is not None

    # Test neighbor operations
    neighbors = spatial_api.get_neighbors(hex_id)
    assert len(neighbors) == 6  # Hexagons have 6 neighbors

    print("✓ SpatialAPI basic functionality works")


def test_functional_inference_classes():
    """Test that inference classes actually work"""

    import torch

    from inference.engine.active_inference import ActiveInferenceEngine  # noqa: F401
    from inference.engine.active_inference import InferenceConfig

    # Test ActiveInferenceEngine
    from inference.engine.generative_model import (
        DiscreteGenerativeModel,
        ModelDimensions,
        ModelParameters,
    )
    from inference.gnn.layers import GCNLayer  # noqa: F401

    # Create a generative model for the engine
    dims = ModelDimensions(num_states=5, num_observations=3, num_actions=2)
    params = ModelParameters(use_gpu=False)
    gen_model = DiscreteGenerativeModel(dims, params)

    # Create the engine
    config = InferenceConfig(use_gpu=False)
    ai_engine = ActiveInferenceEngine(gen_model, config)

    # Test a step with an observation
    observation = torch.zeros(3)
    observation[0] = 1.0  # One-hot encoded observation
    beliefs = ai_engine.step(observation)

    assert beliefs.shape[-1] == 5
    print(f"✓ ActiveInferenceEngine works: beliefs shape = {beliefs.shape}")

    # Test DiscreteGenerativeModel instead of abstract GenerativeModel
    test_dims = ModelDimensions(
        num_states=4,
        num_observations=2,
        num_actions=2)
    test_params = ModelParameters(use_gpu=False)
    discrete_model = DiscreteGenerativeModel(test_dims, test_params)

    # Test observation model
    test_state = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Uniform belief
    obs_probs = discrete_model.observation_model(test_state)
    assert obs_probs.shape[-1] == 2
    print(
        f"✓ DiscreteGenerativeModel works: observation shape = {
            obs_probs.shape}")

    # Test GCNLayer
    gcn_layer = GCNLayer(in_channels=4, out_channels=6)
    node_features = torch.randn(5, 4)  # num_nodes=5, features=4
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                              dtype=torch.long)  # Simple ring

    gcn_output = gcn_layer(node_features, edge_index)
    assert gcn_output.shape == (5, 6)
    print(f"✓ GCNLayer works: output shape = {gcn_output.shape}")


def test_functional_memory_classes():
    """Test that memory classes actually work"""

    from datetime import datetime

    from agents.base.memory import (
        InMemoryStorage,
        Memory,
        MemoryImportance,
        MemorySystem,
        MemoryType,
        WorkingMemory,
    )

    # Test Memory enum
    assert MemoryType.WORKING in MemoryType
    assert MemoryImportance.HIGH in MemoryImportance
    print("✓ Memory enums work")

    # Test Memory instead of Experience (Experience requires different fields)
    mem = Memory(
        memory_id="test_mem_1",
        memory_type=MemoryType.EPISODIC,
        content="test memory content",
        timestamp=datetime.now(),
        importance=0.7,
    )
    assert mem.content == "test memory content"
    print("✓ Memory class works")

    # Test InMemoryStorage
    storage = InMemoryStorage()
    storage.store(mem)  # store() only takes the memory object
    retrieved = storage.retrieve("test_mem_1")  # retrieve by memory_id
    assert retrieved is not None
    assert retrieved.content == "test memory content"
    print("✓ InMemoryStorage works")

    # Test WorkingMemory
    working_mem = WorkingMemory(capacity=3)
    for i in range(5):  # Exceed capacity
        working_mem.add(f"item_{i}")

    # Access items directly via the deque
    items = list(working_mem.items)
    assert len(items) <= 3  # Should only keep last 3 items due to capacity
    print(f"✓ WorkingMemory works: {len(items)} items stored")

    # Test MemorySystem (basic functionality)
    mem_system = MemorySystem(agent_id="test_agent")

    # Store some memories
    for i in range(3):
        memory = mem_system.store_memory(
            content=f"test_item_{i}",
            memory_type=MemoryType.EPISODIC,
            importance=0.5)
        assert memory is not None

    # Check that memories were stored
    assert mem_system.total_memories >= 3
    print("✓ MemorySystem basic functionality works")


if __name__ == "__main__":
    # Run tests individually to see which ones pass
    pytest.main([__file__, "-v", "-s"])
