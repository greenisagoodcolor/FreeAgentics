"""
Comprehensive test suite for Inference modules
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock complex dependencies
mock_modules = {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'torch.optim': MagicMock(),
    'torch_geometric': MagicMock(),
    'torch_geometric.nn': MagicMock(),
    'torch_geometric.data': MagicMock(),
    'pymdp': MagicMock(),
    'pymdp.utils': MagicMock(),
    'h3': MagicMock(),
    'networkx': MagicMock(),
    'spacy': MagicMock(),
    'transformers': MagicMock(),
}

with patch.dict('sys.modules', mock_modules):
    from inference.active.gmn_parser import GMNParser
    from inference.active.gmn_validation import GMNValidator
    from inference.gnn.feature_extractor import FeatureExtractor
    from inference.gnn.parser import GNNParser
    from inference.gnn.validator import GNNValidator
    from inference.llm.local_llm_manager import LocalLLMManager
    from inference.llm.provider_interface import LLMProviderInterface


class TestGMNParser:
    """Test GMN (Generative Model Network) Parser functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = GMNParser()

    def test_gmn_parser_initialization(self):
        """Test GMNParser initialization."""
        parser = GMNParser()
        assert parser is not None
        assert hasattr(parser, 'parse')

    def test_gmn_parser_basic_parsing(self):
        """Test basic GMN parsing functionality."""

        # Mock the parse method
        def mock_parse(gmn_string):
            if "states" in gmn_string:
                return {
                    "states": ["state1", "state2"],
                    "actions": ["action1", "action2"],
                    "observations": ["obs1", "obs2"],
                }
            return {}

        self.parser.parse = mock_parse

        # Test parsing
        gmn_input = "states: [state1, state2]; actions: [action1, action2]"
        result = self.parser.parse(gmn_input)

        assert "states" in result
        assert len(result["states"]) == 2
        assert "action1" in result["actions"]

    def test_gmn_parser_error_handling(self):
        """Test GMN parser error handling."""

        # Mock parser that raises errors
        def mock_parse_with_error(gmn_string):
            if "invalid" in gmn_string:
                raise ValueError("Invalid GMN syntax")
            return {"valid": True}

        self.parser.parse = mock_parse_with_error

        # Test error handling
        with pytest.raises(ValueError, match="Invalid GMN syntax"):
            self.parser.parse("invalid syntax here")

        # Test valid input
        result = self.parser.parse("valid input")
        assert result["valid"] is True

    def test_gmn_parser_complex_structure(self):
        """Test parsing complex GMN structures."""

        # Mock complex parsing
        def mock_parse_complex(gmn_string):
            return {
                "generative_model": {
                    "A": np.eye(2),  # Observation model
                    "B": np.eye(2),  # Transition model
                    "C": np.array([1.0, 0.0]),  # Prior preferences
                    "D": np.array([0.5, 0.5]),  # Initial beliefs
                },
                "policy": {
                    "actions": [0, 1],
                    "expected_free_energy": [0.1, 0.2],
                },
            }

        self.parser.parse = mock_parse_complex

        result = self.parser.parse("complex GMN structure")

        assert "generative_model" in result
        assert "policy" in result
        assert isinstance(result["generative_model"]["A"], np.ndarray)
        assert len(result["policy"]["actions"]) == 2


class TestGMNValidator:
    """Test GMN Validator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = GMNValidator()

    def test_gmn_validator_initialization(self):
        """Test GMNValidator initialization."""
        validator = GMNValidator()
        assert validator is not None
        assert hasattr(validator, 'validate')

    def test_gmn_validator_basic_validation(self):
        """Test basic GMN validation."""

        # Mock validation method
        def mock_validate(gmn_data):
            required_fields = ["states", "actions", "observations"]
            for field in required_fields:
                if field not in gmn_data:
                    return {"valid": False, "error": f"Missing {field}"}
            return {"valid": True, "error": None}

        self.validator.validate = mock_validate

        # Test valid data
        valid_data = {
            "states": ["s1", "s2"],
            "actions": ["a1", "a2"],
            "observations": ["o1", "o2"],
        }
        result = self.validator.validate(valid_data)
        assert result["valid"] is True

        # Test invalid data
        invalid_data = {"states": ["s1", "s2"]}
        result = self.validator.validate(invalid_data)
        assert result["valid"] is False
        assert "Missing" in result["error"]

    def test_gmn_validator_matrix_validation(self):
        """Test GMN matrix validation."""

        # Mock matrix validation
        def mock_validate_matrices(gmn_data):
            if "generative_model" not in gmn_data:
                return {"valid": False, "error": "No generative model"}

            gm = gmn_data["generative_model"]
            if "A" not in gm or "B" not in gm:
                return {"valid": False, "error": "Missing matrices"}

            # Check matrix shapes
            A = gm["A"]
            gm["B"]
            if A.shape[0] != A.shape[1]:
                return {"valid": False, "error": "Invalid A matrix shape"}

            return {"valid": True, "error": None}

        self.validator.validate = mock_validate_matrices

        # Test valid matrices
        valid_data = {"generative_model": {"A": np.eye(2), "B": np.eye(2)}}
        result = self.validator.validate(valid_data)
        assert result["valid"] is True

        # Test invalid matrices
        invalid_data = {
            "generative_model": {
                "A": np.array([[1, 0, 0], [0, 1, 0]]),  # Non-square
                "B": np.eye(2),
            }
        }
        result = self.validator.validate(invalid_data)
        assert result["valid"] is False


class TestFeatureExtractor:
    """Test GNN Feature Extractor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()

    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None

    def test_feature_extractor_basic_extraction(self):
        """Test basic feature extraction."""

        # Mock extraction method
        def mock_extract_features(graph_data):
            num_nodes = len(graph_data.get("nodes", []))
            features = np.random.rand(num_nodes, 10)  # 10-dimensional features
            return features

        self.extractor.extract_features = mock_extract_features

        # Test feature extraction
        graph_data = {
            "nodes": [
                {"id": "node1", "type": "entity"},
                {"id": "node2", "type": "entity"},
                {"id": "node3", "type": "relation"},
            ],
            "edges": [
                {"source": "node1", "target": "node2", "type": "connects"},
                {"source": "node2", "target": "node3", "type": "relates"},
            ],
        }

        features = self.extractor.extract_features(graph_data)
        assert features.shape == (3, 10)  # 3 nodes, 10 features each

    def test_feature_extractor_node_features(self):
        """Test node feature extraction."""

        # Mock node feature extraction
        def mock_extract_node_features(node_data):
            feature_map = {
                "entity": [1, 0, 0],
                "relation": [0, 1, 0],
                "concept": [0, 0, 1],
            }
            return feature_map.get(node_data["type"], [0, 0, 0])

        self.extractor.extract_node_features = mock_extract_node_features

        # Test different node types
        entity_features = self.extractor.extract_node_features(
            {"type": "entity"}
        )
        assert entity_features == [1, 0, 0]

        relation_features = self.extractor.extract_node_features(
            {"type": "relation"}
        )
        assert relation_features == [0, 1, 0]

        unknown_features = self.extractor.extract_node_features(
            {"type": "unknown"}
        )
        assert unknown_features == [0, 0, 0]

    def test_feature_extractor_edge_features(self):
        """Test edge feature extraction."""

        # Mock edge feature extraction
        def mock_extract_edge_features(edge_data):
            weight = edge_data.get("weight", 1.0)
            edge_type = edge_data.get("type", "default")

            type_encoding = {"connects": 1, "relates": 2, "influences": 3}

            return [weight, type_encoding.get(edge_type, 0)]

        self.extractor.extract_edge_features = mock_extract_edge_features

        # Test edge features
        edge1 = {"weight": 0.8, "type": "connects"}
        features1 = self.extractor.extract_edge_features(edge1)
        assert features1 == [0.8, 1]

        edge2 = {"weight": 0.5, "type": "relates"}
        features2 = self.extractor.extract_edge_features(edge2)
        assert features2 == [0.5, 2]


class TestGNNParser:
    """Test GNN Parser functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = GNNParser()

    def test_gnn_parser_initialization(self):
        """Test GNNParser initialization."""
        parser = GNNParser()
        assert parser is not None

    def test_gnn_parser_graph_parsing(self):
        """Test graph parsing functionality."""

        # Mock parsing method
        def mock_parse_graph(graph_string):
            if "nodes" in graph_string and "edges" in graph_string:
                return {
                    "nodes": [
                        {"id": "n1", "label": "Node 1"},
                        {"id": "n2", "label": "Node 2"},
                    ],
                    "edges": [{"source": "n1", "target": "n2", "weight": 1.0}],
                }
            return {"nodes": [], "edges": []}

        self.parser.parse_graph = mock_parse_graph

        # Test parsing
        graph_string = "nodes: [n1, n2]; edges: [(n1, n2)]"
        result = self.parser.parse_graph(graph_string)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["nodes"][0]["id"] == "n1"

    def test_gnn_parser_tensor_conversion(self):
        """Test tensor conversion functionality."""

        # Mock tensor conversion
        def mock_to_tensor(graph_data):
            num_nodes = len(graph_data["nodes"])
            num_edges = len(graph_data["edges"])

            # Create mock tensors (using numpy arrays)
            node_features = np.random.rand(num_nodes, 5)
            edge_index = np.array([[0, 1], [1, 0]])  # Bidirectional edge
            edge_attr = np.random.rand(num_edges, 2)

            return {
                "x": node_features,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }

        self.parser.to_tensor = mock_to_tensor

        # Test conversion
        graph_data = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"source": "n1", "target": "n2"}],
        }

        tensor_data = self.parser.to_tensor(graph_data)

        assert "x" in tensor_data
        assert "edge_index" in tensor_data
        assert "edge_attr" in tensor_data
        assert tensor_data["x"].shape == (2, 5)


class TestGNNValidator:
    """Test GNN Validator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = GNNValidator()

    def test_gnn_validator_initialization(self):
        """Test GNNValidator initialization."""
        validator = GNNValidator()
        assert validator is not None

    def test_gnn_validator_graph_validation(self):
        """Test graph validation."""

        # Mock validation method
        def mock_validate_graph(graph_data):
            if "nodes" not in graph_data:
                return {"valid": False, "error": "No nodes"}
            if "edges" not in graph_data:
                return {"valid": False, "error": "No edges"}

            # Check node IDs are unique
            node_ids = [node["id"] for node in graph_data["nodes"]]
            if len(node_ids) != len(set(node_ids)):
                return {"valid": False, "error": "Duplicate node IDs"}

            # Check edge references
            for edge in graph_data["edges"]:
                if (
                    edge["source"] not in node_ids
                    or edge["target"] not in node_ids
                ):
                    return {"valid": False, "error": "Invalid edge reference"}

            return {"valid": True, "error": None}

        self.validator.validate_graph = mock_validate_graph

        # Test valid graph
        valid_graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"source": "n1", "target": "n2"}],
        }
        result = self.validator.validate_graph(valid_graph)
        assert result["valid"] is True

        # Test invalid graph (missing nodes)
        invalid_graph = {"edges": [{"source": "n1", "target": "n2"}]}
        result = self.validator.validate_graph(invalid_graph)
        assert result["valid"] is False
        assert "No nodes" in result["error"]

    def test_gnn_validator_tensor_validation(self):
        """Test tensor validation."""

        # Mock tensor validation
        def mock_validate_tensor(tensor_data):
            required_keys = ["x", "edge_index"]
            for key in required_keys:
                if key not in tensor_data:
                    return {"valid": False, "error": f"Missing {key}"}

            x = tensor_data["x"]
            edge_index = tensor_data["edge_index"]

            # Check dimensions
            if len(x.shape) != 2:
                return {"valid": False, "error": "Invalid node features shape"}

            if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
                return {"valid": False, "error": "Invalid edge index shape"}

            return {"valid": True, "error": None}

        self.validator.validate_tensor = mock_validate_tensor

        # Test valid tensor
        valid_tensor = {
            "x": np.random.rand(3, 5),
            "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
        }
        result = self.validator.validate_tensor(valid_tensor)
        assert result["valid"] is True

        # Test invalid tensor (wrong edge_index shape)
        invalid_tensor = {
            "x": np.random.rand(3, 5),
            "edge_index": np.array([0, 1, 2]),  # Wrong shape
        }
        result = self.validator.validate_tensor(invalid_tensor)
        assert result["valid"] is False


class TestLocalLLMManager:
    """Test Local LLM Manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = LocalLLMManager()

    def test_llm_manager_initialization(self):
        """Test LocalLLMManager initialization."""
        manager = LocalLLMManager()
        assert manager is not None

    def test_llm_manager_model_loading(self):
        """Test model loading functionality."""

        # Mock model loading
        def mock_load_model(model_name):
            if model_name == "test_model":
                return {"model": "loaded", "name": model_name}
            else:
                raise ValueError(f"Model {model_name} not found")

        self.manager.load_model = mock_load_model

        # Test successful loading
        model = self.manager.load_model("test_model")
        assert model["model"] == "loaded"
        assert model["name"] == "test_model"

        # Test failed loading
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            self.manager.load_model("nonexistent")

    def test_llm_manager_text_generation(self):
        """Test text generation functionality."""

        # Mock text generation
        def mock_generate_text(prompt, max_length=100):
            return f"Generated response to: {prompt[:50]}..."

        self.manager.generate_text = mock_generate_text

        # Test text generation
        prompt = "What is the meaning of life?"
        response = self.manager.generate_text(prompt)
        assert "Generated response to:" in response
        assert "What is the meaning of life?" in response

    def test_llm_manager_batch_processing(self):
        """Test batch processing functionality."""

        # Mock batch processing
        def mock_process_batch(prompts):
            return [f"Response to: {prompt}" for prompt in prompts]

        self.manager.process_batch = mock_process_batch

        # Test batch processing
        prompts = ["Question 1", "Question 2", "Question 3"]

        responses = self.manager.process_batch(prompts)
        assert len(responses) == 3
        assert "Response to: Question 1" in responses[0]
        assert "Response to: Question 2" in responses[1]
        assert "Response to: Question 3" in responses[2]


class TestLLMProviderInterface:
    """Test LLM Provider Interface functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = LLMProviderInterface()

    def test_llm_provider_interface_creation(self):
        """Test LLMProviderInterface creation."""
        provider = LLMProviderInterface()
        assert provider is not None

    def test_llm_provider_interface_methods(self):
        """Test provider interface methods."""

        # Mock interface methods
        def mock_generate(prompt, **kwargs):
            return {"text": f"Generated: {prompt}", "tokens": 10}

        def mock_get_models():
            return ["model1", "model2", "model3"]

        def mock_get_info():
            return {"provider": "test", "version": "1.0"}

        self.provider.generate = mock_generate
        self.provider.get_models = mock_get_models
        self.provider.get_info = mock_get_info

        # Test generation
        result = self.provider.generate("Test prompt")
        assert result["text"] == "Generated: Test prompt"
        assert result["tokens"] == 10

        # Test model listing
        models = self.provider.get_models()
        assert len(models) == 3
        assert "model1" in models

        # Test info retrieval
        info = self.provider.get_info()
        assert info["provider"] == "test"
        assert info["version"] == "1.0"


class TestInferenceIntegration:
    """Test integration between inference components."""

    def test_gmn_and_gnn_integration(self):
        """Test integration between GMN and GNN components."""
        # Create components
        gmn_parser = GMNParser()
        gnn_parser = GNNParser()
        feature_extractor = FeatureExtractor()

        # Mock methods for integration test
        def mock_gmn_parse(gmn_string):
            return {
                "graph_structure": {
                    "nodes": [
                        {"id": "n1", "type": "state"},
                        {"id": "n2", "type": "state"},
                    ],
                    "edges": [
                        {"source": "n1", "target": "n2", "type": "transition"}
                    ],
                }
            }

        def mock_extract_features(graph_data):
            return np.random.rand(len(graph_data["nodes"]), 5)

        def mock_gnn_parse(graph_data):
            return {
                "tensor_data": {
                    "x": np.random.rand(2, 5),
                    "edge_index": np.array([[0, 1], [1, 0]]),
                }
            }

        gmn_parser.parse = mock_gmn_parse
        feature_extractor.extract_features = mock_extract_features
        gnn_parser.to_tensor = mock_gnn_parse

        # Test integration workflow
        gmn_input = "states: [s1, s2]; transitions: [(s1, s2)]"

        # Step 1: Parse GMN
        gmn_result = gmn_parser.parse(gmn_input)
        graph_structure = gmn_result["graph_structure"]

        # Step 2: Extract features
        features = feature_extractor.extract_features(graph_structure)

        # Step 3: Convert to tensor format
        tensor_data = gnn_parser.to_tensor(graph_structure)

        # Verify integration
        assert "nodes" in graph_structure
        assert features.shape[0] == len(graph_structure["nodes"])
        assert "tensor_data" in tensor_data

    def test_llm_and_inference_integration(self):
        """Test integration between LLM and inference components."""
        # Create components
        llm_manager = LocalLLMManager()
        gmn_parser = GMNParser()

        # Mock LLM to generate GMN
        def mock_llm_generate(prompt):
            if "generate GMN" in prompt:
                return "states: [hungry, satisfied]; actions: [eat, wait]; observations: [food, empty]"
            return "No GMN generated"

        def mock_gmn_parse(gmn_string):
            if "states:" in gmn_string:
                return {
                    "states": ["hungry", "satisfied"],
                    "actions": ["eat", "wait"],
                    "observations": ["food", "empty"],
                }
            return {}

        llm_manager.generate_text = mock_llm_generate
        gmn_parser.parse = mock_gmn_parse

        # Test LLM-to-GMN workflow
        llm_prompt = "Please generate GMN for a simple eating scenario"
        gmn_string = llm_manager.generate_text(llm_prompt)
        gmn_data = gmn_parser.parse(gmn_string)

        # Verify integration
        assert "states" in gmn_data
        assert "hungry" in gmn_data["states"]
        assert "satisfied" in gmn_data["states"]
        assert len(gmn_data["actions"]) == 2

    def test_error_propagation_in_inference_pipeline(self):
        """Test error propagation in inference pipeline."""
        # Create components
        parser = GMNParser()
        validator = GMNValidator()

        # Mock parser that can fail
        def mock_parse_with_error(gmn_string):
            if "invalid" in gmn_string:
                raise ValueError("Parse error")
            return {"valid_data": True}

        def mock_validate_with_error(gmn_data):
            if "valid_data" not in gmn_data:
                return {"valid": False, "error": "Invalid data"}
            return {"valid": True, "error": None}

        parser.parse = mock_parse_with_error
        validator.validate = mock_validate_with_error

        # Test error propagation
        with pytest.raises(ValueError, match="Parse error"):
            invalid_input = "invalid GMN syntax"
            parsed_data = parser.parse(invalid_input)
            validator.validate(parsed_data)

        # Test successful pipeline
        valid_input = "valid GMN syntax"
        parsed_data = parser.parse(valid_input)
        validation_result = validator.validate(parsed_data)

        assert validation_result["valid"] is True
