import sys
import unittest
from pathlib import Path

from agents.base.perception import PerceptionSystem
from inference.gnn.parser import GMNParser
from infrastructure.database.models import KnowledgeGraph

"""Test GNN Repository Structure

This test validates that the repository follows the GNN-based structure
and that all components are properly organized.
"""
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGNNStructure(unittest.TestCase):
    """Test that the repository follows GNN-based structure."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.root_dir = Path(__file__).parent.parent

    def test_directory_structure_exists(self) -> None:
        """Test that all required directories exist."""
        required_dirs = [
            "docs",
            "agents",
            "inference",
            "world",
            "coalitions",
            "tests",
            "web",
            "api",
            "infrastructure",
            "scripts",
            "data",
            "config",
        ]
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            self.assertTrue(full_path.exists(), f"Required directory '{dir_path}' does not exist")

    def test_documentation_files_exist(self) -> None:
        """Test that key documentation files exist."""
        required_docs = [
            "README.md",
            "docs/active-inference/active-inference-guide.md",
            "docs/gnn/README.md",
        ]
        for doc_path in required_docs:
            full_path = self.root_dir / doc_path
            self.assertTrue(
                full_path.exists(),
                f"Required documentation '{doc_path}' does not exist",
            )

    def test_main_pipeline_exists(self) -> None:
        """Test that simulation engine exists as main orchestrator."""
        main_path = self.root_dir / "world" / "simulation" / "engine.py"
        self.assertTrue(main_path.exists(), "Simulation engine 'engine.py' does not exist")
        # Check for key simulation components
        with open(main_path) as f:
            content = f.read()
            self.assertIn("SimulationEngine", content)

    def test_gnn_models_exist(self) -> None:
        """Test that GNN model documentation exists."""
        # Check for GNN documentation instead of model files
        gnn_docs = self.root_dir / "docs" / "gnn"
        self.assertTrue(gnn_docs.exists(), "GNN documentation directory does not exist")

    def test_active_inference_components_exist(self) -> None:
        """Test that Active Inference components exist."""
        inference_dir = self.root_dir / "inference"
        self.assertTrue(inference_dir.exists(), "Inference directory does not exist")
        # Check for key inference components
        engine_path = inference_dir / "engine"
        self.assertTrue(engine_path.exists(), "Inference engine directory does not exist")

    def test_knowledge_modules_can_be_imported(self) -> None:
        """Test that knowledge modules can be imported."""
        try:
            # Import and test basic functionality
            # Test basic instantiation
            kg = KnowledgeGraph()
            self.assertIsNotNone(kg)
        except ImportError as e:
            self.fail(f"Failed to import knowledge modules: {e}")

    def test_no_scattered_files(self) -> None:
        """Test that there are no unexpected Python files in root."""
        root_files = list(self.root_dir.glob("*.py"))
        allowed_root_files = [
            "setup.py",
            "conftest.py",
            "fix_imports.py",
            "readiness.py",
            "fix_docstrings.py",
            "fix_triple_quotes.py",
        ]
        unexpected_files = [f for f in root_files if f.name not in allowed_root_files]
        self.assertEqual(
            len(unexpected_files),
            0,
            f"Found unexpected Python files in root: " f"{[f.name for f in unexpected_files]}",
        )

    def test_clean_separation_of_concerns(self) -> None:
        """Test that GNN models are properly documented."""
        # Check for GNN model documentation instead of model files
        gnn_docs_dir = self.root_dir / "docs" / "gnn"
        self.assertTrue(gnn_docs_dir.exists(), "GNN models documentation directory should exist")
        # Check for GNN documentation files
        model_format_doc = gnn_docs_dir / "model-format.md"
        self.assertTrue(model_format_doc.exists(), "GNN model format documentation should exist")

    def test_architecture_screams_active_inference(self) -> None:
        """Test that the architecture clearly shows this is an Active Inference platform."""
        ai_indicators = [
            self.root_dir / "agents" / "active_inference",
            self.root_dir / "docs" / "active-inference" / "active-inference-guide.md",
            self.root_dir / "inference" / "engine" / "active_inference.py",
        ]
        for indicator in ai_indicators:
            self.assertTrue(
                indicator.exists(),
                f"Active Inference indicator '{indicator}' not found",
            )
        # Check simulation engine for Active Inference content
        engine_path = self.root_dir / "world" / "simulation" / "engine.py"
        if engine_path.exists():
            with open(engine_path) as f:
                content = f.read()
                self.assertIn("active", content.lower())


class TestGNNModelParsing(unittest.TestCase):
    """Test GNN model parsing and validation."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.root_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(self.root_dir))

    def test_gnn_parser_imports(self) -> None:
        """Test that GNN parser can be imported."""
        try:
            # These imports will be used when we actually test them
            # Test basic instantiation
            parser = GMNParser()
            self.assertIsNotNone(parser)
        except ImportError as e:
            self.fail(f"Failed to import GNN components: {e}")


class TestPipelineIntegration(unittest.TestCase):
    """Test integration between pipeline components."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.root_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(self.root_dir))

    def test_agent_components_can_be_imported(self) -> None:
        """Test that agent components can be imported."""
        try:
            # These imports will be used when we actually test them
            # Test basic functionality exists
            self.assertTrue(hasattr(PerceptionSystem, "__init__"))
        except ImportError as e:
            self.fail(f"Failed to import agent components: {e}")


if __name__ == "__main__":
    unittest.main()
