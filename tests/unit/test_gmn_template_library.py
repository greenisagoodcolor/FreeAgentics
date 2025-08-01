"""Test suite for GMN Template Library.

Following TDD principles - comprehensive tests for all template types and parameters.
Focus on template generation, caching, and parameter validation.
"""

import time

import pytest

from inference.active.gmn_schema import GMNNodeType, GMNSpecification
from inference.active.gmn_template_library import (
    AgentType,
    AnalystTemplate,
    CreativeTemplate,
    ExplorerTemplate,
    GMNTemplateBuilder,
    GMNTemplateError,
    GMNTemplateFactory,
    TemplateCache,
)


class TestGMNTemplateBuilder:
    """Test base template builder functionality."""

    def test_base_template_builder_is_abstract(self):
        """Test that base template builder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GMNTemplateBuilder()

    def test_template_builder_interface(self):
        """Test that template builders implement required interface."""
        template = ExplorerTemplate()

        # Should have required methods
        assert hasattr(template, "build")
        assert hasattr(template, "with_parameters")
        assert hasattr(template, "validate_parameters")
        assert hasattr(template, "get_default_parameters")

    def test_template_parameter_validation(self):
        """Test template parameter validation."""
        template = ExplorerTemplate()

        # Valid parameters
        valid_params = {
            "num_locations": 4,
            "num_actions": 5,
            "location_names": ["start", "goal", "obstacle", "checkpoint"],
        }
        is_valid, errors = template.validate_parameters(valid_params)
        assert is_valid
        assert len(errors) == 0

        # Invalid parameters
        invalid_params = {
            "num_locations": 0,  # Should be >= 1
            "num_actions": -1,  # Should be >= 1
        }
        is_valid, errors = template.validate_parameters(invalid_params)
        assert not is_valid
        assert len(errors) > 0

    def test_template_default_parameters(self):
        """Test that templates provide sensible defaults."""
        template = ExplorerTemplate()
        defaults = template.get_default_parameters()

        assert "num_locations" in defaults
        assert "num_actions" in defaults
        assert defaults["num_locations"] >= 1
        assert defaults["num_actions"] >= 1

        # Defaults should pass validation
        is_valid, errors = template.validate_parameters(defaults)
        assert is_valid


class TestExplorerTemplate:
    """Test Explorer agent template."""

    def test_create_basic_explorer_template(self):
        """Test creating basic explorer agent with default parameters."""
        template = ExplorerTemplate()
        spec = template.build()

        assert isinstance(spec, GMNSpecification)
        assert spec.name.startswith("Explorer")
        assert len(spec.nodes) >= 3  # At least state, observation, action nodes

        # Check for required node types
        node_types = {node.type for node in spec.nodes}
        assert GMNNodeType.STATE in node_types
        assert GMNNodeType.OBSERVATION in node_types
        assert GMNNodeType.ACTION in node_types

    def test_explorer_template_with_custom_parameters(self):
        """Test explorer template with custom parameters."""
        template = ExplorerTemplate()

        params = {
            "num_locations": 6,
            "num_actions": 4,
            "location_names": ["home", "work", "store", "park", "gym", "restaurant"],
            "action_names": ["move_north", "move_south", "move_east", "move_west"],
        }

        spec = template.with_parameters(params).build()

        # Find location state node
        location_nodes = [
            n for n in spec.nodes if n.type == GMNNodeType.STATE and "location" in n.id.lower()
        ]
        assert len(location_nodes) >= 1
        assert location_nodes[0].properties["num_states"] == 6

        # Find action node
        action_nodes = [n for n in spec.nodes if n.type == GMNNodeType.ACTION]
        assert len(action_nodes) >= 1
        assert action_nodes[0].properties["num_actions"] == 4

    def test_explorer_template_creates_valid_transition_matrices(self):
        """Test that explorer template creates mathematically valid transition matrices."""
        template = ExplorerTemplate()
        spec = template.with_parameters({"num_locations": 4, "num_actions": 4}).build()

        # Find transition nodes
        transition_nodes = [n for n in spec.nodes if n.type == GMNNodeType.TRANSITION]
        assert len(transition_nodes) >= 1

        # Check transition matrix properties
        transition = transition_nodes[0]
        if "matrix" in transition.properties:
            import numpy as np

            matrix = np.array(transition.properties["matrix"])

            # Should be 3D: [next_state, current_state, action]
            assert len(matrix.shape) == 3
            assert matrix.shape[0] == 4  # num_locations
            assert matrix.shape[1] == 4  # num_locations
            assert matrix.shape[2] == 4  # num_actions

            # Each column should sum to 1 (probability constraint)
            for action in range(matrix.shape[2]):
                for state in range(matrix.shape[1]):
                    col_sum = np.sum(matrix[:, state, action])
                    assert abs(col_sum - 1.0) < 1e-6

    def test_explorer_template_grid_world_mode(self):
        """Test explorer template with grid world configuration."""
        template = ExplorerTemplate()

        params = {
            "world_type": "grid",
            "grid_width": 3,
            "grid_height": 3,
            "num_actions": 5,  # N, S, E, W, Stay
        }

        spec = template.with_parameters(params).build()

        # Should have 9 locations (3x3 grid)
        location_nodes = [
            n for n in spec.nodes if n.type == GMNNodeType.STATE and "location" in n.id.lower()
        ]
        assert location_nodes[0].properties["num_states"] == 9

    def test_explorer_template_invalid_parameters(self):
        """Test explorer template with invalid parameters."""
        template = ExplorerTemplate()

        # Zero locations
        with pytest.raises(GMNTemplateError, match="num_locations must be >= 1"):
            template.with_parameters({"num_locations": 0}).build()

        # Negative actions
        with pytest.raises(GMNTemplateError, match="num_actions must be >= 1"):
            template.with_parameters({"num_actions": -1}).build()

        # Mismatched location names
        with pytest.raises(GMNTemplateError, match="location_names length.*num_locations"):
            template.with_parameters(
                {
                    "num_locations": 3,
                    "location_names": ["a", "b"],  # Only 2 names for 3 locations
                }
            ).build()


class TestAnalystTemplate:
    """Test Analyst agent template."""

    def test_create_basic_analyst_template(self):
        """Test creating basic analyst agent with default parameters."""
        template = AnalystTemplate()
        spec = template.build()

        assert isinstance(spec, GMNSpecification)
        assert spec.name.startswith("Analyst")

        # Analyst should have data processing nodes
        node_types = {node.type for node in spec.nodes}
        assert GMNNodeType.STATE in node_types
        assert GMNNodeType.OBSERVATION in node_types
        assert GMNNodeType.ACTION in node_types

        # Should have belief/preference nodes for decision making
        node_ids = {node.id.lower() for node in spec.nodes}
        assert any("belief" in node_id or "decision" in node_id for node_id in node_ids)

    def test_analyst_template_with_data_dimensions(self):
        """Test analyst template with custom data dimensions."""
        template = AnalystTemplate()

        params = {
            "num_data_sources": 5,
            "num_analysis_actions": 7,
            "data_types": ["numerical", "categorical", "text", "temporal", "spatial"],
            "analysis_methods": [
                "aggregate",
                "filter",
                "transform",
                "classify",
                "predict",
                "cluster",
                "visualize",
            ],
        }

        spec = template.with_parameters(params).build()

        # Find data source observation node
        obs_nodes = [
            n for n in spec.nodes if n.type == GMNNodeType.OBSERVATION and "data" in n.id.lower()
        ]
        assert len(obs_nodes) >= 1
        assert obs_nodes[0].properties["num_observations"] == 5

        # Find analysis action node
        action_nodes = [
            n for n in spec.nodes if n.type == GMNNodeType.ACTION and "analysis" in n.id.lower()
        ]
        assert len(action_nodes) >= 1
        assert action_nodes[0].properties["num_actions"] == 7

    def test_analyst_template_creates_preference_model(self):
        """Test that analyst template includes preference model for decision making."""
        template = AnalystTemplate()
        spec = template.build()

        # Should have preference nodes
        pref_nodes = [n for n in spec.nodes if n.type == GMNNodeType.PREFERENCE]
        assert len(pref_nodes) >= 1

        # Preference should be connected to observations
        pref_node = pref_nodes[0]
        pref_edges = [e for e in spec.edges if e.source == pref_node.id or e.target == pref_node.id]
        assert len(pref_edges) >= 1


class TestCreativeTemplate:
    """Test Creative agent template."""

    def test_create_basic_creative_template(self):
        """Test creating basic creative agent with default parameters."""
        template = CreativeTemplate()
        spec = template.build()

        assert isinstance(spec, GMNSpecification)
        assert spec.name.startswith("Creative")

        # Creative should have generation/exploration nodes
        node_ids = {node.id.lower() for node in spec.nodes}
        assert any(
            "creative" in node_id or "generate" in node_id or "explore" in node_id
            for node_id in node_ids
        )

    def test_creative_template_with_generation_parameters(self):
        """Test creative template with generation-specific parameters."""
        template = CreativeTemplate()

        params = {
            "num_creative_states": 6,
            "num_generation_actions": 8,
            "creativity_level": 0.7,  # High creativity
            "exploration_factor": 0.5,
        }

        spec = template.with_parameters(params).build()

        # Find creative state node
        state_nodes = [
            n for n in spec.nodes if n.type == GMNNodeType.STATE and "creative" in n.id.lower()
        ]
        assert len(state_nodes) >= 1
        assert state_nodes[0].properties["num_states"] == 6

    def test_creative_template_exploration_bias(self):
        """Test that creative template includes exploration bias in transition matrices."""
        template = CreativeTemplate()
        spec = template.with_parameters({"exploration_factor": 0.8}).build()

        # Find transition nodes
        transition_nodes = [n for n in spec.nodes if n.type == GMNNodeType.TRANSITION]
        assert len(transition_nodes) >= 1

        # Should have exploration factor in metadata
        transition = transition_nodes[0]
        assert (
            "exploration_factor" in transition.metadata
            or "exploration_factor" in transition.properties
        )


class TestGMNTemplateFactory:
    """Test template factory functionality."""

    def test_create_template_by_agent_type(self):
        """Test creating templates by agent type enum."""
        factory = GMNTemplateFactory()

        explorer = factory.create_template(AgentType.EXPLORER)
        assert isinstance(explorer, ExplorerTemplate)

        analyst = factory.create_template(AgentType.ANALYST)
        assert isinstance(analyst, AnalystTemplate)

        creative = factory.create_template(AgentType.CREATIVE)
        assert isinstance(creative, CreativeTemplate)

    def test_create_template_by_string(self):
        """Test creating templates by string name."""
        factory = GMNTemplateFactory()

        explorer = factory.create_template_by_name("explorer")
        assert isinstance(explorer, ExplorerTemplate)

        analyst = factory.create_template_by_name("analyst")
        assert isinstance(analyst, AnalystTemplate)

        creative = factory.create_template_by_name("creative")
        assert isinstance(creative, CreativeTemplate)

    def test_invalid_template_type_raises_error(self):
        """Test that invalid template types raise appropriate errors."""
        factory = GMNTemplateFactory()

        with pytest.raises(GMNTemplateError, match="Unknown agent type"):
            factory.create_template_by_name("invalid_type")

    def test_factory_template_registration(self):
        """Test registering custom templates with factory."""
        factory = GMNTemplateFactory()

        # Register custom template
        class CustomTemplate(GMNTemplateBuilder):
            def build(self):
                return self._create_minimal_spec("Custom")

        factory.register_template("custom", CustomTemplate)

        custom = factory.create_template_by_name("custom")
        assert isinstance(custom, CustomTemplate)

    def test_list_available_templates(self):
        """Test listing available template types."""
        factory = GMNTemplateFactory()
        templates = factory.list_available_templates()

        assert "explorer" in templates
        assert "analyst" in templates
        assert "creative" in templates
        assert len(templates) >= 3


class TestTemplateCache:
    """Test template caching functionality."""

    def test_template_caching_basic(self):
        """Test basic template caching functionality."""
        cache = TemplateCache(max_size=100)
        template = ExplorerTemplate()

        params = {"num_locations": 4, "num_actions": 5}

        # First call should cache result
        spec1 = cache.get_or_create(template, params)
        assert isinstance(spec1, GMNSpecification)

        # Second call should return cached result
        spec2 = cache.get_or_create(template, params)
        assert spec1 is spec2  # Same object reference

    def test_template_cache_different_parameters(self):
        """Test that cache distinguishes between different parameters."""
        cache = TemplateCache(max_size=100)
        template = ExplorerTemplate()

        params1 = {"num_locations": 4, "num_actions": 5}
        params2 = {"num_locations": 6, "num_actions": 5}

        spec1 = cache.get_or_create(template, params1)
        spec2 = cache.get_or_create(template, params2)

        assert spec1 is not spec2  # Different specs
        assert spec1.nodes[0].properties.get("num_states") != spec2.nodes[0].properties.get(
            "num_states"
        )

    def test_template_cache_size_limit(self):
        """Test that cache respects size limits."""
        cache = TemplateCache(max_size=2)
        template = ExplorerTemplate()

        # Fill cache to capacity
        spec1 = cache.get_or_create(template, {"num_locations": 1})
        spec2 = cache.get_or_create(template, {"num_locations": 2})

        # Adding third item should evict first
        spec3 = cache.get_or_create(template, {"num_locations": 3})

        # First item should be evicted (LRU)
        spec1_again = cache.get_or_create(template, {"num_locations": 1})
        assert spec1 is not spec1_again  # New object, not cached

    def test_template_cache_performance(self):
        """Test that cache provides performance benefits."""
        cache = TemplateCache(max_size=100)
        template = ExplorerTemplate()
        params = {"num_locations": 10, "num_actions": 8}

        # Time first generation
        start_time = time.time()
        spec1 = cache.get_or_create(template, params)
        first_time = time.time() - start_time

        # Time cached retrieval
        start_time = time.time()
        spec2 = cache.get_or_create(template, params)
        cached_time = time.time() - start_time

        # Cached retrieval should be much faster
        assert cached_time < first_time * 0.1  # At least 10x faster
        assert spec1 is spec2


class TestTemplateIntegration:
    """Test integration between templates and existing GMN infrastructure."""

    def test_template_output_validates_with_schema(self):
        """Test that template output passes GMN schema validation."""
        from inference.active.gmn_schema import GMNSchemaValidator

        templates = [ExplorerTemplate(), AnalystTemplate(), CreativeTemplate()]
        validator = GMNSchemaValidator()

        for template in templates:
            spec = template.build()
            is_valid, errors = validator.validate_specification(spec)
            assert is_valid, f"Template {type(template).__name__} failed validation: {errors}"

    def test_template_output_compatible_with_parser(self):
        """Test that template output is compatible with existing GMN parser."""
        from inference.active.gmn_schema import GMNSchemaValidator

        template = ExplorerTemplate()
        spec = template.build()

        validator = GMNSchemaValidator()

        # Convert to parser format (this tests the integration)
        spec_dict = spec.to_dict()

        # Should be valid dictionary format
        assert "nodes" in spec_dict
        assert "edges" in spec_dict
        assert len(spec_dict["nodes"]) >= 3

    def test_template_performance_requirements(self):
        """Test that templates meet performance requirements."""
        templates = [
            (ExplorerTemplate(), "explorer"),
            (AnalystTemplate(), "analyst"),
            (CreativeTemplate(), "creative"),
        ]

        for template, name in templates:
            start_time = time.time()
            spec = template.build()
            generation_time = time.time() - start_time

            # Should generate within performance budget (20ms from requirements)
            assert (
                generation_time < 0.02
            ), f"{name} template took {generation_time:.3f}s (>20ms limit)"

            # Should produce valid output
            assert isinstance(spec, GMNSpecification)
            assert len(spec.nodes) >= 3


class TestTemplateErrorHandling:
    """Test error handling and edge cases."""

    def test_template_with_extreme_parameters(self):
        """Test templates with extreme but valid parameters."""
        template = ExplorerTemplate()

        # Large parameters
        large_params = {"num_locations": 100, "num_actions": 50}
        spec = template.with_parameters(large_params).build()
        assert isinstance(spec, GMNSpecification)

        # Minimal parameters
        minimal_params = {"num_locations": 1, "num_actions": 1}
        spec = template.with_parameters(minimal_params).build()
        assert isinstance(spec, GMNSpecification)

    def test_template_error_messages(self):
        """Test that templates provide clear error messages."""
        template = ExplorerTemplate()

        try:
            template.with_parameters({"num_locations": -1}).build()
            assert False, "Should have raised GMNTemplateError"
        except GMNTemplateError as e:
            assert "num_locations" in str(e)
            assert "must be >= 1" in str(e) or "positive" in str(e)

    def test_template_parameter_types(self):
        """Test that templates handle parameter type validation."""
        template = ExplorerTemplate()

        # String instead of int
        with pytest.raises(GMNTemplateError):
            template.with_parameters({"num_locations": "four"}).build()

        # Float instead of int (might be acceptable if converted)
        spec = template.with_parameters({"num_locations": 4.0}).build()
        assert isinstance(spec, GMNSpecification)


if __name__ == "__main__":
    pytest.main([__file__])
