"""
Test GraphQL schema import error handling
"""

import importlib
import os
import sys

import pytest


class TestGraphQLSchemaImportErrors:
    """Test GraphQL schema when imports fail"""

    def test_schema_with_no_strawberry(self, monkeypatch):
        """Test schema loads with minimal strawberry fallback"""
        # Remove strawberry and mock_strawberry from modules
        modules_to_remove = ["strawberry", "mock_strawberry", "api.graphql.schema"]

        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

        # Block strawberry import
        original_import = __builtins__["__import__"]

        def mock_import(name, *args):
            if "strawberry" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args)

        import builtins

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Now import schema - it should use minimal fallback
        try:
            import api.graphql.schema as schema

            # Verify minimal strawberry is defined
            assert hasattr(schema.strawberry, "type")
            assert hasattr(schema.strawberry, "field")
            assert hasattr(schema.strawberry, "enum")
            assert hasattr(schema.strawberry, "ID")
            assert schema.Info == object
            assert schema.JSON == dict
        finally:
            # Restore import
            monkeypatch.undo()

    def test_schema_with_no_data_models(self, monkeypatch):
        """Test schema loads with fallback enums when data models are missing"""
        # Remove data model modules
        modules_to_remove = [
            "agents.base.data_model",
            "coalitions.coalition.coalition_models",
            "world.h3_world",
            "api.graphql.schema",
        ]

        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

        # Block data model imports
        original_import = __builtins__["__import__"]

        def mock_import(name, *args):
            if any(
                blocked in name
                for blocked in [
                    "agents.base.data_model",
                    "coalitions.coalition.coalition_models",
                    "world.h3_world",
                ]
            ):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args)

        import builtins

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Import strawberry mock first
        sys.path.insert(0, os.path.dirname(__file__))
        import mock_strawberry

        sys.modules["strawberry"] = mock_strawberry

        try:
            # Now import schema - it should define fallback enums
            import api.graphql.schema as schema

            # Verify fallback enums are defined
            assert hasattr(schema, "AgentStatus")
            assert schema.AgentStatus.IDLE.value == "idle"
            assert schema.AgentStatus.MOVING.value == "moving"

            assert hasattr(schema, "AgentClass")
            assert schema.AgentClass.EXPLORER.value == "explorer"

            assert hasattr(schema, "PersonalityTraits")
            assert schema.PersonalityTraits.OPENNESS.value == "openness"

            assert hasattr(schema, "AgentCapability")
            assert schema.AgentCapability.MOVEMENT.value == "movement"

            assert hasattr(schema, "ActionType")
            assert schema.ActionType.MOVE.value == "move"

            assert hasattr(schema, "CoalitionStatus")
            assert schema.CoalitionStatus.FORMING.value == "forming"

            assert hasattr(schema, "CoalitionRole")
            assert schema.CoalitionRole.LEADER.value == "leader"

            assert hasattr(schema, "CoalitionGoalStatus")
            assert schema.CoalitionGoalStatus.PROPOSED.value == "proposed"

            assert hasattr(schema, "Biome")
            assert schema.Biome.FOREST.value == "forest"

            assert hasattr(schema, "TerrainType")
            assert schema.TerrainType.FLAT.value == "flat"

            # Verify all enum values
            assert len(schema.AgentStatus) == 7
            assert len(schema.AgentClass) == 4
            assert len(schema.PersonalityTraits) == 5
            assert len(schema.AgentCapability) == 8
            assert len(schema.ActionType) == 10
            assert len(schema.CoalitionStatus) == 5
            assert len(schema.CoalitionRole) == 5
            assert len(schema.CoalitionGoalStatus) == 6
            assert len(schema.Biome) == 10
            assert len(schema.TerrainType) == 6

        finally:
            # Restore import
            monkeypatch.undo()
            if "strawberry" in sys.modules:
                del sys.modules["strawberry"]
