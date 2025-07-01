"""
Comprehensive tests for Coalition Builder module.

Tests coalition formation, proposal management, and value evaluation
with proper PyMDP alignment and GNN notation support.
"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from coalitions.formation.coalition_builder import Coalition, CoalitionBuilder


class TestCoalition:
    """Test Coalition dataclass functionality."""

    def test_coalition_initialization_minimal(self):
        """Test creating coalition with minimal parameters."""
        coalition = Coalition(
            id="test_coalition", members=["agent1", "agent2"], type="resource_sharing"
        )

        assert coalition.id == "test_coalition"
        assert coalition.members == ["agent1", "agent2"]
        assert coalition.type == "resource_sharing"
        assert coalition.status == "forming"  # Default value
        assert coalition.metadata == {}  # Default empty dict

    def test_coalition_initialization_full(self):
        """Test creating coalition with all parameters."""
        metadata = {"priority": "high", "resources": ["water", "food"]}
        coalition = Coalition(
            id="test_coalition_2",
            members=["agent1", "agent2", "agent3"],
            type="defense",
            status="active",
            metadata=metadata,
        )

        assert coalition.id == "test_coalition_2"
        assert coalition.members == ["agent1", "agent2", "agent3"]
        assert coalition.type == "defense"
        assert coalition.status == "active"
        assert coalition.metadata == metadata

    def test_coalition_post_init_metadata(self):
        """Test that metadata is initialized to empty dict if None."""
        coalition = Coalition(id="test", members=["agent1"], type="exploration", metadata=None)

        assert coalition.metadata == {}
        assert isinstance(coalition.metadata, dict)

    def test_coalition_members_mutability(self):
        """Test that coalition members list is mutable."""
        coalition = Coalition(id="test", members=["agent1", "agent2"], type="trade")

        # Add new member
        coalition.members.append("agent3")
        assert "agent3" in coalition.members
        assert len(coalition.members) == 3

        # Remove member
        coalition.members.remove("agent1")
        assert "agent1" not in coalition.members
        assert len(coalition.members) == 2

    def test_coalition_metadata_mutability(self):
        """Test that coalition metadata dict is mutable."""
        coalition = Coalition(id="test", members=["agent1"], type="research")

        # Add metadata
        coalition.metadata["created_at"] = "2024-01-01"
        coalition.metadata["goals"] = ["discover", "analyze"]

        assert coalition.metadata["created_at"] == "2024-01-01"
        assert coalition.metadata["goals"] == ["discover", "analyze"]

    def test_coalition_equality(self):
        """Test coalition equality based on id."""
        coalition1 = Coalition(id="same_id", members=["agent1"], type="type1")

        coalition2 = Coalition(
            id="same_id", members=["agent2"], type="type2"  # Different members  # Different type
        )

        # Should be equal based on id
        assert coalition1.id == coalition2.id

        # But not the same object
        assert coalition1 is not coalition2


class TestCoalitionBuilder:
    """Test CoalitionBuilder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = CoalitionBuilder()

    def test_initialization(self):
        """Test CoalitionBuilder initialization."""
        assert isinstance(self.builder.coalitions, dict)
        assert len(self.builder.coalitions) == 0
        assert isinstance(self.builder.pending_proposals, list)
        assert len(self.builder.pending_proposals) == 0

    def test_propose_coalition_basic(self):
        """Test basic coalition proposal."""
        proposer = "agent1"
        members = ["agent2", "agent3"]

        coalition = self.builder.propose_coalition(proposer, members)

        # Check coalition properties
        assert coalition.id == "coalition_0"
        assert coalition.members == ["agent1", "agent2", "agent3"]
        assert coalition.type == "resource_sharing"  # Default
        assert coalition.status == "proposed"

        # Check it was stored
        assert coalition.id in self.builder.coalitions
        assert self.builder.coalitions[coalition.id] == coalition

    def test_propose_coalition_custom_type(self):
        """Test coalition proposal with custom type."""
        proposer = "explorer1"
        members = ["explorer2", "explorer3"]
        coalition_type = "exploration"

        coalition = self.builder.propose_coalition(proposer, members, coalition_type)

        assert coalition.type == "exploration"
        assert coalition.members == ["explorer1", "explorer2", "explorer3"]

    def test_propose_multiple_coalitions(self):
        """Test proposing multiple coalitions."""
        # First coalition
        coalition1 = self.builder.propose_coalition("agent1", ["agent2"])
        assert coalition1.id == "coalition_0"

        # Second coalition
        coalition2 = self.builder.propose_coalition("agent3", ["agent4", "agent5"])
        assert coalition2.id == "coalition_1"

        # Third coalition
        coalition3 = self.builder.propose_coalition("agent6", ["agent7"])
        assert coalition3.id == "coalition_2"

        # Check all are stored
        assert len(self.builder.coalitions) == 3
        assert all(f"coalition_{i}" in self.builder.coalitions for i in range(3))

    def test_propose_coalition_empty_members(self):
        """Test proposing coalition with no additional members."""
        coalition = self.builder.propose_coalition("solo_agent", [])

        assert coalition.members == ["solo_agent"]
        assert len(coalition.members) == 1

    def test_accept_proposal_valid(self):
        """Test accepting a valid coalition proposal."""
        # Create proposal
        coalition = self.builder.propose_coalition("agent1", ["agent2", "agent3"])
        coalition_id = coalition.id

        # Accept by a member
        result = self.builder.accept_proposal(coalition_id, "agent2")

        assert result is True
        assert self.builder.coalitions[coalition_id].status == "active"

    def test_accept_proposal_by_proposer(self):
        """Test accepting proposal by the proposer."""
        coalition = self.builder.propose_coalition("agent1", ["agent2"])

        result = self.builder.accept_proposal(coalition.id, "agent1")

        assert result is True
        assert coalition.status == "active"

    def test_accept_proposal_invalid_coalition(self):
        """Test accepting proposal for non-existent coalition."""
        result = self.builder.accept_proposal("non_existent_coalition", "agent1")

        assert result is False

    def test_accept_proposal_non_member(self):
        """Test accepting proposal by non-member agent."""
        coalition = self.builder.propose_coalition("agent1", ["agent2"])

        # Agent3 is not a member
        result = self.builder.accept_proposal(coalition.id, "agent3")

        assert result is False
        assert coalition.status == "proposed"  # Status unchanged

    def test_evaluate_coalition_value_basic(self):
        """Test basic coalition value evaluation."""
        coalition = Coalition(id="test", members=["agent1", "agent2", "agent3"], type="trade")

        value = self.builder.evaluate_coalition_value(coalition)

        # 3 members * 10.0 = 30.0
        assert value == 30.0

    def test_evaluate_coalition_value_single_member(self):
        """Test coalition value for single member."""
        coalition = Coalition(id="solo", members=["agent1"], type="research")

        value = self.builder.evaluate_coalition_value(coalition)

        assert value == 10.0

    def test_evaluate_coalition_value_large_coalition(self):
        """Test coalition value for large coalition."""
        members = [f"agent{i}" for i in range(10)]
        coalition = Coalition(id="large", members=members, type="defense")

        value = self.builder.evaluate_coalition_value(coalition)

        assert value == 100.0  # 10 members * 10.0

    def test_coalition_workflow(self):
        """Test complete coalition formation workflow."""
        # Step 1: Propose coalition
        proposer = "leader_agent"
        members = ["worker1", "worker2", "worker3"]
        coalition = self.builder.propose_coalition(proposer, members, "construction")

        assert coalition.status == "proposed"
        assert len(self.builder.coalitions) == 1

        # Step 2: Evaluate value
        value = self.builder.evaluate_coalition_value(coalition)
        assert value == 40.0  # 4 members * 10.0

        # Step 3: Accept proposal
        accepted = self.builder.accept_proposal(coalition.id, "worker1")
        assert accepted is True
        assert coalition.status == "active"

        # Step 4: Try to accept again (should still return True)
        accepted_again = self.builder.accept_proposal(coalition.id, "worker2")
        assert accepted_again is True
        assert coalition.status == "active"

    def test_multiple_coalition_management(self):
        """Test managing multiple coalitions simultaneously."""
        # Create multiple coalitions
        coalitions = []
        for i in range(5):
            coalition = self.builder.propose_coalition(
                f"leader{i}", [f"member{i}_1", f"member{i}_2"], f"type{i}"
            )
            coalitions.append(coalition)

        assert len(self.builder.coalitions) == 5

        # Accept some coalitions
        self.builder.accept_proposal(coalitions[0].id, "member0_1")
        self.builder.accept_proposal(coalitions[2].id, "member2_2")
        self.builder.accept_proposal(coalitions[4].id, "leader4")

        # Check statuses
        assert self.builder.coalitions["coalition_0"].status == "active"
        assert self.builder.coalitions["coalition_1"].status == "proposed"
        assert self.builder.coalitions["coalition_2"].status == "active"
        assert self.builder.coalitions["coalition_3"].status == "proposed"
        assert self.builder.coalitions["coalition_4"].status == "active"

        # Evaluate all coalitions
        total_value = sum(self.builder.evaluate_coalition_value(c) for c in coalitions)
        assert total_value == 150.0  # 5 coalitions * 3 members each * 10.0

    def test_coalition_builder_state_persistence(self):
        """Test that coalition builder maintains state correctly."""
        # Add some coalitions
        c1 = self.builder.propose_coalition("a1", ["a2", "a3"])
        c2 = self.builder.propose_coalition("b1", ["b2"])

        # Accept one
        self.builder.accept_proposal(c1.id, "a2")

        # Create new builder instance (simulating reload)
        new_builder = CoalitionBuilder()

        # State should be independent
        assert len(new_builder.coalitions) == 0
        assert len(self.builder.coalitions) == 2

        # Original builder state unchanged
        assert self.builder.coalitions[c1.id].status == "active"
        assert self.builder.coalitions[c2.id].status == "proposed"


class TestCoalitionBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = CoalitionBuilder()

    def test_propose_coalition_duplicate_members(self):
        """Test proposing coalition with duplicate members."""
        # Proposer is also in members list
        coalition = self.builder.propose_coalition(
            "agent1", ["agent1", "agent2", "agent1"]  # Duplicates
        )

        # Should still work, duplicates included
        assert coalition.members == ["agent1", "agent1", "agent2", "agent1"]

    def test_accept_proposal_multiple_times(self):
        """Test accepting same proposal multiple times."""
        coalition = self.builder.propose_coalition("a1", ["a2"])

        # First accept
        result1 = self.builder.accept_proposal(coalition.id, "a1")
        assert result1 is True
        assert coalition.status == "active"

        # Second accept by same agent
        result2 = self.builder.accept_proposal(coalition.id, "a1")
        assert result2 is True
        assert coalition.status == "active"  # Still active

        # Accept by different member
        result3 = self.builder.accept_proposal(coalition.id, "a2")
        assert result3 is True
        assert coalition.status == "active"

    def test_evaluate_empty_coalition(self):
        """Test evaluating coalition with no members."""
        coalition = Coalition(id="empty", members=[], type="test")

        value = self.builder.evaluate_coalition_value(coalition)
        assert value == 0.0

    def test_coalition_type_variations(self):
        """Test various coalition types."""
        types = [
            "resource_sharing",
            "defense",
            "exploration",
            "trade",
            "research",
            "construction",
            "diplomatic",
            "military",
            "economic",
            "cultural",
            "scientific",
            "agricultural",
        ]

        for i, ctype in enumerate(types):
            coalition = self.builder.propose_coalition(f"leader{i}", [f"member{i}"], ctype)
            assert coalition.type == ctype
            assert coalition.id == f"coalition_{i}"

    def test_large_scale_coalition_operations(self):
        """Test coalition builder with many coalitions."""
        # Create 100 coalitions
        for i in range(100):
            members = [f"agent{i}_{j}" for j in range(i % 5 + 1)]
            self.builder.propose_coalition(f"leader{i}", members)

        assert len(self.builder.coalitions) == 100

        # Accept half of them
        for i in range(0, 100, 2):
            self.builder.accept_proposal(f"coalition_{i}", f"leader{i}")

        # Count active vs proposed
        active_count = sum(1 for c in self.builder.coalitions.values() if c.status == "active")
        proposed_count = sum(1 for c in self.builder.coalitions.values() if c.status == "proposed")

        assert active_count == 50
        assert proposed_count == 50

        # Calculate total value
        total_value = sum(
            self.builder.evaluate_coalition_value(c) for c in self.builder.coalitions.values()
        )

        # Should be sum of (i % 5 + 2) * 10 for i in range(100)
        # where +2 accounts for leader + members
        expected_value = sum((i % 5 + 2) * 10 for i in range(100))
        assert total_value == expected_value


class TestCoalitionBuilderIntegration:
    """Integration tests for coalition builder with other components."""

    def test_coalition_builder_with_pymdp_alignment(self):
        """Test coalition builder integration with PyMDP concepts."""
        builder = CoalitionBuilder()

        # Create coalition representing belief sharing
        belief_coalition = builder.propose_coalition(
            "belief_updater", ["observer1", "observer2"], "belief_sharing"
        )

        # Add PyMDP-related metadata
        belief_coalition.metadata["shared_beliefs"] = {
            "state_dim": 4,
            "obs_dim": 3,
            "policy_len": 5,
        }
        belief_coalition.metadata["free_energy_threshold"] = 0.5

        # Accept and evaluate
        builder.accept_proposal(belief_coalition.id, "observer1")
        value = builder.evaluate_coalition_value(belief_coalition)

        assert belief_coalition.status == "active"
        assert value == 30.0
        assert "shared_beliefs" in belief_coalition.metadata

    def test_coalition_builder_with_gnn_notation(self):
        """Test coalition builder with GNN (Generalized Notation Notation) concepts."""
        builder = CoalitionBuilder()

        # Create coalition for notation standardization
        notation_coalition = builder.propose_coalition(
            "notation_expert", ["analyst1", "analyst2", "analyst3"], "notation_standardization"
        )

        # Add GNN metadata
        notation_coalition.metadata["notation_system"] = "GNN"
        notation_coalition.metadata["notation_version"] = "1.0"
        notation_coalition.metadata["supported_notations"] = [
            "belief_updates",
            "policy_selection",
            "free_energy_calculation",
        ]

        # Process coalition
        builder.accept_proposal(notation_coalition.id, "analyst2")

        assert notation_coalition.status == "active"
        assert notation_coalition.metadata["notation_system"] == "GNN"
        assert len(notation_coalition.metadata["supported_notations"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
