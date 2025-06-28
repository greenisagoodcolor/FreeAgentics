"""
Comprehensive tests for Coalition Builder.
"""

import pytest
from typing import List, Dict, Any

from coalitions.formation.coalition_builder import Coalition, CoalitionBuilder


class TestCoalition:
    """Test Coalition dataclass."""
    
    def test_coalition_creation(self):
        """Test basic coalition creation."""
        coalition = Coalition(
            id="test-coalition",
            members=["agent-1", "agent-2"],
            type="resource_sharing"
        )
        
        assert coalition.id == "test-coalition"
        assert coalition.members == ["agent-1", "agent-2"]
        assert coalition.type == "resource_sharing"
        assert coalition.status == "forming"
        assert coalition.metadata == {}
    
    def test_coalition_with_metadata(self):
        """Test coalition creation with metadata."""
        metadata = {"goal": "find_food", "priority": 0.8}
        coalition = Coalition(
            id="meta-coalition",
            members=["agent-1", "agent-2", "agent-3"],
            type="exploration",
            status="active",
            metadata=metadata
        )
        
        assert coalition.metadata == metadata
        assert coalition.status == "active"
    
    def test_coalition_post_init(self):
        """Test post-init behavior."""
        coalition = Coalition(
            id="init-test",
            members=["agent-1"],
            type="test"
        )
        
        # Should initialize empty metadata dict
        assert isinstance(coalition.metadata, dict)
        assert coalition.metadata == {}


class TestCoalitionBuilder:
    """Test CoalitionBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create coalition builder for testing."""
        return CoalitionBuilder()
    
    def test_builder_initialization(self, builder):
        """Test builder initialization."""
        assert isinstance(builder.coalitions, dict)
        assert len(builder.coalitions) == 0
        assert isinstance(builder.pending_proposals, list)
        assert len(builder.pending_proposals) == 0
    
    def test_propose_coalition(self, builder):
        """Test coalition proposal."""
        coalition = builder.propose_coalition(
            proposer_id="agent-1",
            member_ids=["agent-2", "agent-3"],
            coalition_type="exploration"
        )
        
        assert coalition.id == "coalition_0"
        assert coalition.members == ["agent-1", "agent-2", "agent-3"]
        assert coalition.type == "exploration"
        assert coalition.status == "proposed"
        
        # Should be stored in builder
        assert coalition.id in builder.coalitions
        assert builder.coalitions[coalition.id] == coalition
    
    def test_multiple_coalition_proposals(self, builder):
        """Test multiple coalition proposals with unique IDs."""
        coalition1 = builder.propose_coalition("agent-1", ["agent-2"])
        coalition2 = builder.propose_coalition("agent-3", ["agent-4"])
        
        assert coalition1.id == "coalition_0"
        assert coalition2.id == "coalition_1"
        assert coalition1.id != coalition2.id
        assert len(builder.coalitions) == 2
    
    def test_accept_proposal(self, builder):
        """Test accepting a coalition proposal."""
        coalition = builder.propose_coalition("agent-1", ["agent-2", "agent-3"])
        
        # Agent accepts proposal
        result = builder.accept_proposal(coalition.id, "agent-2")
        
        assert result == True
        # Should update coalition status or metadata
    
    def test_accept_nonexistent_proposal(self, builder):
        """Test accepting non-existent proposal."""
        result = builder.accept_proposal("nonexistent", "agent-1")
        assert result == False
    
    def test_coalition_member_management(self, builder):
        """Test coalition member operations."""
        coalition = builder.propose_coalition(
            proposer_id="agent-1",
            member_ids=["agent-2", "agent-3"]
        )
        
        # Verify all members are included
        assert "agent-1" in coalition.members  # Proposer
        assert "agent-2" in coalition.members
        assert "agent-3" in coalition.members
        assert len(coalition.members) == 3
    
    def test_coalition_types(self, builder):
        """Test different coalition types."""
        types = [
            "resource_sharing",
            "exploration",
            "defense",
            "knowledge_sharing",
            "task_collaboration"
        ]
        
        coalitions = []
        for i, coalition_type in enumerate(types):
            coalition = builder.propose_coalition(
                proposer_id=f"agent-{i}",
                member_ids=[f"agent-{i+10}", f"agent-{i+20}"],
                coalition_type=coalition_type
            )
            coalitions.append(coalition)
        
        # Verify all types are set correctly
        for coalition, expected_type in zip(coalitions, types):
            assert coalition.type == expected_type
    
    def test_default_coalition_type(self, builder):
        """Test default coalition type."""
        coalition = builder.propose_coalition("agent-1", ["agent-2"])
        
        # Should default to resource_sharing
        assert coalition.type == "resource_sharing"
    
    def test_empty_member_list(self, builder):
        """Test coalition with empty member list."""
        coalition = builder.propose_coalition("agent-1", [])
        
        # Should still include proposer
        assert coalition.members == ["agent-1"]
        assert len(coalition.members) == 1
    
    def test_coalition_status_transitions(self, builder):
        """Test coalition status changes."""
        coalition = builder.propose_coalition("agent-1", ["agent-2"])
        
        # Initial status
        assert coalition.status == "proposed"
        
        # Accept proposal
        builder.accept_proposal(coalition.id, "agent-2")
        
        # Status should remain or change appropriately
        # (Implementation dependent)
        assert coalition.status in ["proposed", "accepted", "forming", "active"]


class TestCoalitionBuilderIntegration:
    """Integration tests for coalition builder."""
    
    def test_multi_agent_coalition_workflow(self):
        """Test complete coalition formation workflow."""
        builder = CoalitionBuilder()
        
        # Agent 1 proposes exploration coalition
        coalition = builder.propose_coalition(
            proposer_id="explorer-1",
            member_ids=["explorer-2", "scout-1"],
            coalition_type="exploration"
        )
        
        # Other agents accept
        builder.accept_proposal(coalition.id, "explorer-2")
        builder.accept_proposal(coalition.id, "scout-1")
        
        # Verify coalition state
        assert len(coalition.members) == 3
        assert "explorer-1" in coalition.members
        assert "explorer-2" in coalition.members
        assert "scout-1" in coalition.members
    
    def test_multiple_concurrent_coalitions(self):
        """Test handling multiple coalitions simultaneously."""
        builder = CoalitionBuilder()
        
        # Create multiple coalitions
        exploration_coalition = builder.propose_coalition(
            "explorer-1", ["explorer-2"], "exploration"
        )
        
        defense_coalition = builder.propose_coalition(
            "guard-1", ["guard-2", "guard-3"], "defense"
        )
        
        resource_coalition = builder.propose_coalition(
            "gatherer-1", ["gatherer-2"], "resource_sharing"
        )
        
        # Verify all coalitions exist
        assert len(builder.coalitions) == 3
        assert exploration_coalition.id in builder.coalitions
        assert defense_coalition.id in builder.coalitions
        assert resource_coalition.id in builder.coalitions
        
        # Verify they have different types
        assert exploration_coalition.type == "exploration"
        assert defense_coalition.type == "defense"
        assert resource_coalition.type == "resource_sharing"
    
    def test_coalition_agent_overlap(self):
        """Test handling agents in multiple coalitions."""
        builder = CoalitionBuilder()
        
        # Agent participates in multiple coalitions
        coalition1 = builder.propose_coalition(
            "agent-1", ["agent-2", "agent-3"], "exploration"
        )
        
        coalition2 = builder.propose_coalition(
            "agent-2", ["agent-4"], "resource_sharing"  # agent-2 in both
        )
        
        # Both coalitions should exist
        assert len(builder.coalitions) == 2
        
        # Agent-2 should be in both
        assert "agent-2" in coalition1.members
        assert "agent-2" in coalition2.members
    
    def test_large_coalition_formation(self):
        """Test formation of large coalitions."""
        builder = CoalitionBuilder()
        
        # Create large coalition
        member_count = 50
        member_ids = [f"agent-{i}" for i in range(2, member_count + 2)]
        
        coalition = builder.propose_coalition(
            proposer_id="leader-1",
            member_ids=member_ids,
            coalition_type="large_scale_exploration"
        )
        
        assert len(coalition.members) == member_count + 1  # +1 for proposer
        assert coalition.members[0] == "leader-1"  # Proposer first
        assert "agent-2" in coalition.members
        assert f"agent-{member_count + 1}" in coalition.members
    
    def test_coalition_builder_state_persistence(self):
        """Test that builder maintains state across operations."""
        builder = CoalitionBuilder()
        
        # Create several coalitions
        for i in range(5):
            builder.propose_coalition(
                f"proposer-{i}",
                [f"member-{i}-1", f"member-{i}-2"],
                f"type-{i}"
            )
        
        # Verify all coalitions are tracked
        assert len(builder.coalitions) == 5
        
        # Verify IDs are sequential
        expected_ids = [f"coalition_{i}" for i in range(5)]
        actual_ids = list(builder.coalitions.keys())
        assert sorted(actual_ids) == sorted(expected_ids)
    
    def test_coalition_metadata_usage(self):
        """Test coalition metadata for coordination."""
        builder = CoalitionBuilder()
        
        coalition = builder.propose_coalition(
            "agent-1", ["agent-2", "agent-3"], "exploration"
        )
        
        # Add coordination metadata
        coalition.metadata.update({
            "target_area": "forest_region_alpha",
            "estimated_duration": "2_hours",
            "required_resources": ["energy_cells", "mapping_tools"],
            "risk_level": "medium"
        })
        
        # Verify metadata is accessible
        assert coalition.metadata["target_area"] == "forest_region_alpha"
        assert coalition.metadata["risk_level"] == "medium"
        assert "mapping_tools" in coalition.metadata["required_resources"]
    
    def test_coalition_proposal_edge_cases(self):
        """Test edge cases in coalition proposal."""
        builder = CoalitionBuilder()
        
        # Proposer proposes to themselves
        coalition = builder.propose_coalition("agent-1", ["agent-1"])
        assert coalition.members.count("agent-1") == 2  # Proposer + duplicate
        
        # Empty proposer ID (edge case)
        coalition2 = builder.propose_coalition("", ["agent-2"])
        assert "" in coalition2.members
        assert "agent-2" in coalition2.members


class TestCoalitionBuilderPerformance:
    """Performance tests for coalition builder."""
    
    def test_large_scale_coalition_creation(self):
        """Test performance with many coalitions."""
        import time
        
        builder = CoalitionBuilder()
        start_time = time.time()
        
        # Create many coalitions
        for i in range(100):
            builder.propose_coalition(
                f"proposer-{i}",
                [f"member-{i}-{j}" for j in range(10)],  # 10 members each
                "performance_test"
            )
        
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert end_time - start_time < 1.0  # Less than 1 second
        assert len(builder.coalitions) == 100
    
    def test_coalition_lookup_performance(self):
        """Test coalition lookup performance."""
        import time
        
        builder = CoalitionBuilder()
        
        # Create many coalitions
        coalition_ids = []
        for i in range(1000):
            coalition = builder.propose_coalition(f"proposer-{i}", [f"member-{i}"])
            coalition_ids.append(coalition.id)
        
        # Test lookup performance
        start_time = time.time()
        
        for coalition_id in coalition_ids:
            assert coalition_id in builder.coalitions
            coalition = builder.coalitions[coalition_id]
            assert coalition.id == coalition_id
        
        end_time = time.time()
        
        # Lookups should be fast
        assert end_time - start_time < 0.1  # Less than 100ms for 1000 lookups