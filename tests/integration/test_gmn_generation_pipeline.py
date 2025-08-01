"""Integration tests for the complete GMN generation pipeline.

Tests the end-to-end flow from natural language prompt to stored GMN specification.
"""

import uuid
from typing import Dict, Any

import pytest
from sqlalchemy.orm import Session

from database.gmn_versioned_models import GMNVersionedSpecification
from database.gmn_versioned_repository import GMNVersionedRepository
from services.gmn_generator import GMNGenerator
from llm.providers.mock import MockLLMProvider


class TestGMNGenerationPipeline:
    """Test the complete GMN generation pipeline."""

    @pytest.fixture
    def db_session(self, test_db):
        """Create a test database session."""
        return test_db.session

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        return MockLLMProvider()

    @pytest.fixture
    def gmn_generator(self, mock_llm_provider):
        """Create GMN generator with mock provider."""
        return GMNGenerator(llm_provider=mock_llm_provider)

    @pytest.fixture
    def gmn_repository(self, db_session):
        """Create GMN repository."""
        return GMNVersionedRepository(db_session)

    @pytest.fixture
    def sample_prompt(self):
        """Sample natural language prompt."""
        return "Create an explorer agent that navigates a 4x4 grid world, avoiding obstacles and seeking rewards"

    async def test_complete_generation_pipeline(
        self, gmn_generator, gmn_repository, sample_prompt
    ):
        """Test the complete pipeline from prompt to storage."""
        # Step 1: Generate GMN from natural language
        agent_id = uuid.uuid4()
        
        gmn_spec = await gmn_generator.prompt_to_gmn(
            prompt=sample_prompt,
            agent_type="explorer"
        )
        
        # Verify GMN was generated
        assert gmn_spec is not None
        assert len(gmn_spec.strip()) > 0
        assert "node" in gmn_spec
        
        # Step 2: Validate the generated GMN
        is_valid, errors = await gmn_generator.validate_gmn(gmn_spec)
        
        # Verify validation works
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # Step 3: Get improvement suggestions
        suggestions = await gmn_generator.suggest_improvements(gmn_spec)
        
        # Verify suggestions are provided
        assert isinstance(suggestions, list)
        
        # Step 4: Parse GMN for storage
        parsed_data = self._parse_gmn_for_storage(gmn_spec)
        
        # Step 5: Store with versioning
        stored_spec = gmn_repository.create_gmn_specification_versioned(
            agent_id=agent_id,
            specification=gmn_spec,
            name="ExplorerAgent_Test",
            parsed_data=parsed_data,
            version_metadata={
                "agent_type": "explorer",
                "original_prompt": sample_prompt,
                "validation_status": "valid" if is_valid else "invalid",
                "suggestions_count": len(suggestions)
            }
        )
        
        # Verify storage was successful
        assert stored_spec is not None
        assert stored_spec.id is not None
        assert stored_spec.agent_id == agent_id
        assert stored_spec.version_number == 1
        assert stored_spec.specification_text == gmn_spec
        assert stored_spec.name == "ExplorerAgent_Test"
        
        # Verify metadata was stored
        metadata = stored_spec.version_metadata
        assert metadata["agent_type"] == "explorer"
        assert metadata["original_prompt"] == sample_prompt
        assert "validation_status" in metadata
        
        # Step 6: Verify retrieval works
        retrieved_spec = gmn_repository.get_specification_by_id(stored_spec.id)
        assert retrieved_spec is not None
        assert retrieved_spec.specification_text == gmn_spec

    async def test_versioning_pipeline(
        self, gmn_generator, gmn_repository, sample_prompt
    ):
        """Test GMN versioning through refinement."""
        agent_id = uuid.uuid4()
        
        # Generate initial GMN
        initial_gmn = await gmn_generator.prompt_to_gmn(
            prompt=sample_prompt,
            agent_type="explorer"
        )
        
        # Store initial version
        initial_spec = gmn_repository.create_gmn_specification_versioned(
            agent_id=agent_id,
            specification=initial_gmn,
            name="ExplorerAgent_v1",
            parsed_data=self._parse_gmn_for_storage(initial_gmn)
        )
        
        assert initial_spec.version_number == 1
        
        # Refine the GMN
        refined_gmn = await gmn_generator.refine_gmn(
            initial_gmn,
            "Add preference nodes to define exploration goals"
        )
        
        # Create new version from refined GMN
        refined_spec = gmn_repository.create_new_version(
            parent_specification_id=initial_spec.id,
            specification=refined_gmn,
            parsed_data=self._parse_gmn_for_storage(refined_gmn),
            version_metadata={
                "refinement_feedback": "Add preference nodes to define exploration goals",
                "parent_version": initial_spec.version_number
            }
        )
        
        # Verify versioning
        assert refined_spec.version_number == 2
        assert refined_spec.parent_version_id == initial_spec.id
        assert refined_spec.agent_id == agent_id
        
        # Verify lineage
        lineage = gmn_repository.get_version_lineage(agent_id)
        assert len(lineage) == 2
        
        # Find the parent and child in lineage
        parent_entry = next(entry for entry in lineage if entry["version_number"] == 1)
        child_entry = next(entry for entry in lineage if entry["version_number"] == 2)
        
        assert str(refined_spec.id) in parent_entry["children"]
        assert child_entry["parent_version_id"] == str(initial_spec.id)

    async def test_validation_and_error_handling(
        self, gmn_generator, gmn_repository
    ):
        """Test validation and error handling in the pipeline."""
        agent_id = uuid.uuid4()
        
        # Test with a minimal prompt that might generate invalid GMN
        minimal_prompt = "agent"
        
        try:
            # This should either generate valid GMN or raise appropriate error
            gmn_spec = await gmn_generator.prompt_to_gmn(
                prompt=minimal_prompt,
                agent_type="general"
            )
            
            # If GMN was generated, validate it
            is_valid, errors = await gmn_generator.validate_gmn(gmn_spec)
            
            # Store even if invalid for testing
            stored_spec = gmn_repository.create_gmn_specification_versioned(
                agent_id=agent_id,
                specification=gmn_spec,
                name="MinimalAgent_Test",
                parsed_data=self._parse_gmn_for_storage(gmn_spec),
                version_metadata={
                    "validation_status": "valid" if is_valid else "invalid",
                    "validation_errors": errors
                }
            )
            
            # Verify the error information was stored
            if not is_valid:
                assert stored_spec.version_metadata["validation_status"] == "invalid"
                assert len(stored_spec.version_metadata["validation_errors"]) > 0
            
        except ValueError as e:
            # Expected for minimal prompt
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()

    async def test_concurrent_generation_pipeline(
        self, gmn_generator, gmn_repository
    ):
        """Test concurrent GMN generation and storage."""
        import asyncio
        
        agent_ids = [uuid.uuid4() for _ in range(3)]
        prompts = [
            "Create an explorer agent for maze navigation",
            "Create a trader agent for stock market analysis", 
            "Create a coordinator agent for task delegation"
        ]
        agent_types = ["explorer", "trader", "coordinator"]
        
        async def generate_and_store(agent_id, prompt, agent_type):
            """Generate and store a single GMN."""
            # Generate GMN
            gmn_spec = await gmn_generator.prompt_to_gmn(
                prompt=prompt,
                agent_type=agent_type
            )
            
            # Store GMN
            stored_spec = gmn_repository.create_gmn_specification_versioned(
                agent_id=agent_id,
                specification=gmn_spec,
                name=f"{agent_type}_agent_{agent_id}",
                parsed_data=self._parse_gmn_for_storage(gmn_spec),
                version_metadata={
                    "agent_type": agent_type,
                    "original_prompt": prompt
                }
            )
            
            return stored_spec
        
        # Run concurrent generation
        tasks = [
            generate_and_store(agent_id, prompt, agent_type)
            for agent_id, prompt, agent_type in zip(agent_ids, prompts, agent_types)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all generations succeeded
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.agent_id == agent_ids[i]
            assert result.version_number == 1
            assert agent_types[i] in result.version_metadata["agent_type"]

    async def test_rollback_pipeline(
        self, gmn_generator, gmn_repository
    ):
        """Test the rollback functionality in the pipeline."""
        agent_id = uuid.uuid4()
        
        # Create initial version
        initial_gmn = await gmn_generator.prompt_to_gmn(
            prompt="Create a simple navigation agent",
            agent_type="explorer"
        )
        
        initial_spec = gmn_repository.create_gmn_specification_versioned(
            agent_id=agent_id,
            specification=initial_gmn,
            name="NavigationAgent_v1",
            parsed_data=self._parse_gmn_for_storage(initial_gmn)
        )
        
        # Create a problematic second version
        problematic_gmn = await gmn_generator.refine_gmn(
            initial_gmn,
            "Make this agent overly complex and hard to understand"
        )
        
        problematic_spec = gmn_repository.create_new_version(
            parent_specification_id=initial_spec.id,
            specification=problematic_gmn,
            parsed_data=self._parse_gmn_for_storage(problematic_gmn)
        )
        
        # Rollback to initial version
        rollback_success = gmn_repository.rollback_to_version(
            agent_id=agent_id,
            target_version_id=initial_spec.id,
            rollback_reason="Version 2 was too complex and performed poorly"
        )
        
        assert rollback_success is True
        
        # Verify rollback worked
        active_spec = gmn_repository.get_active_specification(agent_id)
        assert active_spec.id == initial_spec.id
        assert active_spec.version_number == 1

    def _parse_gmn_for_storage(self, gmn_spec: str) -> Dict[str, Any]:
        """Parse GMN specification for storage.
        
        Args:
            gmn_spec: GMN specification string
            
        Returns:
            Dictionary with parsed information
        """
        # Simple parsing for testing
        lines = gmn_spec.strip().split('\n')
        nodes = []
        edges = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('node '):
                parts = line.split()
                if len(parts) >= 3:
                    node_info = {
                        "name": parts[2].rstrip('{'),
                        "type": parts[1]
                    }
                    nodes.append(node_info)
            elif 'from:' in line or 'to:' in line:
                edges.append({"edge": line.strip()})
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }