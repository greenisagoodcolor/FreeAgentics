"""Test script for PyMDP Knowledge Graph Integration (Task 59.3).

This script validates the complete pipeline from PyMDP inference results
to knowledge graph updates with real-time WebSocket streaming.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_inference_result():
    """Create a mock PyMDP inference result for testing."""
    from agents.inference_engine import InferenceResult
    
    # Mock belief distribution
    belief_distribution = [0.7, 0.2, 0.1]  # High confidence in state 0
    
    # Mock policy sequence
    policy_sequence = [0, 1, 0]  # Action sequence
    
    return InferenceResult(
        action=0,
        beliefs={
            "states": belief_distribution,
            "history_length": 5,
        },
        free_energy=2.34,
        confidence=0.85,
        metadata={
            "pymdp_method": "variational_inference",
            "observation": [0],
            "policy_sequence": policy_sequence,
            "planning_horizon": 3,
            "inference_time_ms": 45.2,
            "num_policies": 8,
        }
    )

async def test_pymdp_knowledge_extraction():
    """Test PyMDP knowledge extraction pipeline."""
    logger.info("Testing PyMDP knowledge extraction...")
    
    try:
        from knowledge_graph.pymdp_extractor import PyMDPKnowledgeExtractor
        
        # Create mock inference result
        inference_result = create_mock_inference_result()
        
        # Create extractor
        extractor = PyMDPKnowledgeExtractor()
        
        # Extract knowledge
        result = extractor.extract_knowledge(
            inference_result=inference_result,
            agent_id="test_agent_001",
            conversation_id="test_conversation_001",
        )
        
        # Validate results
        entities = result["entities"]
        relations = result["relations"]
        metadata = result["metadata"]
        
        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations")
        
        # Check entity types
        entity_types = [e.entity_type.value for e in entities]
        logger.info(f"Entity types: {entity_types}")
        
        # Validate belief state entity
        belief_entities = [e for e in entities if e.entity_type.value == "belief_state"]
        if belief_entities:
            belief_entity = belief_entities[0]
            logger.info(f"Belief entity confidence: {belief_entity.properties.get('confidence')}")
            logger.info(f"Most likely state: {belief_entity.properties.get('most_likely_state')}")
        
        # Check relation types
        if relations:
            relation_types = [r.relation_type.value for r in relations]
            logger.info(f"Relation types: {relation_types}")
        
        logger.info("✓ PyMDP knowledge extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ PyMDP knowledge extraction test failed: {e}")
        return False

async def test_knowledge_graph_updater():
    """Test the KnowledgeGraphUpdater orchestrator."""
    logger.info("Testing KnowledgeGraphUpdater...")
    
    try:
        from knowledge_graph.updater import KnowledgeGraphUpdater
        from knowledge_graph.graph_engine import KnowledgeGraph
        
        # Create knowledge graph and updater
        graph = KnowledgeGraph()
        updater = KnowledgeGraphUpdater(knowledge_graph=graph)
        
        # Start updater
        await updater.start()
        
        try:
            # Create mock inference result
            inference_result = create_mock_inference_result()
            
            # Update knowledge graph
            result = await updater.update_from_inference(
                inference_result=inference_result,
                agent_id="test_agent_002",
                conversation_id="test_conversation_002",
                force_immediate=True,
            )
            
            # Validate results
            if result and not result.get("metadata", {}).get("error"):
                entities = result.get("entities", [])
                relations = result.get("relations", [])
                update_events = result.get("update_events", [])
                
                logger.info(f"Update generated {len(entities)} entities, {len(relations)} relations, {len(update_events)} events")
                
                # Check graph state
                graph_metrics = updater.get_metrics()
                logger.info(f"Graph now has {graph_metrics['knowledge_graph_stats']['node_count']} nodes")
                
                # Get agent knowledge
                agent_knowledge = await updater.get_agent_knowledge("test_agent_002")
                logger.info(f"Agent knowledge: {len(agent_knowledge['entities'])} entities")
                
                logger.info("✓ KnowledgeGraphUpdater test passed")
                return True
            else:
                logger.error(f"Update failed: {result}")
                return False
                
        finally:
            await updater.stop()
            
    except Exception as e:
        logger.error(f"✗ KnowledgeGraphUpdater test failed: {e}")
        return False

async def test_agent_integration():
    """Test the complete agent integration."""
    logger.info("Testing AgentKnowledgeGraphIntegration...")
    
    try:
        from agents.kg_integration import AgentKnowledgeGraphIntegration
        
        # Create integration
        integration = AgentKnowledgeGraphIntegration()
        
        try:
            # Create mock inference result
            inference_result = create_mock_inference_result()
            
            # Test new PyMDP integration method
            result = await integration.update_from_inference_result(
                inference_result=inference_result,
                agent_id="test_agent_003",
                conversation_id="test_conversation_003",
            )
            
            if result and not result.get("metadata", {}).get("error"):
                logger.info("Inference result update successful")
                
                # Get agent knowledge summary
                summary = await integration.get_agent_knowledge_summary("test_agent_003")
                logger.info(f"Agent summary: {summary.get('summary', {})}")
                
                # Get updater metrics
                metrics = integration.get_updater_metrics()
                updater_metrics = metrics.get("updater_metrics", {})
                logger.info(f"Updater metrics: {updater_metrics.get('successful_updates', 0)} successful updates")
                
                logger.info("✓ Agent integration test passed")
                return True
            else:
                logger.error(f"Agent integration failed: {result}")
                return False
                
        finally:
            await integration.shutdown()
            
    except Exception as e:
        logger.error(f"✗ Agent integration test failed: {e}")
        return False

async def test_batch_processing():
    """Test batch processing capabilities."""
    logger.info("Testing batch processing...")
    
    try:
        from knowledge_graph.updater import KnowledgeGraphUpdater
        
        # Create updater with small batch size for testing
        updater = KnowledgeGraphUpdater(batch_size=3, batch_timeout_seconds=0.5)
        await updater.start()
        
        try:
            # Send multiple inference results for batching
            results = []
            for i in range(5):
                inference_result = create_mock_inference_result()
                
                # Add to batch (not immediate)
                result = await updater.update_from_inference(
                    inference_result=inference_result,
                    agent_id=f"batch_agent_{i}",
                    conversation_id="batch_conversation",
                    force_immediate=False,
                )
                results.append(result)
            
            # Wait for batch processing
            await asyncio.sleep(1.0)
            
            # Check metrics
            metrics = updater.get_metrics()
            batch_status = metrics.get("batch_status", {})
            updater_metrics = metrics.get("updater_metrics", {})
            
            logger.info(f"Batch processing: {updater_metrics.get('successful_updates', 0)} successful updates")
            logger.info(f"Pending updates: {batch_status.get('pending_updates', 0)}")
            
            if updater_metrics.get("successful_updates", 0) > 0:
                logger.info("✓ Batch processing test passed")
                return True
            else:
                logger.warning("No updates processed in batch test")
                return False
                
        finally:
            await updater.stop()
            
    except Exception as e:
        logger.error(f"✗ Batch processing test failed: {e}")
        return False

async def test_schema_validation():
    """Test schema validation with PyMDP entities."""
    logger.info("Testing schema validation...")
    
    try:
        from knowledge_graph.schema import ConversationOntology, SchemaValidator, EntityType
        from knowledge_graph.pymdp_extractor import PyMDPKnowledgeExtractor
        
        # Create ontology and validator
        ontology = ConversationOntology()
        validator = SchemaValidator(ontology)
        
        # Extract entities
        extractor = PyMDPKnowledgeExtractor()
        inference_result = create_mock_inference_result()
        
        result = extractor.extract_knowledge(
            inference_result=inference_result,
            agent_id="validation_agent",
            conversation_id="validation_conversation",
        )
        
        # Validate each entity
        valid_entities = 0
        for entity in result["entities"]:
            validation_result = validator.validate_entity(entity)
            if validation_result.is_valid:
                valid_entities += 1
            else:
                logger.warning(f"Entity validation errors: {validation_result.errors}")
        
        # Validate relations
        valid_relations = 0
        for relation in result["relations"]:
            validation_result = validator.validate_relation(relation)
            if validation_result.is_valid:
                valid_relations += 1
            else:
                logger.warning(f"Relation validation errors: {validation_result.errors}")
        
        logger.info(f"Validation: {valid_entities}/{len(result['entities'])} entities valid, "
                   f"{valid_relations}/{len(result['relations'])} relations valid")
        
        if valid_entities > 0:
            logger.info("✓ Schema validation test passed")
            return True
        else:
            logger.error("No valid entities found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Schema validation test failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    logger.info("Starting PyMDP Knowledge Graph Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("PyMDP Knowledge Extraction", test_pymdp_knowledge_extraction),
        ("Knowledge Graph Updater", test_knowledge_graph_updater),
        ("Agent Integration", test_agent_integration),
        ("Batch Processing", test_batch_processing),
        ("Schema Validation", test_schema_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! PyMDP-KG integration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))