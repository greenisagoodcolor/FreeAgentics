"""Demo: PyMDP Knowledge Graph Integration in Action (Task 59.3).

This demo shows the complete pipeline from PyMDP agent inference 
to real-time knowledge graph updates with WebSocket streaming.
"""

import asyncio
import json
import logging
import time
from datetime import datetime

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_pymdp_kg_pipeline():
    """Demonstrate the complete PyMDP-KG integration pipeline."""
    
    print("🚀 PyMDP Knowledge Graph Integration Demo")
    print("=" * 60)
    
    # Import components
    from agents.inference_engine import InferenceResult
    from knowledge_graph.updater import KnowledgeGraphUpdater
    from agents.kg_integration import AgentKnowledgeGraphIntegration
    
    # Create mock inference results for different scenarios
    def create_high_confidence_inference():
        return InferenceResult(
            action=1,
            beliefs={"states": [0.1, 0.8, 0.1], "history_length": 3},
            free_energy=1.23,
            confidence=0.9,
            metadata={
                "pymdp_method": "variational_inference",
                "observation": [1],
                "policy_sequence": [1, 0, 1],
                "planning_horizon": 3,
                "inference_time_ms": 23.4,
                "num_policies": 4,
            }
        )
    
    def create_uncertain_inference():
        return InferenceResult(
            action=0,
            beliefs={"states": [0.4, 0.3, 0.3], "history_length": 5},
            free_energy=3.45,
            confidence=0.4,
            metadata={
                "pymdp_method": "message_passing",
                "observation": [0],
                "policy_sequence": [0, 0, 1, 0],
                "planning_horizon": 4,
                "inference_time_ms": 67.8,
                "num_policies": 8,
            }
        )
    
    # Initialize agent integration
    print("\n📊 Initializing Agent Knowledge Graph Integration...")
    integration = AgentKnowledgeGraphIntegration()
    
    try:
        # Scenario 1: High-confidence inference
        print("\n🎯 Scenario 1: High-Confidence Inference")
        print("-" * 40)
        
        high_conf_result = create_high_confidence_inference()
        print(f"Inference: action={high_conf_result.action}, confidence={high_conf_result.confidence}")
        
        kg_result = await integration.update_from_inference_result(
            inference_result=high_conf_result,
            agent_id="demo_agent_alpha",
            conversation_id="demo_conversation_001",
            message_id="msg_001"
        )
        
        if kg_result:
            entities = kg_result.get("entities", [])
            relations = kg_result.get("relations", [])
            events = kg_result.get("update_events", [])
            
            print(f"✓ Generated {len(entities)} entities, {len(relations)} relations, {len(events)} events")
            
            # Show entity details
            for entity in entities:
                entity_type = entity.entity_type.value
                confidence = entity.properties.get('confidence', 'N/A')
                print(f"  - {entity_type}: confidence={confidence}")
        
        # Wait a bit for real-time processing
        await asyncio.sleep(0.1)
        
        # Scenario 2: Uncertain inference
        print("\n🤔 Scenario 2: Uncertain Inference")
        print("-" * 40)
        
        uncertain_result = create_uncertain_inference()
        print(f"Inference: action={uncertain_result.action}, confidence={uncertain_result.confidence}")
        
        kg_result = await integration.update_from_inference_result(
            inference_result=uncertain_result,
            agent_id="demo_agent_alpha",
            conversation_id="demo_conversation_001",
            message_id="msg_002"
        )
        
        if kg_result:
            entities = kg_result.get("entities", [])
            relations = kg_result.get("relations", [])
            
            print(f"✓ Generated {len(entities)} entities, {len(relations)} relations")
            
            # Show temporal relationships
            temporal_rels = [r for r in relations if r.relation_type.value == "temporal_sequence"]
            print(f"  - Temporal sequences: {len(temporal_rels)}")
        
        # Scenario 3: Multiple agents interaction
        print("\n🤖 Scenario 3: Multi-Agent Knowledge Building")
        print("-" * 50)
        
        agent_results = []
        for i, agent_id in enumerate(["agent_beta", "agent_gamma", "agent_delta"]):
            # Create varied inference results
            beliefs = [0.2 + 0.2*i, 0.6 - 0.1*i, 0.2 - 0.1*i]
            beliefs = [max(0.01, b) for b in beliefs]  # Ensure positive
            beliefs = [b/sum(beliefs) for b in beliefs]  # Normalize
            
            result = InferenceResult(
                action=i % 3,
                beliefs={"states": beliefs, "history_length": 2 + i},
                free_energy=2.0 + i * 0.5,
                confidence=0.6 + i * 0.1,
                metadata={
                    "pymdp_method": "active_inference",
                    "observation": [i % 2],
                    "policy_sequence": [i % 3, (i+1) % 3],
                    "planning_horizon": 2,
                    "inference_time_ms": 30.0 + i * 10,
                    "agent_interaction": True,
                }
            )
            
            kg_result = await integration.update_from_inference_result(
                inference_result=result,
                agent_id=agent_id,
                conversation_id="multi_agent_conversation",
                message_id=f"msg_multi_{i}"
            )
            
            agent_results.append((agent_id, kg_result))
            print(f"  {agent_id}: {len(kg_result.get('entities', []))} entities")
        
        # Get agent knowledge summaries
        print("\n📈 Agent Knowledge Summaries")
        print("-" * 30)
        
        for agent_id in ["demo_agent_alpha", "agent_beta", "agent_gamma", "agent_delta"]:
            summary = await integration.get_agent_knowledge_summary(agent_id, limit=10)
            
            entity_count = summary.get('summary', {}).get('entity_count', 0)
            relation_count = summary.get('summary', {}).get('relation_count', 0)
            entity_types = summary.get('summary', {}).get('entity_types', [])
            
            print(f"  {agent_id}: {entity_count} entities, {relation_count} relations")
            if entity_types:
                print(f"    Types: {', '.join(entity_types)}")
        
        # Show updater metrics
        print("\n📊 System Metrics")
        print("-" * 20)
        
        metrics = integration.get_updater_metrics()
        updater_metrics = metrics.get("updater_metrics", {})
        graph_stats = metrics.get("knowledge_graph_stats", {})
        
        print(f"Total updates: {updater_metrics.get('successful_updates', 0)}")
        print(f"Success rate: {updater_metrics.get('success_rate', 0):.2%}")
        print(f"Avg processing time: {updater_metrics.get('avg_processing_time_ms', 0):.2f}ms")
        print(f"Graph nodes: {graph_stats.get('node_count', 0)}")
        print(f"Graph edges: {graph_stats.get('edge_count', 0)}")
        
        # Demo batch processing
        print("\n⚡ Batch Processing Demo")
        print("-" * 25)
        
        batch_updater = KnowledgeGraphUpdater(batch_size=3, batch_timeout_seconds=0.2)
        await batch_updater.start()
        
        try:
            # Send rapid updates for batching
            batch_tasks = []
            for i in range(7):
                inference = InferenceResult(
                    action=i % 2,
                    beliefs={"states": [0.5, 0.3, 0.2], "history_length": 1},
                    free_energy=1.5,
                    confidence=0.7,
                    metadata={"batch_demo": True, "index": i}
                )
                
                # Non-immediate updates for batching
                task = batch_updater.update_from_inference(
                    inference_result=inference,
                    agent_id=f"batch_agent_{i}",
                    conversation_id="batch_demo",
                    force_immediate=False
                )
                batch_tasks.append(task)
            
            # Wait for all updates
            await asyncio.gather(*batch_tasks)
            await asyncio.sleep(0.5)  # Allow batch processing
            
            batch_metrics = batch_updater.get_metrics()
            batch_updater_metrics = batch_metrics.get("updater_metrics", {})
            
            print(f"Batch updates processed: {batch_updater_metrics.get('successful_updates', 0)}")
            print(f"Batch avg time: {batch_updater_metrics.get('avg_processing_time_ms', 0):.2f}ms")
            
        finally:
            await batch_updater.stop()
        
        # Show real-time update events
        print("\n🔄 Real-Time Event Stream Sample")
        print("-" * 35)
        
        # Get a recent update with events
        if kg_result and kg_result.get("update_events"):
            for event in kg_result["update_events"][:3]:  # Show first 3 events
                event_type = event.get("event_type", "unknown")
                timestamp = event.get("timestamp", "")
                entity_id = event.get("entity_id", "N/A")
                
                print(f"  {event_type}: {entity_id} at {timestamp}")
        
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        
        print("\n✨ Key Features Demonstrated:")
        print("• Real-time knowledge extraction from PyMDP inference results")
        print("• Entity creation for beliefs, policies, and free energy landscapes")  
        print("• Relationship inference from belief transitions")
        print("• Confidence-weighted edge creation")
        print("• Batch processing for high-frequency updates")
        print("• Real-time WebSocket event streaming")
        print("• Multi-agent knowledge graph building")
        print("• Comprehensive monitoring and metrics")
        
        print(f"\n🏆 Successfully processed knowledge for multiple agents")
        print(f"   with full PyMDP semantic understanding!")
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        
    finally:
        # Clean shutdown
        await integration.shutdown()
        print("\n🔄 Integration shutdown complete")

if __name__ == "__main__":
    asyncio.run(demo_pymdp_kg_pipeline())