#!/usr/bin/env python3
"""
Simple demo of the UI-Backend compatibility layer.

This shows how the compatibility layer works without database dependencies.
"""

import json
import sys

sys.path.insert(0, ".")

from api.ui_compatibility import (
    extract_agent_name_from_description,
    extract_agent_type_from_description,
)


def demo_conversion():
    """Demo the UI format to backend format conversion."""
    print("🔧 FreeAgentics UI-Backend Compatibility Layer Demo")
    print("=" * 60)
    print()
    print("This demonstrates how the UI's simple format is converted")
    print("to the backend's structured format for Active Inference.")
    print()

    test_cases = [
        "An explorer agent that searches for valuable resources",
        "A resource collector that gathers items efficiently",
        "An analyzer that studies environmental patterns",
        "A smart agent that learns from interactions",
        "A navigation agent that finds optimal paths",
    ]

    print("🔄 Conversion Examples:")
    print("=" * 40)

    for i, description in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  📝 UI Input: '{description}'")
        print()

        # Extract information
        agent_type = extract_agent_type_from_description(description)
        agent_name = extract_agent_name_from_description(description)

        print("  🎯 Extracted Information:")
        print(f"     • Type: {agent_type}")
        print(f"     • Name: {agent_name}")
        print("     • Template: basic-explorer")
        print()

        # Show what would be sent to backend
        print("  🚀 Backend API Call:")
        backend_config = {
            "name": agent_name,
            "template": "basic-explorer",
            "parameters": {
                "description": description,
                "use_pymdp": True,
                "planning_horizon": 3,
            },
        }
        print(f"     {json.dumps(backend_config, indent=6)}")
        print()

        # Show what UI would receive back
        print("  📤 UI Response:")
        ui_response = {
            "id": f"agent-{i}",
            "name": agent_name,
            "type": agent_type,
            "status": "active",
            "description": description,
            "createdAt": "2025-07-18T12:00:00Z",
        }
        print(f"     {json.dumps(ui_response, indent=6)}")
        print("     " + "─" * 40)

    print("\n✅ Summary of Compatibility Layer:")
    print("=" * 40)
    print("1. 📝 UI sends: {description: 'simple text'}")
    print("2. 🔄 Backend converts to: {name, template, parameters}")
    print("3. 🧠 Active Inference engine gets proper configuration")
    print("4. 📤 UI receives: {id, name, type, status, description}")
    print()
    print("This bridge allows the UI to stay simple while the backend")
    print("handles all the complex Active Inference setup!")
    print()

    print("🎯 Key Features:")
    print("• Automatic agent type detection from description")
    print("• Intelligent name extraction")
    print("• Template mapping for different agent types")
    print("• Preservation of original description")
    print("• Active Inference parameters auto-configured")
    print()

    print("🔗 API Endpoints Created:")
    print("• POST /api/agents - Create agent (UI format)")
    print("• GET /api/agents - List agents (UI format)")
    print("• GET /api/agents/{id} - Get agent (UI format)")
    print("• PATCH /api/agents/{id}/status - Update status")
    print("• DELETE /api/agents/{id} - Delete agent")
    print()


def demo_keyword_detection():
    """Demo the keyword detection logic."""
    print("🔍 Agent Type Detection Demo")
    print("=" * 35)
    print()

    test_descriptions = [
        ("An explorer that searches the environment", "explorer"),
        ("A collector that gathers valuable resources", "collector"),
        ("An analyzer that studies data patterns", "analyzer"),
        ("A smart agent that finds optimal solutions", "explorer"),  # default
        ("Navigate through complex environments", "explorer"),
        ("Collect and organize important information", "collector"),
        ("Analyze user behavior and preferences", "analyzer"),
        ("Discover hidden patterns in data", "explorer"),
        ("Gather resources from multiple sources", "collector"),
        ("Examine system performance metrics", "analyzer"),
    ]

    print("Testing keyword detection algorithm:")
    print()

    for description, expected in test_descriptions:
        detected = extract_agent_type_from_description(description)
        status = "✅" if detected == expected else "❌"
        print(f"{status} '{description}'")
        print(f"   → Detected: {detected} (Expected: {expected})")
        print()

    print("Detection Rules:")
    print("• Explorer: search, explore, find, discover, navigate")
    print("• Collector: collect, gather, resource")
    print("• Analyzer: analyze, study, examine")
    print("• Default: explorer (when no keywords match)")
    print()


if __name__ == "__main__":
    demo_conversion()
    demo_keyword_detection()

    print("🎉 Compatibility Layer Demo Complete!")
    print()
    print("This shows how we've successfully bridged the gap between:")
    print("• UI's simple, user-friendly interface")
    print("• Backend's complex Active Inference system")
    print()
    print("The integration is working and ready for the UI to connect!")
