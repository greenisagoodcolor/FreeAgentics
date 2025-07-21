"""Example integration of LLM providers with FreeAgentics.

This script demonstrates how to:
1. Configure different LLM providers
2. Generate GMN specifications
3. Handle fallbacks and errors
4. Compare provider outputs
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llm import (
    LLMError,
    LLMMessage,
    LLMProviderFactory,
    LLMRole,
    ProviderType,
    create_llm_factory,
)


async def basic_usage_example():
    """Basic usage of LLM providers."""
    print("=== Basic LLM Provider Usage ===\n")

    # Create factory with auto-detection
    factory = create_llm_factory()

    # Check health
    health = await factory.health_check()
    print(f"Primary provider: {health['primary_provider']}")
    print(f"Available providers: {list(health['providers'].keys())}\n")

    # Generate a simple response
    messages = [
        LLMMessage(
            role=LLMRole.SYSTEM,
            content="You are an expert in active inference and GMN notation.",
        ),
        LLMMessage(role=LLMRole.USER, content="Explain what GMN is in one sentence."),
    ]

    try:
        response = await factory.generate(messages, temperature=0.7, max_tokens=100)
        print(f"Response from {response.metadata.get('provider')}:")
        print(f"{response.content}\n")
        print(f"Tokens used: {response.usage}\n")
    except LLMError as e:
        print(f"Error: {e}\n")


async def gmn_generation_example():
    """Example of GMN generation for different agent types."""
    print("=== GMN Generation Example ===\n")

    factory = create_llm_factory()

    # Define different agent scenarios
    scenarios = [
        {
            "prompt": "Create an agent that explores a 5x5 grid world, avoiding walls and searching for treasure",
            "agent_type": "explorer",
            "name": "Treasure Hunter",
        },
        {
            "prompt": "Create a trading agent that monitors price trends and makes buy/sell decisions to maximize profit",
            "agent_type": "trader",
            "name": "Market Trader",
        },
        {
            "prompt": "Create a coordinator agent that assigns tasks to a team of 3 worker agents",
            "agent_type": "coordinator",
            "name": "Team Manager",
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} Agent ---")
        print(f"Prompt: {scenario['prompt']}")

        try:
            gmn = await factory.generate_gmn(
                prompt=scenario["prompt"],
                agent_type=scenario["agent_type"],
                constraints={"max_states": 10, "max_actions": 5},
            )

            print(f"\nGenerated GMN:\n{gmn}\n")

            # Validate the GMN
            provider = await factory.get_provider()
            is_valid, errors = await provider.validate_gmn(gmn)

            if is_valid:
                print("✓ GMN validation passed")
            else:
                print("✗ GMN validation failed:")
                for error in errors:
                    print(f"  - {error}")

        except Exception as e:
            print(f"Error generating GMN: {e}")


async def provider_comparison_example():
    """Compare outputs from different providers."""
    print("=== Provider Comparison Example ===\n")

    prompt = "Create a simple agent that can move in 4 directions on a grid"

    # Test with each available provider
    providers_to_test = []

    # Always test mock
    providers_to_test.append(ProviderType.MOCK)

    # Add real providers if available
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(ProviderType.OPENAI)

    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(ProviderType.ANTHROPIC)

    # Check if Ollama is available
    from llm.providers import OllamaProvider

    ollama = OllamaProvider()
    if await ollama._check_ollama_running():
        providers_to_test.append(ProviderType.OLLAMA)
    await ollama.close()

    results = {}

    for provider_type in providers_to_test:
        print(f"\nTesting {provider_type.value} provider...")

        config = {"provider": provider_type.value}
        factory = LLMProviderFactory(config)

        try:
            gmn = await factory.generate_gmn(prompt, agent_type="explorer")

            # Extract key metrics
            results[provider_type.value] = {
                "length": len(gmn),
                "nodes": gmn.count("node"),
                "states": gmn.count("node state"),
                "actions": gmn.count("node action"),
                "has_preferences": "preference" in gmn,
            }

            print(f"✓ Generated GMN with {results[provider_type.value]['nodes']} nodes")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results[provider_type.value] = {"error": str(e)}

    # Summary
    print("\n--- Comparison Summary ---")
    print(json.dumps(results, indent=2))


async def advanced_configuration_example():
    """Example of advanced provider configuration."""
    print("=== Advanced Configuration Example ===\n")

    # Custom configuration
    config = {
        "provider": "auto",
        "openai": {"model": "gpt-4o", "temperature": 0.3, "max_retries": 5},
        "anthropic": {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.2,
        },
        "ollama": {
            "model": "mixtral",
            "num_ctx": 8192,
            "num_gpu": -1,  # Use all GPU layers
        },
    }

    factory = LLMProviderFactory(config)

    # Generate with specific requirements
    prompt = """Create a sophisticated multi-agent system where:
    1. Explorer agents map unknown territory
    2. Analyzer agents process discoveries
    3. A coordinator manages the team
    Include communication between agents."""

    try:
        gmn = await factory.generate_gmn(
            prompt=prompt,
            agent_type="coordinator",
            examples=["node communication c1 { type: message, size: 10 }"],
        )

        print("Generated complex multi-agent GMN:")
        print(gmn[:500] + "..." if len(gmn) > 500 else gmn)

    except Exception as e:
        print(f"Error: {e}")


async def error_handling_example():
    """Example of error handling and fallbacks."""
    print("=== Error Handling Example ===\n")

    # Configure with intentionally bad primary provider
    config = {
        "provider": "openai",
        "openai": {
            "api_key": "invalid-key-to-trigger-fallback",
            "model": "gpt-4",
        },
    }

    factory = LLMProviderFactory(config)

    print("Attempting generation with invalid API key...")

    messages = [LLMMessage(role=LLMRole.USER, content="Generate a simple GMN")]

    try:
        response = await factory.generate(messages)
        print(f"Success! Fell back to: {response.metadata.get('provider')}")
        print(f"Response: {response.content[:200]}...")

    except LLMError as e:
        print(f"All providers failed: {e}")

    # Check health after failure
    health = await factory.health_check()

    print("\nProvider health after error:")
    for provider, status in health["providers"].items():
        if "healthy" in status:
            print(
                f"  {provider}: {'✓' if status['healthy'] else '✗'} "
                f"(success rate: {status.get('success_rate', 0):.1f}%)"
            )


async def interactive_gmn_refinement():
    """Interactive GMN refinement example."""
    print("=== Interactive GMN Refinement ===\n")

    factory = create_llm_factory()

    # Initial GMN
    initial_prompt = "Create a basic agent that can sense its environment"

    print(f"Initial prompt: {initial_prompt}")
    gmn = await factory.generate_gmn(initial_prompt)

    print(f"\nInitial GMN:\n{gmn}\n")

    # Refinement loop
    refinements = [
        "Add curiosity-driven exploration preferences",
        "Include memory of visited locations",
        "Add ability to mark interesting discoveries",
    ]

    provider = await factory.get_provider()

    for refinement in refinements:
        print(f"\nRefinement: {refinement}")

        gmn = await provider.refine_gmn(gmn, refinement)
        print(f"Updated GMN (first 300 chars):\n{gmn[:300]}...\n")

    print("Final GMN after all refinements:")
    print(gmn)


async def main():
    """Run all examples."""
    examples = [
        ("Basic Usage", basic_usage_example),
        ("GMN Generation", gmn_generation_example),
        ("Provider Comparison", provider_comparison_example),
        ("Advanced Configuration", advanced_configuration_example),
        ("Error Handling", error_handling_example),
        ("Interactive Refinement", interactive_gmn_refinement),
    ]

    print("FreeAgentics LLM Integration Examples")
    print("=" * 50)
    print()

    for name, example_func in examples:
        print(f"\n{'=' * 50}")
        print(f"Running: {name}")
        print("=" * 50 + "\n")

        try:
            await example_func()
        except Exception as e:
            print(f"Example failed: {e}")

        print("\nPress Enter to continue...")
        input()

    print("\nAll examples completed!")


if __name__ == "__main__":
    # Set up any necessary environment variables for testing
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

    asyncio.run(main())
