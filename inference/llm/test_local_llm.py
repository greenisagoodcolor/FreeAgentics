#!/usr/bin/env python3
"""
Test script for Local LLM Integration

Tests the local LLM functionality including Ollama integration,
quantization, and fallback mechanisms.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llm.fallback_mechanisms import FallbackManager, ResourceConstraints
from llm.local_llm_manager import (
    LocalLLMConfig,
    LocalLLMManager,
    LocalLLMProvider,
    QuantizationLevel,
)
from llm.model_quantization import EdgeOptimizer
from llm.ollama_integration import OllamaAgentAdapter, OllamaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_ollama_integration():
    """Test Ollama integration"""
    print("\n=== Testing Ollama Integration ===")

    manager = OllamaManager()

    # Check if Ollama is installed
    is_installed = await manager.is_installed()
    print(f"Ollama installed: {is_installed}")

    if not is_installed:
        print("Attempting to install Ollama...")
        success = await manager.install_ollama()
        print(f"Installation {'successful' if success else 'failed'}")

        if not success:
            print("Please install Ollama manually from https://ollama.ai")
            return

    # List available models
    models = await manager.list_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models:
        print(f"  - {model.name} ({model.size_gb:.2f}GB, {model.quantization})")

    # Recommend model based on hardware
    import psutil

    ram_gb = psutil.virtual_memory().total / (1024**3)
    storage_gb = psutil.disk_usage("/").free / (1024**3)

    recommended = manager.recommend_model(ram_gb, storage_gb, "general")
    print(f"\nRecommended model for this system: {recommended}")

    # Test generation if a model is available
    if models:
        model_name = models[0].name
        print(f"\nTesting generation with {model_name}...")

        prompt = "Hello! What's your purpose?"
        response_parts = []

        async for chunk in manager.generate(model_name, prompt, stream=True):
            response_parts.append(chunk)
            print(chunk, end="", flush=True)

        print("\n")
        return True
    else:
        print("\nNo models available. Pull a model with: ollama pull llama2")
        return False


def test_local_llm_manager() -> None:
    """Test Local LLM Manager"""
    print("\n=== Testing Local LLM Manager ===")

    # Create configuration
    config = LocalLLMConfig(
        provider=LocalLLMProvider.OLLAMA,
        model_name="llama2:7b-q4_K_M",
        quantization=QuantizationLevel.INT4,
        context_size=2048,
        max_tokens=256,
        threads=4,
        cache_size_mb=50,
        enable_fallback=True,
    )

    # Initialize manager
    manager = LocalLLMManager(config)

    # Get status
    status = manager.get_status()
    print(f"Manager status: {json.dumps(status, indent=2)}")

    # Test generation
    try:
        print("\nTesting generation...")
        response = manager.generate("What are the benefits of edge computing?", temperature=0.7)

        print(f"Response: {response.text}")
        print(f"Provider: {response.provider}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Time: {response.generation_time:.2f}s")
        print(f"Cached: {response.cached}")
        print(f"Fallback: {response.fallback_used}")

    except Exception as e:
        print(f"Generation failed: {e}")

        # Test fallback
        print("\nTesting fallback response...")
        response = manager.generate("Hello!")
        print(f"Fallback response: {response.text}")
        print(f"Fallback used: {response.fallback_used}")

    # Test hardware optimization
    print("\nTesting hardware optimization...")
    import psutil

    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = psutil.cpu_count()
    has_gpu = False  # Would need to detect properly

    optimized_config = manager.optimize_for_hardware(ram_gb, cpu_cores, has_gpu)
    print(f"Optimized config for {ram_gb:.1f}GB RAM, {cpu_cores} cores:")
    print(f"  Quantization: {optimized_config.quantization.value}")
    print(f"  Context size: {optimized_config.context_size}")
    print(f"  Threads: {optimized_config.threads}")
    print(f"  Cache size: {optimized_config.cache_size_mb}MB")


def test_edge_optimization() -> None:
    """Test edge device optimization"""
    print("\n=== Testing Edge Optimization ===")

    optimizer = EdgeOptimizer()

    # Test optimization for different devices
    devices = ["raspberry_pi", "jetson_nano", "mobile_phone"]

    # Create a dummy model file for testing
    dummy_model = Path("/tmp/test_model.bin")
    dummy_model.write_bytes(b"dummy model data" * 1000000)  # ~16MB

    output_dir = Path("/tmp/edge_optimized")
    output_dir.mkdir(exist_ok=True)

    for device in devices:
        print(f"\nOptimizing for {device}...")

        try:
            results = optimizer.optimize_for_device(dummy_model, device, output_dir)

            print(f"Results: {json.dumps(results, indent=2)}")

            # Check deployment package
            package_dir = output_dir / f"{device}_deployment"
            if package_dir.exists():
                print(f"Deployment package created at: {package_dir}")
                print(f"  Contents: {list(package_dir.iterdir())}")

        except Exception as e:
            print(f"Optimization failed: {e}")

    # Cleanup
    dummy_model.unlink()


def test_fallback_mechanisms() -> None:
    """Test fallback mechanisms"""
    print("\n=== Testing Fallback Mechanisms ===")

    cache_dir = Path("/tmp/fallback_cache")
    cache_dir.mkdir(exist_ok=True)

    manager = FallbackManager(cache_dir)

    # Test with different resource constraints
    test_cases = [
        {
            "name": "High resources",
            "constraints": ResourceConstraints(
                available_memory_mb=1000,
                cpu_usage_percent=20,
                battery_level=80,
                network_available=True,
            ),
        },
        {
            "name": "Medium resources",
            "constraints": ResourceConstraints(
                available_memory_mb=200,
                cpu_usage_percent=60,
                battery_level=50,
                network_available=True,
            ),
        },
        {
            "name": "Low resources",
            "constraints": ResourceConstraints(
                available_memory_mb=50,
                cpu_usage_percent=90,
                battery_level=10,
                network_available=False,
            ),
        },
    ]

    # Test prompts
    prompts = [
        "Hello! How are you?",
        "I need to explore the northern region for resources.",
        "Can we trade 10 food for 5 energy?",
        "Move to coordinates (10, 20).",
        "What resources do I currently have?",
    ]

    context = {
        "agent_name": "Explorer-1",
        "current_location": "(5, 5)",
        "resources": {"food": 20, "water": 15, "energy": 10},
    }

    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        constraints = test_case["constraints"]

        for prompt in prompts[:2]:  # Test first 2 prompts
            response = manager.generate_response(prompt, context, constraints)

            print(f"\nPrompt: {prompt}")
            print(f"Response: {response.text}")
            print(f"Fallback level: {response.fallback_level.name}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Time: {response.computation_time_ms:.1f}ms")

    # Show performance stats
    stats = manager.get_performance_stats()
    print(f"\nPerformance stats: {json.dumps(stats, indent=2)}")


async def test_agent_adapter():
    """Test Ollama agent adapter"""
    print("\n=== Testing Agent Adapter ===")

    # Test different agent classes
    agent_classes = ["explorer", "merchant", "scholar", "guardian"]

    for agent_class in agent_classes:
        print(f"\nTesting {agent_class} agent...")

        adapter = OllamaAgentAdapter(model_name="llama2:7b-q4_K_M", agent_class=agent_class)

        # Setup custom model (this would fail without Ollama)
        try:
            success = await adapter.setup_agent_model()
            print(f"Model setup: {'successful' if success else 'failed'}")
        except Exception as e:
            print(f"Model setup failed: {e}")
            continue

        # Test thinking
        context = """
        Current situation:
        - Location: Grid position (15, 20)
        - Resources: Food: 25, Water: 18, Energy: 12
        - Nearby: Unknown territory to the north
        - Time: Day 5, Morning

        What should I do next?
        """

        try:
            response = await adapter.think(context)
            print(f"Agent response: {response[:200]}...")
        except Exception as e:
            print(f"Thinking failed: {e}")


async def main():
    """Run all tests"""
    print("=== Local LLM Integration Test Suite ===")

    # Test Ollama integration
    ollama_available = await test_ollama_integration()

    # Test Local LLM Manager
    test_local_llm_manager()

    # Test edge optimization
    test_edge_optimization()

    # Test fallback mechanisms
    test_fallback_mechanisms()

    # Test agent adapter if Ollama is available
    if ollama_available:
        await test_agent_adapter()

    print("\n=== Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
