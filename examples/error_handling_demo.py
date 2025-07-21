"""Demonstration of robust error handling in FreeAgentics agents."""

from unittest.mock import MagicMock

import numpy as np

from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent


def demonstrate_error_recovery():
    """Demonstrate how agents handle PyMDP failures gracefully."""
    print("=== FreeAgentics Error Handling Demonstration ===\n")

    # Create agent
    agent = BasicExplorerAgent("demo_agent", "Error Recovery Demo", grid_size=5)
    agent.start()

    print(f"Agent {agent.agent_id} created and started")
    print(f"PyMDP available: {PYMDP_AVAILABLE}")

    # Demonstrate normal operation
    print("\n1. Normal Operation:")
    observation = {"position": [2, 2], "surroundings": np.zeros((3, 3))}
    action = agent.step(observation)
    print(f"   Normal step: observation -> action '{action}'")

    # Demonstrate error handling
    print("\n2. PyMDP Failure Handling:")

    # Mock PyMDP agent to simulate failures
    mock_agent = MagicMock()
    mock_agent.infer_policies.side_effect = Exception("Simulated PyMDP failure")
    agent.pymdp_agent = mock_agent

    # Agent should handle this gracefully
    action = agent.step(observation)
    print(f"   PyMDP failure -> fallback action '{action}'")

    # Check error was recorded
    error_summary = agent.error_handler.get_error_summary()
    print(f"   Errors recorded: {error_summary['total_errors']}")

    # Demonstrate multiple failure types
    print("\n3. Different Error Types:")

    test_cases = [
        (None, "None observation"),
        ({"invalid": "data"}, "Invalid observation format"),
        ("string_obs", "String observation"),
    ]

    for obs, description in test_cases:
        action = agent.step(obs)
        print(f"   {description} -> action '{action}'")

    # Show final error summary
    print("\n4. Final Error Summary:")
    final_summary = agent.error_handler.get_error_summary()
    print(f"   Total errors: {final_summary['total_errors']}")
    print(f"   Error types: {list(final_summary['error_counts'].keys())}")

    # Show agent status with error information
    print("\n5. Agent Status with Error Info:")
    status = agent.get_status()
    if "error_summary" in status:
        print(
            f"   Agent handled {status['error_summary']['total_errors']} errors gracefully"
        )
        print(f"   Agent completed {status['total_steps']} steps successfully")

    print("\n=== Demo Complete: Agent remained operational despite errors ===")


def demonstrate_error_recovery_strategies():
    """Demonstrate different error recovery strategies."""
    print("\n=== Error Recovery Strategies Demo ===\n")

    agent = BasicExplorerAgent("strategy_demo", "Strategy Demo", grid_size=3)
    agent.start()

    # Test retry mechanism
    print("1. Retry Mechanism:")

    # Mock agent that fails then succeeds
    mock_agent = MagicMock()
    failure_count = 0

    def failing_infer_policies():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:  # Fail first 2 times
            raise Exception(f"Simulated failure #{failure_count}")
        return (np.array([0.2, 0.8]), None)  # Succeed on 3rd try

    mock_agent.infer_policies.side_effect = failing_infer_policies
    mock_agent.sample_action.return_value = np.array(1)
    agent.pymdp_agent = mock_agent

    observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}

    # Multiple attempts should eventually succeed
    for i in range(5):
        action = agent.step(observation)
        print(f"   Attempt {i + 1}: action '{action}'")

        if i == 2:  # After success, reset mock for normal operation
            mock_agent.infer_policies.side_effect = None
            mock_agent.infer_policies.return_value = (
                np.array([0.2, 0.8]),
                None,
            )

    print(f"   Total failures handled: {failure_count}")

    # Show recovery strategy status
    print("\n2. Recovery Strategy Status:")
    for name, strategy in agent.error_handler.recovery_strategies.items():
        print(f"   {name}: {strategy.retry_count}/{strategy.max_retries} retries used")


def demonstrate_concurrent_error_handling():
    """Demonstrate error handling with multiple agents."""
    print("\n=== Concurrent Error Handling Demo ===\n")

    # Create multiple agents with different failure modes
    agents = []
    for i in range(3):
        agent = BasicExplorerAgent(f"concurrent_{i}", f"Agent {i}", grid_size=3)
        agent.start()
        agents.append(agent)

    # Set up different failure patterns
    failure_patterns = [
        "inference_failure",
        "action_selection_failure",
        "normal_operation",
    ]

    for i, (agent, pattern) in enumerate(zip(agents, failure_patterns)):
        if pattern != "normal_operation":
            mock_agent = MagicMock()
            if pattern == "inference_failure":
                mock_agent.infer_states.side_effect = Exception("Inference failed")
            elif pattern == "action_selection_failure":
                mock_agent.infer_policies.side_effect = Exception(
                    "Action selection failed"
                )
            agent.pymdp_agent = mock_agent

        print(f"Agent {i}: {pattern}")

    # Run all agents simultaneously
    observation = {"position": [1, 1], "surroundings": np.zeros((3, 3))}

    print("\nSimultaneous execution:")
    for i, agent in enumerate(agents):
        action = agent.step(observation)
        error_count = agent.error_handler.get_error_summary()["total_errors"]
        print(f"   Agent {i}: action '{action}', errors: {error_count}")

    print("\nAll agents remained operational despite individual failures")


if __name__ == "__main__":
    demonstrate_error_recovery()
    demonstrate_error_recovery_strategies()
    demonstrate_concurrent_error_handling()
