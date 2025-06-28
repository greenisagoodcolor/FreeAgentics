# Async Testing Guide for FreeAgentics

## Overview

This document provides guidance on writing and running asynchronous tests in the FreeAgentics project. Asynchronous code requires special handling in tests to ensure that coroutines are properly awaited and that the event loop is correctly managed.

## Setup

### Required Dependencies

Make sure you have the following dependencies installed:

```bash
pip install pytest pytest-asyncio
```

These are included in the `requirements-dev.txt` file.

### Configuration

We use a `pytest.ini` file in the root directory to configure pytest for async testing:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    asyncio: mark a test as an asyncio test
    slow: mark test as slow
    integration: mark test as integration test
    unit: mark test as unit test
```

The `asyncio_mode = auto` setting automatically handles async tests without requiring explicit markers in most cases.

## Writing Async Tests

### Basic Structure

Async tests should be written using the `async def` syntax and the `pytest.mark.asyncio` decorator:

```python
import pytest
import pytest_asyncio
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected_value
```

### Async Fixtures

For fixtures that need to perform async operations, use `pytest_asyncio.fixture` instead of `pytest.fixture`:

```python
import pytest_asyncio

@pytest_asyncio.fixture
async def async_resource():
    resource = await create_resource()
    yield resource
    await cleanup_resource(resource)
```

### Handling Timeouts

For tests that might hang, use the `asyncio.wait_for` function:

```python
@pytest.mark.asyncio
async def test_with_timeout():
    result = await asyncio.wait_for(long_running_function(), timeout=2.0)
    assert result == expected_value
```

## Common Patterns

### Testing Async Communication

When testing async communication between components:

1. Use `asyncio.sleep(small_value)` to allow time for messages to be processed
2. Create mock message handlers that set flags or store values when called
3. Use `asyncio.gather` to run multiple coroutines concurrently

Example:

```python
@pytest.mark.asyncio
async def test_message_handling():
    sender = create_sender()
    receiver = create_receiver()

    await sender.send_message(receiver.id, "test_message")
    await asyncio.sleep(0.1)  # Allow time for message processing

    assert receiver.last_message == "test_message"
```

### Testing Error Conditions

For testing error handling in async code:

```python
@pytest.mark.asyncio
async def test_error_handling():
    with pytest.raises(SomeException):
        await function_that_should_raise()
```

## Best Practices

1. **Avoid Mixing Sync and Async**: Keep test functions either all sync or all async in a single test module.
2. **Clean Up Resources**: Always clean up async resources in fixture teardown.
3. **Use Small Sleep Values**: When using `asyncio.sleep()`, use the smallest value that makes the test reliable.
4. **Test Both Success and Failure**: Test both successful completion and error handling.
5. **Mock External Services**: Use mock objects for external services to avoid real network calls.
6. **Avoid Infinite Loops**: Ensure all async loops have exit conditions.

## Troubleshooting

### Common Issues

1. **"Event loop is closed"**: This usually means a test is trying to use the event loop after it's been closed. Make sure all coroutines are awaited properly.

2. **"No current event loop"**: This can happen when trying to run async code outside of an async function. Ensure all async code is properly awaited.

3. **Tests hanging**: This often indicates a coroutine that never completes. Use timeouts and ensure all async operations can complete.

### Debugging Techniques

1. Use `pytest -vv` for more verbose output
2. Add debug logging to see the sequence of async operations
3. Use `asyncio.current_task()` and `asyncio.all_tasks()` to inspect running tasks

## Example: Testing Agent Communication

Here's a complete example of testing async communication between agents:

```python
import pytest
import pytest_asyncio
import asyncio
from agents.base.data_model import AgentClass, Position
from communication.message_system import MessageSystem, MessageType, Message

class TestMessageSystem(MessageSystem):
    """Extended MessageSystem for tests"""
    def __init__(self):
        super().__init__()
        self.registered_agents = []
        self.enabled = True

    def register_agent(self, agent):
        """Register an agent with the message system"""
        self.registered_agents.append(agent)

    def disable(self):
        """Disable the message system"""
        self.enabled = False

    def enable(self):
        """Enable the message system"""
        self.enabled = True

class TestAgentCommunication:
    @pytest_asyncio.fixture
    async def message_system(self):
        return TestMessageSystem()

    @pytest_asyncio.fixture
    async def agents(self, message_system):
        # Create test agents
        agents = []
        for i in range(2):
            agent = Agent(
                agent_id=f"agent_{i}",
                name=f"Agent{i}",
                message_system=message_system
            )
            agents.append(agent)
            message_system.register_agent(agent)
        return agents

    @pytest.mark.asyncio
    async def test_message_delivery(self, agents):
        sender = agents[0]
        receiver = agents[1]

        await sender.send_message(
            receiver.agent_id,
            MessageType.SYSTEM_ALERT,
            "Test message"
        )

        # Allow time for message processing
        await asyncio.sleep(0.1)

        messages = receiver.get_recent_messages()
        assert len(messages) > 0
        assert messages[0].content == "Test message"
```

## References

- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [pytest documentation](https://docs.pytest.org/)
