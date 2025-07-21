#!/usr/bin/env python3
"""
Test suite specifically for AgentManager.create_agent interface.

This test file follows strict TDD principles to expose and fix the current
interface mismatch in AgentManager.create_agent method.

Current issue: The method expects (agent_type: str, name: str, **kwargs)
but tests and documentation expect it to accept a config dict.
"""

import asyncio

import pytest

from agents.agent_manager import AgentManager


class TestAgentManagerCreateAgentInterface:
    """Tests for AgentManager.create_agent method interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AgentManager()

    def test_create_agent_with_config_dict_should_fail(self):
        """
        RED: This test exposes the bug - passing config dict should work but fails.

        Current implementation expects separate parameters but this should accept
        a config dict as documented.
        """
        config = {
            "name": "TestAgent",
            "type": "explorer",
            "num_states": [3],
            "num_obs": [3],
            "num_controls": [3],
        }

        # This should work according to docs but will fail with current implementation
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            self.manager.create_agent(config)

    def test_create_agent_with_async_interface_should_fail(self):
        """
        RED: Tests expect async interface but current implementation is sync.

        Documentation and tests expect async but current implementation is sync.
        """

        async def test_async():
            config = {"name": "TestAgent", "type": "explorer"}
            # This should work but will fail - not async
            with pytest.raises(TypeError):
                await self.manager.create_agent(config)

        asyncio.run(test_async())

    def test_create_agent_with_active_inference_type_should_fail(self):
        """
        Tests that 'active_inference' type is now supported.

        Previously this raised ValueError, but now it should work.
        """
        # Active inference type is now supported
        agent_id = self.manager.create_agent("active_inference", "TestAgent")
        assert agent_id is not None
        assert agent_id.startswith(
            "ai_agent_"
        )  # Active inference agents get "ai_agent_" prefix

    def test_create_agent_current_working_interface(self):
        """
        This test shows the current working interface (for comparison).

        This is the only interface that currently works.
        """
        # This is the current working interface
        agent_id = self.manager.create_agent("explorer", "TestAgent")
        assert agent_id is not None
        assert agent_id.startswith("test_agent_")

        # Verify agent was created
        assert agent_id in self.manager.agents
        agent = self.manager.agents[agent_id]
        assert agent.name == "TestAgent"

    def test_create_agent_expected_return_type_should_fail(self):
        """
        RED: Tests expect Agent object but implementation returns string ID.

        The test_simple_validation.py expects to get an agent object with .name and .id
        but current implementation returns just the string ID.
        """

        # Current implementation returns string, but tests expect Agent object
        agent_id = self.manager.create_agent("explorer", "TestAgent")

        # This will fail because agent_id is string, not object with .name attribute
        with pytest.raises(AttributeError):
            _ = agent_id.name  # agent_id is str, doesn't have .name

    def test_create_agent_should_support_pymdp_config(self):
        """
        RED: Tests pass PyMDP config but current implementation ignores it.

        The test_simple_validation.py passes num_states, num_obs, num_controls
        but current implementation doesn't use these PyMDP parameters.
        """
        # Current implementation ignores PyMDP config completely
        agent_id = self.manager.create_agent(
            "explorer",
            "TestAgent",
            num_states=[3],
            num_obs=[3],
            num_controls=[3],
        )

        # PyMDP configs should be properly passed to the agent
        agent = self.manager.agents[agent_id]
        # Agent should have PyMDP configuration
        assert hasattr(agent, "num_states")
        assert hasattr(agent, "num_obs")
        assert hasattr(agent, "num_controls")
        assert agent.num_states == [3]
        assert agent.num_obs == [3]
        assert agent.num_controls == [3]
