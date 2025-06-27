Feature: Agent Exploration Behavior
  As a researcher studying Active Inference agents
  I want agents to explore their environment intelligently
  So that they can discover resources and reduce epistemic uncertainty

  Background:
    Given a world with unknown territories
    And resources are distributed randomly

  Scenario: Curious explorer discovers new areas
    Given an Explorer agent with high curiosity
    When the agent explores for 50 timesteps
    Then the agent should discover new areas
    And exploration efficiency should improve over time

  Scenario: Cautious explorer avoids risky areas
    Given an Explorer agent with high caution
    And dangerous areas are marked in the world
    When the agent explores for 30 timesteps
    Then the agent should avoid dangerous territories
    And the agent should find safe paths to resources

  Scenario: Multiple explorers coordinate exploration
    Given 3 Explorer agents in the same world
    When agents explore independently for 20 timesteps
    Then agents should cover different territories
    And total area coverage should be maximized 