Feature: Coalition Formation
  As a multi-agent system
  I want agents to form coalitions
  So that they can collaborate effectively

  Scenario: Basic coalition formation
    Given there are 3 agents in the system
    When agent 1 proposes a coalition
    And agent 2 accepts the proposal
    Then a coalition should be formed between agent 1 and agent 2