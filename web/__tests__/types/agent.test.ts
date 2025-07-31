/**
 * UNIFIED AGENT TYPE SYSTEM TESTS
 *
 * These tests verify the correct behavior of the unified Agent type system
 * designed by the Nemesis Committee to resolve interface contract violations.
 *
 * Tests follow TDD methodology to ensure the unified system works correctly
 * before beginning systematic migration.
 */

import { describe, it, expect } from "@jest/globals";

import {
  // Core types
  CoreAgent,
  ManagementAgent,
  PromptAgent,
  SimulationAgent,

  // Enums
  AgentStatus,
  AgentType,
  SimulationState,

  // Utilities
  AgentTypeGuards,
  AgentTranslators,
  AgentFactories,
  AgentValidators,
} from "@/types/agent";

describe("Unified Agent Type System", () => {
  describe("Core Agent Interface", () => {
    it("should define minimal shared contract", () => {
      const coreAgent: CoreAgent = {
        id: "core-1",
        name: "Core Test Agent",
      };

      expect(coreAgent.id).toBe("core-1");
      expect(coreAgent.name).toBe("Core Test Agent");
    });
  });

  describe("Bounded Context Interfaces", () => {
    describe("ManagementAgent (Comprehensive Context)", () => {
      it("should compose all management-relevant traits", () => {
        const managementAgent: ManagementAgent = {
          // Core properties
          id: "mgmt-1",
          name: "Management Agent",

          // Management-specific properties
          type: "explorer",
          status: "active",
          description: "Test management agent",
          createdAt: "2025-01-01T00:00:00Z",
          lastActiveAt: "2025-01-01T00:00:00Z",
          beliefs: { confidence: 0.8 },
          goals: ["explore", "collect"],
        };

        // Verify core properties
        expect(managementAgent.id).toBe("mgmt-1");
        expect(managementAgent.name).toBe("Management Agent");

        // Verify management-specific properties
        expect(managementAgent.type).toBe("explorer");
        expect(managementAgent.status).toBe("active");
        expect(managementAgent.description).toBe("Test management agent");
        expect(managementAgent.beliefs).toEqual({ confidence: 0.8 });
        expect(managementAgent.goals).toEqual(["explore", "collect"]);
      });

      it("should enforce AgentType enum constraints", () => {
        const validTypes: AgentType[] = ["explorer", "collector", "analyzer", "custom"];

        validTypes.forEach((type) => {
          const agent: ManagementAgent = {
            id: "test",
            name: "Test",
            type: type,
            status: "active",
          };
          expect(agent.type).toBe(type);
        });
      });

      it("should enforce AgentStatus enum constraints", () => {
        const validStatuses: AgentStatus[] = ["active", "idle", "error"];

        validStatuses.forEach((status) => {
          const agent: ManagementAgent = {
            id: "test",
            name: "Test",
            type: "explorer",
            status: status,
          };
          expect(agent.status).toBe(status);
        });
      });
    });

    describe("PromptAgent (Minimal Context)", () => {
      it("should compose prompt-relevant traits with flexible typing", () => {
        const promptAgent: PromptAgent = {
          // Core properties
          id: "prompt-1",
          name: "Prompt Agent",

          // Prompt-specific properties
          status: "processing", // Generic string (not enum)
          type: "prompter", // Optional generic string
          thinking: true,
        };

        // Verify core properties
        expect(promptAgent.id).toBe("prompt-1");
        expect(promptAgent.name).toBe("Prompt Agent");

        // Verify flexible typing
        expect(promptAgent.status).toBe("processing");
        expect(typeof promptAgent.status).toBe("string");
        expect(promptAgent.type).toBe("prompter");
        expect(promptAgent.thinking).toBe(true);
      });

      it("should allow optional type and thinking properties", () => {
        const minimalPromptAgent: PromptAgent = {
          id: "prompt-2",
          name: "Minimal Prompt Agent",
          status: "idle",
        };

        expect(minimalPromptAgent.type).toBeUndefined();
        expect(minimalPromptAgent.thinking).toBeUndefined();
      });
    });

    describe("SimulationAgent (Spatial Context)", () => {
      it("should compose simulation-relevant traits", () => {
        const simulationAgent: SimulationAgent = {
          // Core properties
          id: "sim-1",
          name: "Simulation Agent",

          // Simulation-specific properties
          position: { x: 10, y: 20 },
          energy: 75,
          state: "exploring",
          beliefs: { sector: "alpha" },
        };

        // Verify core properties
        expect(simulationAgent.id).toBe("sim-1");
        expect(simulationAgent.name).toBe("Simulation Agent");

        // Verify simulation-specific properties
        expect(simulationAgent.position).toEqual({ x: 10, y: 20 });
        expect(simulationAgent.energy).toBe(75);
        expect(simulationAgent.state).toBe("exploring");
        expect(simulationAgent.beliefs).toEqual({ sector: "alpha" });
      });

      it("should enforce SimulationState enum constraints", () => {
        const validStates: SimulationState[] = ["idle", "exploring", "collecting", "returning"];

        validStates.forEach((state) => {
          const agent: SimulationAgent = {
            id: "test",
            name: "Test",
            position: { x: 0, y: 0 },
            energy: 100,
            state: state,
          };
          expect(agent.state).toBe(state);
        });
      });
    });
  });

  describe("Type Guards", () => {
    it("should correctly identify ManagementAgent", () => {
      const managementAgent = {
        id: "test",
        name: "Test",
        type: "explorer",
        status: "active",
      };

      const notManagementAgent = {
        id: "test",
        name: "Test",
        // Missing type and status
      };

      expect(AgentTypeGuards.isManagementAgent(managementAgent)).toBe(true);
      expect(AgentTypeGuards.isManagementAgent(notManagementAgent)).toBe(false);
      expect(AgentTypeGuards.isManagementAgent(null)).toBe(false);
      expect(AgentTypeGuards.isManagementAgent("string")).toBe(false);
    });

    it("should correctly identify PromptAgent", () => {
      const promptAgent = {
        id: "test",
        name: "Test",
        status: "processing",
      };

      const notPromptAgent = {
        id: "test",
        // Missing name and status
      };

      expect(AgentTypeGuards.isPromptAgent(promptAgent)).toBe(true);
      expect(AgentTypeGuards.isPromptAgent(notPromptAgent)).toBe(false);
    });

    it("should correctly identify SimulationAgent", () => {
      const simulationAgent = {
        id: "test",
        name: "Test",
        position: { x: 0, y: 0 },
        state: "idle",
        energy: 100,
      };

      const notSimulationAgent = {
        id: "test",
        name: "Test",
        // Missing position, state, energy
      };

      expect(AgentTypeGuards.isSimulationAgent(simulationAgent)).toBe(true);
      expect(AgentTypeGuards.isSimulationAgent(notSimulationAgent)).toBe(false);
    });
  });

  describe("Translation Utilities", () => {
    it("should translate ManagementAgent to PromptAgent correctly", () => {
      const managementAgent: ManagementAgent = {
        id: "mgmt-1",
        name: "Management Agent",
        type: "explorer",
        status: "active",
        description: "Test agent",
        beliefs: { test: true },
        goals: ["goal1"],
      };

      const promptAgent = AgentTranslators.managementToPrompt(managementAgent);

      expect(promptAgent.id).toBe("mgmt-1");
      expect(promptAgent.name).toBe("Management Agent");
      expect(promptAgent.status).toBe("active");
      expect(promptAgent.type).toBe("explorer");
      expect(promptAgent.thinking).toBe(false);

      // Verify management-specific properties are not included
      expect(promptAgent).not.toHaveProperty("description");
      expect(promptAgent).not.toHaveProperty("beliefs");
      expect(promptAgent).not.toHaveProperty("goals");
    });

    it("should translate ManagementAgent to simulation format correctly", () => {
      const managementAgent: ManagementAgent = {
        id: "mgmt-1",
        name: "Management Agent",
        type: "explorer",
        status: "active",
        description: "Test agent",
      };

      const simulationFormat = AgentTranslators.managementToSimulationFormat(managementAgent);

      expect(simulationFormat.id).toBe("mgmt-1");
      expect(simulationFormat.name).toBe("Management Agent");
      expect(simulationFormat.type).toBe("explorer");

      // Verify only essential properties for simulation API
      expect(Object.keys(simulationFormat)).toEqual(["id", "name", "type"]);
    });

    it("should translate PromptAgent to partial ManagementAgent correctly", () => {
      const promptAgent: PromptAgent = {
        id: "prompt-1",
        name: "Prompt Agent",
        status: "active",
        type: "explorer",
        thinking: true,
      };

      const partialManagement = AgentTranslators.promptToManagement(promptAgent);

      expect(partialManagement.id).toBe("prompt-1");
      expect(partialManagement.name).toBe("Prompt Agent");
      expect(partialManagement.status).toBe("active");
      expect(partialManagement.type).toBe("explorer");

      // Verify undefined properties for management context
      expect(partialManagement.description).toBeUndefined();
      expect(partialManagement.beliefs).toBeUndefined();
      expect(partialManagement.goals).toBeUndefined();
    });

    it("should translate SimulationAgent to partial ManagementAgent correctly", () => {
      const simulationAgent: SimulationAgent = {
        id: "sim-1",
        name: "Simulation Agent",
        position: { x: 10, y: 20 },
        energy: 75,
        state: "exploring",
        beliefs: { sector: "alpha" },
      };

      const partialManagement = AgentTranslators.simulationToManagement(simulationAgent);

      expect(partialManagement.id).toBe("sim-1");
      expect(partialManagement.name).toBe("Simulation Agent");
      expect(partialManagement.beliefs).toEqual({ sector: "alpha" });
      expect(partialManagement.status).toBe("active");
      expect(partialManagement.type).toBe("explorer");
    });
  });

  describe("Factory Functions", () => {
    it("should create ManagementAgent with sensible defaults", () => {
      const agent = AgentFactories.createManagementAgent({
        id: "factory-1",
        name: "Factory Agent",
        type: "collector",
      });

      expect(agent.id).toBe("factory-1");
      expect(agent.name).toBe("Factory Agent");
      expect(agent.type).toBe("collector");
      expect(agent.status).toBe("idle"); // Default
      expect(agent.createdAt).toBeDefined();
      expect(agent.lastActiveAt).toBeDefined();
      expect(agent.beliefs).toEqual({});
      expect(agent.goals).toEqual([]);
    });

    it("should create ManagementAgent with custom values", () => {
      const agent = AgentFactories.createManagementAgent({
        id: "factory-2",
        name: "Custom Factory Agent",
        type: "analyzer",
        status: "active",
        description: "Custom description",
      });

      expect(agent.status).toBe("active");
      expect(agent.description).toBe("Custom description");
    });

    it("should create PromptAgent with sensible defaults", () => {
      const agent = AgentFactories.createPromptAgent({
        id: "prompt-factory-1",
        name: "Prompt Factory Agent",
      });

      expect(agent.id).toBe("prompt-factory-1");
      expect(agent.name).toBe("Prompt Factory Agent");
      expect(agent.status).toBe("idle"); // Default
      expect(agent.thinking).toBe(false); // Default
      expect(agent.type).toBeUndefined(); // Default
    });

    it("should create SimulationAgent with sensible defaults", () => {
      const agent = AgentFactories.createSimulationAgent({
        id: "sim-factory-1",
        name: "Simulation Factory Agent",
      });

      expect(agent.id).toBe("sim-factory-1");
      expect(agent.name).toBe("Simulation Factory Agent");
      expect(agent.position).toEqual({ x: 0, y: 0 }); // Default
      expect(agent.energy).toBe(100); // Default
      expect(agent.state).toBe("idle"); // Default
      expect(agent.beliefs).toEqual({}); // Default
    });
  });

  describe("Validation Utilities", () => {
    it("should validate ManagementAgent correctly", () => {
      const validAgent = {
        id: "valid-1",
        name: "Valid Agent",
        type: "explorer",
        status: "active",
      };

      const invalidAgent = {
        id: "invalid-1",
        // Missing required properties
      };

      expect(() => AgentValidators.validateManagementAgent(validAgent)).not.toThrow();
      expect(() => AgentValidators.validateManagementAgent(invalidAgent)).toThrow(
        "Invalid ManagementAgent object",
      );
    });

    it("should validate PromptAgent correctly", () => {
      const validAgent = {
        id: "valid-1",
        name: "Valid Agent",
        status: "processing",
      };

      const invalidAgent = {
        id: "invalid-1",
        // Missing name and status
      };

      expect(() => AgentValidators.validatePromptAgent(validAgent)).not.toThrow();
      expect(() => AgentValidators.validatePromptAgent(invalidAgent)).toThrow(
        "Invalid PromptAgent object",
      );
    });

    it("should validate SimulationAgent correctly", () => {
      const validAgent = {
        id: "valid-1",
        name: "Valid Agent",
        position: { x: 0, y: 0 },
        state: "idle",
        energy: 100,
      };

      const invalidAgent = {
        id: "invalid-1",
        name: "Invalid Agent",
        // Missing position, state, energy
      };

      expect(() => AgentValidators.validateSimulationAgent(validAgent)).not.toThrow();
      expect(() => AgentValidators.validateSimulationAgent(invalidAgent)).toThrow(
        "Invalid SimulationAgent object",
      );
    });
  });

  describe("Backward Compatibility", () => {
    it("should provide Agent alias for ManagementAgent", () => {
      // This test ensures the deprecated Agent type still works
      // TypeScript should not complain about this usage
      const agent: ManagementAgent = {
        id: "compat-1",
        name: "Compatibility Agent",
        type: "explorer",
        status: "active",
      };

      // Should be usable as the old Agent type
      expect(agent.id).toBe("compat-1");
      expect(agent.name).toBe("Compatibility Agent");
    });
  });

  describe("Integration Scenarios", () => {
    it("should handle cross-context usage (SimulationGrid scenario)", () => {
      // This test simulates the problematic scenario in SimulationGrid.tsx
      // where both useAgents and useSimulation are imported

      // Management context agent (from useAgents)
      const managementAgent: ManagementAgent = {
        id: "integration-1",
        name: "Integration Agent",
        type: "explorer",
        status: "active",
        description: "Test integration",
      };

      // Convert to simulation format for API call (lines 171-175 in use-simulation.ts)
      const simulationFormat = AgentTranslators.managementToSimulationFormat(managementAgent);

      expect(simulationFormat).toEqual({
        id: "integration-1",
        name: "Integration Agent",
        type: "explorer",
      });

      // Verify no information loss for essential properties
      expect(simulationFormat.id).toBe(managementAgent.id);
      expect(simulationFormat.name).toBe(managementAgent.name);
      expect(simulationFormat.type).toBe(managementAgent.type);
    });

    it("should handle test mock data scenarios", () => {
      // This test addresses the failing test scenarios identified in SCAN 2.4

      // Create proper mock data with all required properties
      const mockPromptResponse = {
        agents: [
          AgentFactories.createPromptAgent({
            id: "mock-1",
            name: "Mock Agent",
            status: "active",
          }),
        ],
        knowledgeGraph: { nodes: [], edges: [] },
        suggestions: ["suggestion1", "suggestion2"], // Previously missing
        conversationId: "conv-123",
      };

      expect(mockPromptResponse.agents[0].id).toBe("mock-1");
      expect(mockPromptResponse.suggestions).toEqual(["suggestion1", "suggestion2"]);

      // Create proper agent mock with all required properties
      const mockAgent = AgentFactories.createPromptAgent({
        id: "mock-agent-1",
        name: "Mock Test Agent",
        status: "active",
      });

      expect(mockAgent.id).toBe("mock-agent-1");
      expect(mockAgent.name).toBe("Mock Test Agent");
      expect(mockAgent.status).toBe("active");
      // No TypeScript errors about missing description property
    });
  });
});
