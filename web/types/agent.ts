/**
 * UNIFIED AGENT TYPE SYSTEM
 *
 * This module implements the Nemesis Committee's consensus approach to resolving
 * Agent interface contract violations through systematic type unification.
 *
 * DESIGN PRINCIPLES:
 * - Interface Segregation Principle (Robert Martin)
 * - Bounded Context Pattern (Martin Fowler)
 * - Composition over Inheritance (Kent Beck)
 * - Backward Compatibility (Michael Feathers)
 *
 * ARCHITECTURE:
 * 1. Core Agent Interface - minimal shared contract
 * 2. Domain-Specific Extensions - compose additional properties as needed
 * 3. Bounded Context Types - explicit domain separations
 * 4. Translation Utilities - safe conversion between contexts
 */

// =============================================================================
// CORE AGENT INTERFACE (Minimal Shared Contract)
// =============================================================================

/**
 * Core Agent properties shared across ALL contexts.
 * This represents the absolute minimum contract that every agent must satisfy.
 *
 * Following Interface Segregation Principle - only essential properties.
 */
export interface CoreAgent {
  readonly id: string;
  readonly name: string;
}

// =============================================================================
// COMPOSABLE TRAIT INTERFACES (Interface Segregation)
// =============================================================================

/**
 * Agent status trait - for agents that have lifecycle status
 */
export interface HasStatus {
  readonly status: AgentStatus;
}

/**
 * Agent typing trait - for agents that have classification types
 */
export interface HasType {
  readonly type: AgentType;
}

/**
 * Agent description trait - for agents with human-readable descriptions
 */
export interface HasDescription {
  readonly description?: string;
}

/**
 * Agent timestamps trait - for agents with lifecycle tracking
 */
export interface HasTimestamps {
  readonly createdAt?: string;
  readonly lastActiveAt?: string;
}

/**
 * Agent beliefs trait - for agents with belief state
 */
export interface HasBeliefs {
  readonly beliefs?: Record<string, unknown>;
}

/**
 * Agent goals trait - for agents with goal-oriented behavior
 */
export interface HasGoals {
  readonly goals?: string[];
}

/**
 * Agent thinking trait - for agents with processing state
 */
export interface HasThinking {
  readonly thinking?: boolean;
}

/**
 * Agent position trait - for agents with spatial coordinates
 */
export interface HasPosition {
  readonly position: { x: number; y: number };
}

/**
 * Agent energy trait - for agents with energy/resource tracking
 */
export interface HasEnergy {
  readonly energy: number;
}

/**
 * Agent simulation state trait - for agents in simulation contexts
 */
export interface HasSimulationState {
  readonly state: SimulationState;
}

// =============================================================================
// ENUMERATION TYPES (Strongly Typed Constraints)
// =============================================================================

/**
 * Standard agent status enumeration
 * Used across all contexts that implement HasStatus
 */
export type AgentStatus = "active" | "idle" | "error";

/**
 * Standard agent type enumeration
 * Used across all contexts that implement HasType
 */
export type AgentType = "explorer" | "collector" | "analyzer" | "custom";

/**
 * Simulation-specific state enumeration
 * Used only in simulation contexts
 */
export type SimulationState = "idle" | "exploring" | "collecting" | "returning";

// =============================================================================
// BOUNDED CONTEXT INTERFACES (Domain-Specific Compositions)
// =============================================================================

/**
 * MANAGEMENT CONTEXT: Agent interface for agent management operations
 * Used in: use-agents.ts, AgentCreatorPanel.tsx, etc.
 *
 * Comprehensive interface for full agent lifecycle management
 */
export interface ManagementAgent
  extends CoreAgent,
    HasStatus,
    HasType,
    HasDescription,
    HasTimestamps,
    HasBeliefs,
    HasGoals {}

/**
 * PROMPT PROCESSING CONTEXT: Agent interface for prompt processing operations
 * Used in: use-prompt-processor.ts, prompt-interface.tsx, etc.
 *
 * Minimal interface focused on prompt processing needs
 */
export interface PromptAgent extends CoreAgent, HasThinking {
  // Note: status and type are intentionally string to maintain current behavior
  // This preserves backward compatibility during migration
  readonly status: string; // Generic string for flexibility in prompt context
  readonly type?: string; // Optional generic string
}

/**
 * SIMULATION CONTEXT: Agent interface for simulation operations
 * Used in: use-simulation.ts, SimulationGrid.tsx, etc.
 *
 * Spatial interface focused on simulation environment needs
 */
export interface SimulationAgent
  extends CoreAgent,
    HasPosition,
    HasEnergy,
    HasSimulationState,
    HasBeliefs {}

// =============================================================================
// BACKWARD COMPATIBILITY ALIASES (Migration Support)
// =============================================================================

/**
 * @deprecated Use ManagementAgent instead
 * Provided for backward compatibility during migration
 */
export type Agent = ManagementAgent;

// =============================================================================
// TYPE UTILITIES (Safe Conversion Between Contexts)
// =============================================================================

/**
 * Type guards for safe type checking across contexts
 */
export const AgentTypeGuards = {
  isManagementAgent: (agent: unknown): agent is ManagementAgent => {
    return (
      typeof agent === "object" &&
      agent !== null &&
      "id" in agent &&
      "name" in agent &&
      "status" in agent &&
      "type" in agent
    );
  },

  isPromptAgent: (agent: unknown): agent is PromptAgent => {
    return (
      typeof agent === "object" &&
      agent !== null &&
      "id" in agent &&
      "name" in agent &&
      "status" in agent
    );
  },

  isSimulationAgent: (agent: unknown): agent is SimulationAgent => {
    return (
      typeof agent === "object" &&
      agent !== null &&
      "id" in agent &&
      "name" in agent &&
      "position" in agent &&
      "state" in agent &&
      "energy" in agent
    );
  },
} as const;

/**
 * Translation utilities for safe conversion between bounded contexts
 * These maintain data integrity while converting between different agent representations
 */
export const AgentTranslators = {
  /**
   * Convert ManagementAgent to PromptAgent for prompt processing context
   */
  managementToPrompt: (agent: ManagementAgent): PromptAgent => ({
    id: agent.id,
    name: agent.name,
    status: agent.status, // Convert enum to string
    type: agent.type, // Convert enum to string
    thinking: false, // Default value for prompt context
  }),

  /**
   * Convert ManagementAgent to simulation format (for API calls)
   */
  managementToSimulationFormat: (agent: ManagementAgent) => ({
    id: agent.id,
    name: agent.name,
    type: agent.type,
  }),

  /**
   * Convert PromptAgent to partial ManagementAgent (for upgrading context)
   */
  promptToManagement: (agent: PromptAgent): Partial<ManagementAgent> => ({
    id: agent.id,
    name: agent.name,
    status: agent.status as AgentStatus, // Attempt safe conversion
    type: agent.type as AgentType, // Attempt safe conversion
    description: undefined,
    createdAt: undefined,
    lastActiveAt: undefined,
    beliefs: undefined,
    goals: undefined,
  }),

  /**
   * Convert SimulationAgent to partial ManagementAgent
   */
  simulationToManagement: (agent: SimulationAgent): Partial<ManagementAgent> => ({
    id: agent.id,
    name: agent.name,
    beliefs: agent.beliefs,
    status: "active" as AgentStatus, // Default status for simulation agents
    type: "explorer" as AgentType, // Default type for simulation agents
  }),
} as const;

// =============================================================================
// FACTORY FUNCTIONS (Consistent Object Creation)
// =============================================================================

/**
 * Factory functions for creating properly typed agent objects
 * These ensure consistent object creation across contexts
 */
export const AgentFactories = {
  /**
   * Create a new ManagementAgent with required fields and sensible defaults
   */
  createManagementAgent: (params: {
    id: string;
    name: string;
    type: AgentType;
    status?: AgentStatus;
    description?: string;
  }): ManagementAgent => ({
    id: params.id,
    name: params.name,
    type: params.type,
    status: params.status ?? "idle",
    description: params.description,
    createdAt: new Date().toISOString(),
    lastActiveAt: new Date().toISOString(),
    beliefs: {},
    goals: [],
  }),

  /**
   * Create a new PromptAgent with required fields and sensible defaults
   */
  createPromptAgent: (params: {
    id: string;
    name: string;
    status?: string;
    type?: string;
  }): PromptAgent => ({
    id: params.id,
    name: params.name,
    status: params.status ?? "idle",
    type: params.type,
    thinking: false,
  }),

  /**
   * Create a new SimulationAgent with required fields and sensible defaults
   */
  createSimulationAgent: (params: {
    id: string;
    name: string;
    position?: { x: number; y: number };
    energy?: number;
    state?: SimulationState;
  }): SimulationAgent => ({
    id: params.id,
    name: params.name,
    position: params.position ?? { x: 0, y: 0 },
    energy: params.energy ?? 100,
    state: params.state ?? "idle",
    beliefs: {},
  }),
} as const;

// =============================================================================
// VALIDATION UTILITIES (Runtime Type Safety)
// =============================================================================

/**
 * Runtime validation utilities for ensuring type safety at boundaries
 */
export const AgentValidators = {
  /**
   * Validate that an object conforms to ManagementAgent interface
   */
  validateManagementAgent: (obj: unknown): ManagementAgent => {
    if (!AgentTypeGuards.isManagementAgent(obj)) {
      throw new Error("Invalid ManagementAgent object");
    }
    return obj;
  },

  /**
   * Validate that an object conforms to PromptAgent interface
   */
  validatePromptAgent: (obj: unknown): PromptAgent => {
    if (!AgentTypeGuards.isPromptAgent(obj)) {
      throw new Error("Invalid PromptAgent object");
    }
    return obj;
  },

  /**
   * Validate that an object conforms to SimulationAgent interface
   */
  validateSimulationAgent: (obj: unknown): SimulationAgent => {
    if (!AgentTypeGuards.isSimulationAgent(obj)) {
      throw new Error("Invalid SimulationAgent object");
    }
    return obj;
  },
} as const;
