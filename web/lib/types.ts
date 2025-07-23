/**
 * Common TypeScript type definitions for FreeAgentics
 */

export interface Agent {
  id: string;
  name: string;
  description: string;
  status: string;
  type?: string;
  created_at?: string;
  updated_at?: string;
  thinking?: boolean;
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface KnowledgeGraphNode {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, unknown>;
}

export interface KnowledgeGraphEdge {
  source: string;
  target: string;
  relationship: string;
}

export interface KnowledgeGraph {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
}

export interface AppError {
  id: string;
  type: "api" | "network" | "validation" | "auth" | "system" | "unknown";
  message: string;
  userMessage: string;
  status?: number;
  timestamp: number;
  context?: Record<string, unknown>;
  retryable: boolean;
  originalError?: Error;
}

export interface IterationContext {
  iteration_number: number;
  total_agents: number;
  kg_nodes: number;
  conversation_summary: {
    iteration_count: number;
    belief_evolution: {
      trend: string;
      stability: number;
    };
    prompt_themes?: string[];
    suggestion_patterns?: {
      diversity: number;
      common_themes: string[];
    };
  };
}

// Re-export from api-client for compatibility
export type { Agent as AgentConfig } from "./api-client";
