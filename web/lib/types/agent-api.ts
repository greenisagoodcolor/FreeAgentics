// Agent API Types and Interfaces

export interface AgentPersonality {
  openness: number; // 0-1
  conscientiousness: number; // 0-1
  extraversion: number; // 0-1
  agreeableness: number; // 0-1
  neuroticism: number; // 0-1
}

export type AgentCapability =
  | "movement"
  | "perception"
  | "communication"
  | "planning"
  | "learning"
  | "memory"
  | "resource_management"
  | "social_interaction";

export type AgentStatus =
  | "idle"
  | "moving"
  | "interacting"
  | "planning"
  | "executing"
  | "learning"
  | "error"
  | "offline";

export interface Position {
  x: number;
  y: number;
  z?: number;
}

export interface AgentResources {
  energy: number; // 0-100
  health: number; // 0-100
  memory_used: number;
  memory_capacity: number;
}

export interface AgentBelief {
  id: string;
  content: string;
  confidence: number; // 0-1
}

export interface AgentGoal {
  id: string;
  description: string;
  priority: number; // 0-1
  deadline: string | null;
}

export interface AgentRelationship {
  agent_id: string;
  trust_level: number; // 0-1
  last_interaction: string;
}

// Base Agent interface
export interface Agent {
  id: string;
  name: string;
  status: AgentStatus;
  personality: AgentPersonality;
  capabilities: AgentCapability[];
  position: Position;
  resources: AgentResources;
  tags: string[];
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

// Extended Agent interface with additional details
export interface AgentDetailed extends Agent {
  beliefs: AgentBelief[];
  goals: AgentGoal[];
  relationships: AgentRelationship[];
}

// API Request/Response Types

export interface CreateAgentRequest {
  name: string;
  personality: AgentPersonality;
  capabilities?: AgentCapability[];
  initialPosition?: Position;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface UpdateAgentRequest {
  name?: string;
  status?: AgentStatus;
  position?: Position;
  resources?: Partial<AgentResources>;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface ListAgentsQuery {
  status?: AgentStatus;
  capability?: string;
  tag?: string;
  limit?: number;
  offset?: number;
  sortBy?: "created_at" | "updated_at" | "name" | "status";
  sortOrder?: "asc" | "desc";
}

export interface ListAgentsResponse {
  agents: Agent[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
  };
}

// State Management Types

export interface StateTransition {
  timestamp: string;
  from_state: AgentStatus;
  to_state: AgentStatus;
  reason: string;
  metadata?: Record<string, any>;
}

export interface UpdateStateRequest {
  status: AgentStatus;
  force?: boolean;
}

export interface StateHistoryResponse {
  agent_id: string;
  current_state: AgentStatus;
  state_history: StateTransition[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
  };
}

// Command Execution Types

export type CommandType =
  | "move"
  | "interact"
  | "observe"
  | "plan"
  | "learn"
  | "rest";

export interface Command {
  id: string;
  agent_id: string;
  command: CommandType;
  parameters: Record<string, any>;
  status: "queued" | "executing" | "completed" | "failed";
  issued_at: string;
  started_at: string | null;
  completed_at: string | null;
  result: any | null;
}

export interface ExecuteCommandRequest {
  command: CommandType;
  parameters: Record<string, any>;
  async?: boolean;
}

export interface ExecuteCommandResponse {
  command: Command;
  async: boolean;
  status_url: string;
}

// Memory Types

export type MemoryType =
  | "event"
  | "interaction"
  | "location"
  | "pattern"
  | "general";

export interface Memory {
  id: string;
  type: MemoryType;
  content: string;
  importance: number; // 0-1
  timestamp: string;
  access_count: number;
  last_accessed: string;
  tags: string[];
  metadata: Record<string, any>;
}

export interface AddMemoryRequest {
  type: MemoryType;
  content: string;
  importance?: number;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface QueryMemoryRequest {
  type?: MemoryType;
  query?: string;
  tags?: string[];
  min_importance?: number;
  limit?: number;
  offset?: number;
}

export interface MemoryResponse {
  agent_id: string;
  memories: Memory[];
  memory_stats: {
    total_memories: number;
    total_capacity: number;
    used_capacity: number;
    consolidation_pending: boolean;
  };
  pagination: {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
  };
}

// Export Types

export interface ExportAgentRequest {
  target: string;
}

export interface AgentExportPackage {
  manifest: {
    package_id: string;
    agent_id: string;
    created_at: string;
    target: {
      name: string;
      platform: string;
      cpu_arch: string;
      ram_gb: number;
    };
    contents: {
      model: {
        path: string;
        size_mb: number;
        checksum: string;
      };
      knowledge: {
        path: string;
        size_mb: number;
        checksum: string;
      };
      config: {
        path: string;
        checksum: string;
      };
      scripts: {
        path: string;
        checksum: string;
      };
    };
    metrics: {
      total_size_mb: number;
      compression_ratio: number;
    };
  };
  config: any;
  deployment_scripts: string[];
}

// WebSocket Types

export interface WebSocketMessage {
  action: "subscribe" | "unsubscribe";
  agent_ids: string[];
}

export interface WebSocketEvent {
  type: "state_change" | "resource_update" | "command_update" | "memory_update";
  agent_id: string;
  data: any;
  timestamp: string;
}

// Error Response
export interface ErrorResponse {
  error: string;
  details?: any;
}
