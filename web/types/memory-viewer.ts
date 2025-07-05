export interface KnowledgeEntry {
  id: string;
  type: string;
  // Database fields with defaults
  label?: string;
  properties?: Record<string, any>;
  version?: number;
  is_current?: boolean;
  confidence: number;
  source?: string;
  creator_agent_id?: string;
  created_at?: string;
  updated_at?: string;
  // UI compatibility fields (required for existing components)
  title: string;
  content: string;
  tags: string[];
  importance: number;
  timestamp: string;
  lastUpdated: string;
  agentId: string;
  relatedAgents?: string[];
  relatedKnowledge?: string[];
  verified?: boolean;
}

export interface SelectedKnowledgeNode {
  id: string;
  type: string;
  data?: any;
}

export interface AgentToolPermissions {
  [toolKey: string]: boolean;
}

export interface Coalition {
  id: string;
  name: string;
  description?: string;
  status: "forming" | "active" | "disbanding" | "dissolved";
  objectives: Record<string, any>;
  required_capabilities: string[];
  achieved_objectives: string[];
  performance_score: number;
  cohesion_score: number;
  created_at: string;
  dissolved_at?: string;
  updated_at: string;
  agent_count: number;
}

export interface KnowledgeEdge {
  id: string;
  source_id: string;
  target_id: string;
  type: string;
  properties: Record<string, any>;
  confidence: number;
  created_at: string;
}
