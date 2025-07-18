import { useState, useCallback, useEffect } from "react";
import { useWebSocket } from "./use-websocket";

export type AgentStatus = "active" | "idle" | "error";
export type AgentType = "explorer" | "collector" | "analyzer" | "custom";

export interface Agent {
  id: string;
  name: string;
  type: AgentType;
  status: AgentStatus;
  description?: string;
  createdAt?: string;
  lastActiveAt?: string;
  beliefs?: Record<string, any>;
  goals?: string[];
}

export interface CreateAgentParams {
  description: string;
}

export interface UpdateAgentParams {
  name?: string;
  type?: AgentType;
  goals?: string[];
}

export interface AgentsState {
  agents: Agent[];
  createAgent: (params: CreateAgentParams) => Promise<Agent>;
  updateAgent: (id: string, params: UpdateAgentParams) => Promise<void>;
  deleteAgent: (id: string) => Promise<void>;
  isLoading: boolean;
  error: Error | null;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useAgents(): AgentsState {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const { lastMessage } = useWebSocket();

  // Fetch agents on mount
  useEffect(() => {
    fetchAgents();
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "agent_update") {
      const { agentId, ...updates } = lastMessage.data;
      setAgents((prev) =>
        prev.map((agent) => (agent.id === agentId ? { ...agent, ...updates } : agent)),
      );
    } else if (lastMessage.type === "agent_created") {
      setAgents((prev) => [...prev, lastMessage.data]);
    } else if (lastMessage.type === "agent_deleted") {
      setAgents((prev) => prev.filter((agent) => agent.id !== lastMessage.data.agentId));
    }
  }, [lastMessage]);

  const fetchAgents = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/agents`);
      if (!response.ok) {
        throw new Error("Failed to fetch agents");
      }

      const data = await response.json();
      setAgents(data.agents || []);
    } catch (err) {
      setError(err as Error);
      console.error("Failed to fetch agents:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const createAgent = useCallback(async (params: CreateAgentParams): Promise<Agent> => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/agents`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        throw new Error("Failed to create agent");
      }

      const newAgent = await response.json();

      // Optimistically add to state (WebSocket will confirm)
      setAgents((prev) => [...prev, newAgent]);

      return newAgent;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateAgent = useCallback(async (id: string, params: UpdateAgentParams): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/agents/${id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        throw new Error("Failed to update agent");
      }

      // Optimistically update state (WebSocket will confirm)
      setAgents((prev) => prev.map((agent) => (agent.id === id ? { ...agent, ...params } : agent)));
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const deleteAgent = useCallback(async (id: string): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/agents/${id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete agent");
      }

      // Optimistically remove from state (WebSocket will confirm)
      setAgents((prev) => prev.filter((agent) => agent.id !== id));
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    agents,
    createAgent,
    updateAgent,
    deleteAgent,
    isLoading,
    error,
  };
}
