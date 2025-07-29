import { useState, useCallback, useEffect } from "react";
import { useWebSocket } from "./use-websocket";
import { apiGet, apiPost, apiDelete, ApiError } from "../lib/api";
import { useAuth } from "./use-auth";

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
  beliefs?: Record<string, unknown>;
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
  const { isAuthenticated, isLoading: isAuthLoading } = useAuth();

  // Fetch agents only after authentication is ready
  useEffect(() => {
    // Don't fetch if auth is still loading
    if (isAuthLoading) {
      console.log("[useAgents] Waiting for auth to complete...");
      return;
    }

    // Only fetch if authenticated
    if (isAuthenticated) {
      console.log("[useAgents] Auth ready, fetching agents...");
      fetchAgents();
    } else {
      console.log("[useAgents] Not authenticated, skipping fetch");
    }
  }, [isAuthenticated, isAuthLoading]);

  // Handle WebSocket updates
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "agent_update") {
      const data = lastMessage.data as { agentId: string } & Partial<Agent>;
      const { agentId, ...updates } = data;
      setAgents((prev) =>
        prev.map((agent) => (agent.id === agentId ? { ...agent, ...updates } : agent)),
      );
    } else if (lastMessage.type === "agent_created") {
      const newAgent = lastMessage.data as Agent;
      setAgents((prev) => [...prev, newAgent]);
    } else if (lastMessage.type === "agent_deleted") {
      const data = lastMessage.data as { agentId: string };
      setAgents((prev) => prev.filter((agent) => agent.id !== data.agentId));
    }
  }, [lastMessage]);

  const fetchAgents = async () => {
    try {
      setIsLoading(true);
      setError(null);

      console.log("[useAgents] Fetching agents from API...");
      const data = await apiGet("/api/agents");
      console.log("[useAgents] Fetched agents:", data.agents?.length || 0);
      setAgents(data.agents || []);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err);
      } else {
        setError(new Error(err instanceof Error ? err.message : "Failed to fetch agents"));
      }
      console.error("Failed to fetch agents:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const createAgent = useCallback(async (params: CreateAgentParams): Promise<Agent> => {
    try {
      setIsLoading(true);
      setError(null);

      const newAgent = await apiPost("/api/agents", params);

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

      // Using POST instead of PATCH as apiPatch is not available
      const response = await fetch(`${API_BASE_URL}/api/agents/${id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          ...(typeof window !== "undefined" && localStorage.getItem("fa.jwt")
            ? { Authorization: `Bearer ${localStorage.getItem("fa.jwt")}` }
            : {}),
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

      await apiDelete(`/api/agents/${id}`);

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
