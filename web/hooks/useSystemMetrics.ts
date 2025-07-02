import { useAppSelector } from "@/store/hooks";
import { useEffect, useState } from "react";
import { socketService } from "@/services/socketService";

export interface SystemMetrics {
  activeAgents: number;
  conversations: number;
  knowledgeNodes: number;
  cpu: number;
  memory: number;
  latency: number;
  status: "online" | "offline" | "degraded";
}

interface UseSystemMetricsReturn {
  metrics: SystemMetrics;
  isLoading: boolean;
  error: Error | null;
}

export function useSystemMetrics(): UseSystemMetricsReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Get data from Redux store
  const agents = useAppSelector((state) => state.agents.agents);
  const conversations = useAppSelector(
    (state) => state.conversations.conversations,
  );
  const knowledgeNodes = useAppSelector((state) => state.knowledge.graph.nodes);
  const analytics = useAppSelector((state) => state.analytics);
  const connectionStatus = useAppSelector(
    (state) => state.connection.status.websocket,
  );
  const latency = useAppSelector((state) => state.connection.status.latency);

  // Calculate metrics from Redux state
  const activeAgents = Object.values(agents).filter(
    (agent) => agent.status === "active" || agent.status === "idle",
  ).length;

  const activeConversations = Object.values(conversations).filter(
    (conv) => conv.isActive,
  ).length;

  const totalKnowledgeNodes = Object.keys(knowledgeNodes).length;

  // Get system status based on connection and activity
  const getSystemStatus = (): "online" | "offline" | "degraded" => {
    if (connectionStatus === "disconnected") return "offline";
    if (connectionStatus === "connecting") return "degraded";
    if (activeAgents === 0) return "degraded";
    return "online";
  };

  const metrics: SystemMetrics = {
    activeAgents,
    conversations: activeConversations,
    knowledgeNodes: totalKnowledgeNodes,
    cpu: Math.random() * 60 + 20, // Mock CPU usage
    memory: Math.random() * 4 + 2, // Mock memory usage in GB
    latency: latency || 0,
    status: getSystemStatus(),
  };

  useEffect(() => {
    // Request system stats from WebSocket
    if (socketService.isConnected()) {
      socketService.getConnectionStats();
    }
  }, [connectionStatus]);

  return {
    metrics,
    isLoading,
    error,
  };
}
