import { useAppSelector } from "@/store/hooks";
import { useEffect, useState } from "react";
import { socketService } from "@/services/socketService";

export interface ConversationMessage {
  id: string;
  type: "llm" | "gnn" | "pymdp" | "system";
  content: string;
  timestamp: string;
  agentId?: string;
  metadata?: any;
}

interface UseConversationDataReturn {
  messages: ConversationMessage[];
  isLoading: boolean;
  error: Error | null;
  isConnected: boolean;
}

// Map agent message types to our expected types
function mapMessageType(agentType: string): "llm" | "gnn" | "pymdp" | "system" {
  if (agentType.includes("llm") || agentType.includes("language")) return "llm";
  if (agentType.includes("gnn") || agentType.includes("graph")) return "gnn";
  if (agentType.includes("pymdp") || agentType.includes("active"))
    return "pymdp";
  return "system";
}

export function useConversationData(): UseConversationDataReturn {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Get data from Redux store
  const conversations = useAppSelector(
    (state) => state.conversations.conversations,
  );
  const activeConversationId = useAppSelector(
    (state) => state.conversations.activeConversationId,
  );
  const connectionStatus = useAppSelector(
    (state) => state.connection.status.websocket,
  );
  const agents = useAppSelector((state) => state.agents.agents);

  // Get active conversation or first available conversation
  const activeConversation = activeConversationId
    ? conversations[activeConversationId]
    : Object.values(conversations)[0];

  // Transform Redux messages to our format
  const messages: ConversationMessage[] =
    activeConversation?.messages.map((msg) => ({
      id: msg.id,
      type: mapMessageType(agents[msg.agentId]?.templateId || "system"),
      content: msg.content,
      timestamp: new Date(msg.timestamp).toLocaleTimeString(),
      agentId: msg.agentId,
      metadata: msg.metadata,
    })) || [];

  const isConnected = connectionStatus === "connected";

  useEffect(() => {
    // Initialize WebSocket connection
    if (!socketService.isConnected() && !socketService.getIsConnecting()) {
      socketService.connect();
    }

    // Subscribe to conversation updates if we have an active conversation
    if (activeConversation?.id) {
      socketService.subscribeToConversation(activeConversation.id);
    }

    setIsLoading(false);

    return () => {
      if (activeConversation?.id) {
        socketService.unsubscribeFromConversation(activeConversation.id);
      }
    };
  }, [activeConversation?.id]);

  // Handle connection errors
  useEffect(() => {
    if (connectionStatus === "disconnected") {
      setError(new Error("Connection lost"));
    } else {
      setError(null);
    }
  }, [connectionStatus]);

  return {
    messages,
    isLoading,
    error,
    isConnected,
  };
}
