import { useState, useCallback, useEffect, useRef } from "react";
import { useWebSocket } from "./use-websocket";

export type MessageRole = "user" | "assistant" | "system";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  metadata?: Record<string, unknown>;
}

export interface SendMessageParams {
  content: string;
  role?: MessageRole;
}

export interface ConversationState {
  messages: Message[];
  sendMessage: (params: SendMessageParams) => void;
  isLoading: boolean;
  error: Error | null;
  conversationId: string | null;
  clearConversation: () => void;
}

export function useConversation(): ConversationState {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const { sendMessage: sendWebSocketMessage, lastMessage, isConnected } = useWebSocket();
  const messageIdCounter = useRef(0);

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "conversation_started") {
      const data = lastMessage.data as { conversationId: string };
      setConversationId(data.conversationId);
    } else if (lastMessage.type === "connection_established") {
      // Initialize conversation ID if not set
      if (!conversationId) {
        setConversationId("default-" + Date.now());
      }
    } else if (lastMessage.type === "message") {
      const messageData = lastMessage.data as {
        id: string;
        role: MessageRole;
        content: string;
        timestamp: string;
        isStreaming?: boolean;
      };
      const { id, role, content, timestamp, isStreaming } = messageData;

      setMessages((prev) => {
        const existingIndex = prev.findIndex((msg) => msg.id === id);

        if (existingIndex >= 0) {
          // Update existing message (for streaming)
          const updated = [...prev];
          updated[existingIndex] = {
            ...updated[existingIndex],
            content,
            isStreaming,
          };
          return updated;
        } else {
          // Add new message
          return [
            ...prev,
            {
              id,
              role,
              content,
              timestamp: timestamp || new Date().toISOString(),
              isStreaming,
            },
          ];
        }
      });

      if (!isStreaming) {
        setIsLoading(false);
      }
    } else if (lastMessage.type === "agent_created") {
      // Handle agent creation notifications
      const data = lastMessage.data as { agent: any; message: string };
      console.log("[Conversation] Agent created:", data);
      // Optionally add a system message about the agent creation
      const agentMessage = {
        id: `system-${Date.now()}`,
        role: "system" as MessageRole,
        content: data.message,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, agentMessage]);
    } else if (lastMessage.type === "error") {
      const errorData = lastMessage.data as { message?: string };
      setError(new Error(errorData.message || "An error occurred"));
      setIsLoading(false);
    }
  }, [lastMessage]);

  const sendMessage = useCallback(
    (params: SendMessageParams) => {
      if (!isConnected) {
        setError(new Error("Not connected to server"));
        return;
      }

      const { content, role = "user" } = params;

      if (!content.trim()) return;

      setIsLoading(true);
      setError(null);

      // Generate a temporary ID
      const tempId = `temp-${++messageIdCounter.current}`;
      const timestamp = new Date().toISOString();

      // Optimistically add the user message
      if (role === "user") {
        setMessages((prev) => [
          ...prev,
          {
            id: tempId,
            role,
            content,
            timestamp,
          },
        ]);
      }

      // For HTTP-based flow, we'll use the process-prompt endpoint
      // The backend will send the response via WebSocket
      if (role === "user") {
        fetch("/api/process-prompt", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("fa.jwt")}`,
          },
          body: JSON.stringify({
            prompt: content,
            conversationId: conversationId || "default",
          }),
        })
          .then((response) => {
            if (!response.ok) {
              throw new Error("Failed to process prompt");
            }
            return response.json();
          })
          .catch((error) => {
            console.error("Failed to send prompt:", error);
            setError(error);
            setIsLoading(false);
          });
      } else {
        // For non-user messages, still use WebSocket directly
        sendWebSocketMessage({
          type: "message",
          data: {
            conversationId,
            content,
            role,
            timestamp,
          },
        });
      }
    },
    [isConnected, conversationId, sendWebSocketMessage],
  );

  const clearConversation = useCallback(() => {
    setMessages([]);
    setConversationId(null);
    setError(null);

    // Notify server
    if (isConnected && conversationId) {
      sendWebSocketMessage({
        type: "clear_conversation",
        data: { conversationId },
      });
    }
  }, [isConnected, conversationId, sendWebSocketMessage]);

  return {
    messages,
    sendMessage,
    isLoading,
    error,
    conversationId,
    clearConversation,
  };
}
