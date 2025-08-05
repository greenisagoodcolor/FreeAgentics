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
  goalPrompt: string | null;
  setGoalPrompt: (prompt: string | null) => void;
}

export function useConversation(): ConversationState {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [goalPrompt, setGoalPrompt] = useState<string | null>(null);

  const { sendMessage: sendWebSocketMessage, lastMessage, isConnected } = useWebSocket();
  const messageIdCounter = useRef(0);
  const loadingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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
        content: data.message || "Agent created successfully",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, agentMessage]);
      // Clear loading state on successful agent creation
      setIsLoading(false);
    } else if (lastMessage.type === "prompt_acknowledged") {
      // Handle prompt acknowledgment
      const data = lastMessage.data as { message?: string; conversation_id?: string };
      console.log("[Conversation] Prompt acknowledged:", data);
      // Don't clear loading here, wait for actual response
    } else if (lastMessage.type === "error") {
      const errorData = lastMessage.data as { message?: string };
      setError(new Error(errorData.message || "An error occurred"));
      setIsLoading(false);
    }
  }, [lastMessage]);

  // Loading timeout fallback - ensure loading doesn't persist indefinitely
  useEffect(() => {
    if (isLoading) {
      // Set timeout to clear loading after 5 seconds
      loadingTimeoutRef.current = setTimeout(() => {
        console.warn("[Conversation] Loading timeout exceeded, clearing loading state");
        setIsLoading(false);
        setError(new Error("Request timed out. Please try again."));
      }, 5000);
    } else {
      // Clear timeout when loading stops
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
    }

    // Cleanup on unmount
    return () => {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
        loadingTimeoutRef.current = null;
      }
    };
  }, [isLoading]);

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
        // Get the auth token
        const token = localStorage.getItem("fa.jwt");
        const headers: HeadersInit = {
          "Content-Type": "application/json",
        };

        // Only add auth header if token exists
        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }

        fetch("/api/process-prompt", {
          method: "POST",
          headers,
          body: JSON.stringify({
            prompt: content,
            conversationId: conversationId || "default",
          }),
        })
          .then((response) => {
            if (!response.ok) {
              return response.text().then((text) => {
                throw new Error(`Failed to process prompt: ${text}`);
              });
            }
            return response.json();
          })
          .then((data) => {
            console.log("[Conversation] Prompt processed:", data);
          })
          .catch((error) => {
            console.error("[Conversation] Failed to send prompt:", error);
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
    setGoalPrompt(null);

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
    goalPrompt,
    setGoalPrompt,
  };
}
