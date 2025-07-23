/**
 * Custom hook for managing agent conversations
 */

import { useState, useCallback } from "react";

interface ConversationMessage {
  id: string;
  message: string;
  agentId?: string;
  timestamp: string;
  sender: 'user' | 'agent' | 'system';
}

export interface UseAgentConversationReturn {
  sendMessage: (message: string, agentId?: string) => Promise<void>;
  getConversationHistory: (agentId?: string) => ConversationMessage[];
  createSession: (agentId: string) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

export const useAgentConversation = (): UseAgentConversationReturn => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationHistory, setConversationHistory] = useState<ConversationMessage[]>([]);

  const sendMessage = useCallback(async (message: string, agentId?: string) => {
    setIsLoading(true);
    setError(null);

    try {
      // Mock implementation for now
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Add message to history
      const newMessage = {
        id: Date.now().toString(),
        message,
        agentId,
        timestamp: new Date().toISOString(),
        sender: "user",
      };

      setConversationHistory((prev) => [...prev, newMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getConversationHistory = useCallback(
    (agentId?: string) => {
      if (agentId) {
        return conversationHistory.filter((msg) => msg.agentId === agentId);
      }
      return conversationHistory;
    },
    [conversationHistory],
  );

  const createSession = useCallback(async (agentId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      // Mock implementation for now
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Initialize session
      const sessionMessage = {
        id: Date.now().toString(),
        message: `Session created for agent ${agentId}`,
        agentId,
        timestamp: new Date().toISOString(),
        sender: "system",
      };

      setConversationHistory((prev) => [...prev, sessionMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    sendMessage,
    getConversationHistory,
    createSession,
    isLoading,
    error,
  };
};
