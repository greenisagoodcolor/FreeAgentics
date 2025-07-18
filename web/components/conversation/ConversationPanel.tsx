import React, { useState, useEffect, useRef } from "react";
import { getWebSocketClient } from "@/lib/websocket-client";
import { useAgentConversation } from "@/hooks/use-agent-conversation";
import type { Agent } from "@/lib/types";

export interface ConversationMessage {
  id: string;
  agentId: string;
  agent_id: string; // snake_case for compatibility
  user_id?: string;
  conversation_id?: string; // for compatibility
  message: string;
  content: string; // content field for compatibility
  timestamp: string;
  type: "user" | "agent" | "system";
  message_type: string; // snake_case for compatibility
  metadata?: Record<string, any>;
}

export interface ConversationPanelProps {
  conversationId: string;
  currentUser: {
    id: string;
    name: string;
    avatar?: string;
  };
  agents: Agent[];
  messages?: ConversationMessage[];
  onSendMessage?: (message: string) => void;
  onMessageSent?: (message: any) => void;
  isLoading?: boolean;
}

export default function ConversationPanel({
  conversationId,
  currentUser,
  agents,
  messages = [],
  onSendMessage,
  onMessageSent,
  isLoading = false,
}: ConversationPanelProps) {
  const [messageInput, setMessageInput] = useState("");
  const [selectedAgent, setSelectedAgent] = useState<string>("");
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [isTyping, setIsTyping] = useState(false);
  const [lastActivity, setLastActivity] = useState(new Date());
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  const wsClient = getWebSocketClient();
  const { sendMessage, getConversationHistory } = useAgentConversation();

  // Initialize WebSocket connection
  useEffect(() => {
    let mounted = true;
    let unsubscribeMessage: (() => void) | undefined;
    let unsubscribeChunk: (() => void) | undefined;
    let unsubscribeTyping: (() => void) | undefined;

    const initWebSocket = async () => {
      await wsClient.connect();

      if (!mounted) return;

      wsClient.send({
        type: "subscribe_conversation",
        conversation_id: conversationId,
      });

      // Subscribe to events
      unsubscribeMessage = wsClient.subscribe("conversation_message", (data: any) => {
        // Handle incoming messages
        console.log("Received message:", data);
        if (mounted) {
          setLastActivity(new Date());
        }
      });

      unsubscribeChunk = wsClient.subscribe("llm_response_chunk", (data: any) => {
        // Handle LLM response chunks
        console.log("Received chunk:", data);
      });

      unsubscribeTyping = wsClient.subscribe("user_typing", (data: any) => {
        // Handle typing indicators
        console.log("User typing:", data);
      });

      // Update connection status
      const status = wsClient.getConnectionState();
      if (mounted) {
        setConnectionStatus(status);
      }
    };

    initWebSocket();

    return () => {
      mounted = false;
      if (unsubscribeMessage) unsubscribeMessage();
      if (unsubscribeChunk) unsubscribeChunk();
      if (unsubscribeTyping) unsubscribeTyping();
    };
  }, [conversationId, wsClient]);

  // Load conversation history when agent is selected
  useEffect(() => {
    if (selectedAgent) {
      const history = getConversationHistory(selectedAgent);
      console.log("Loaded history:", history);
    }
  }, [selectedAgent, getConversationHistory]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessageInput(e.target.value);

    // Send typing indicator
    if (!isTyping) {
      setIsTyping(true);
      wsClient.send({
        type: "user_typing",
        conversation_id: conversationId,
        is_typing: true,
      });
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set new timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      wsClient.send({
        type: "user_typing",
        conversation_id: conversationId,
        is_typing: false,
      });
    }, 3000);
  };

  const sendMessageHandler = async () => {
    if (!messageInput.trim()) return;

    const messageData = {
      conversation_id: conversationId,
      content: messageInput,
      message_type: "user",
      user_id: currentUser.id,
    };

    if (selectedAgent) {
      // Use API when agent is selected
      await sendMessage(messageInput);
    } else {
      // Use WebSocket for general conversation
      wsClient.send({
        type: "send_message",
        data: messageData,
      });
    }

    if (onMessageSent) {
      onMessageSent(messageData);
    }

    if (onSendMessage) {
      onSendMessage(messageInput);
    }

    setMessageInput("");
    setIsTyping(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessageHandler();
    }
  };

  const participantCount = agents.length + 1; // agents + current user

  return (
    <div className="conversation-panel">
      <div className="conversation-header">
        <h2>Conversation</h2>
        <div className="status-info">
          <span className="connection-status">{connectionStatus}</span>
          <span className="participant-count">{participantCount} participants</span>
          <span className="last-activity">Last activity: {lastActivity.toLocaleTimeString()}</span>
        </div>
      </div>

      <div className="agent-selector">
        <select value={selectedAgent} onChange={(e) => setSelectedAgent(e.target.value)}>
          <option value="">Select Agent</option>
          {agents.map((agent) => (
            <option key={agent.id} value={agent.id}>
              {agent.name}
            </option>
          ))}
        </select>
      </div>

      {connectionStatus === "disconnected" && (
        <div className="connection-warning">
          Connection lost. Messages will be queued until reconnected.
        </div>
      )}

      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.type}`}>
            <span className="timestamp">{msg.timestamp}</span>
            <span className="content">{msg.content || msg.message}</span>
          </div>
        ))}
      </div>

      <div className="message-input-area">
        <textarea
          placeholder="Type your message..."
          value={messageInput}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        <button onClick={sendMessageHandler} disabled={isLoading || !messageInput.trim()}>
          Send
        </button>
      </div>

      {isLoading && <div>Loading...</div>}
    </div>
  );
}
