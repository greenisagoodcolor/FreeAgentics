import React, { useState, useEffect, useRef } from "react";
import { getWebSocketClient } from "@/lib/websocket-client";
import type { Agent } from "@/lib/types";

export interface Channel {
  id: string;
  name: string;
  type: "group" | "direct";
  participants: string[];
  unreadCount?: number;
  lastMessage?: {
    id: string;
    agentId: string;
    content: string;
    timestamp: string;
    type: string;
  };
}

export interface AgentChatProps {
  agent: Agent;
  onSendMessage?: (message: string) => void;
  onMessageSent?: (message: any) => void;
  onChannelChange?: (channelId: string) => void;
  messages?: Array<{
    id: string;
    content: string;
    role: "user" | "agent";
    timestamp: string;
  }>;
  channels?: Channel[];
  activeChannelId?: string;
}

export default function AgentChat({
  agent,
  onSendMessage,
  onMessageSent,
  onChannelChange,
  messages = [],
  channels = [],
  activeChannelId = "channel-1",
}: AgentChatProps) {
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("connected");
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  const wsClient = getWebSocketClient();

  // Initialize WebSocket connection
  useEffect(() => {
    let mounted = true;
    let unsubscribeMessage: (() => void) | undefined;
    let unsubscribePresence: (() => void) | undefined;
    let unsubscribeTyping: (() => void) | undefined;

    const initWebSocket = async () => {
      await wsClient.connect();

      if (!mounted) return;

      wsClient.send({
        type: "subscribe_agent_chat",
        agent_id: agent.id,
      });

      // Subscribe to events
      unsubscribeMessage = wsClient.subscribe("agent_chat_message", (data: any) => {
        console.log("Received agent message:", data);
      });

      unsubscribePresence = wsClient.subscribe("agent_presence_update", (data: any) => {
        console.log("Presence update:", data);
      });

      unsubscribeTyping = wsClient.subscribe("agent_typing", (data: any) => {
        console.log("Agent typing:", data);
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
      if (unsubscribePresence) unsubscribePresence();
      if (unsubscribeTyping) unsubscribeTyping();
    };
  }, [agent.id, wsClient]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => {
    setInput(e.target.value);

    // Send typing indicator
    if (!isTyping && e.target.value.length > 0) {
      setIsTyping(true);
      wsClient.send({
        type: "agent_typing",
        agent_id: agent.id,
        channel_id: activeChannelId,
        is_typing: true,
      });
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set new timeout to stop typing indicator
    if (e.target.value.length > 0) {
      typingTimeoutRef.current = setTimeout(() => {
        setIsTyping(false);
        wsClient.send({
          type: "agent_typing",
          agent_id: agent.id,
          channel_id: activeChannelId,
          is_typing: false,
        });
      }, 3000);
    }
  };

  const handleSend = () => {
    if (input.trim()) {
      const messageData = {
        channel_id: activeChannelId,
        message: {
          content: input,
          agentId: agent.id,
          type: "text",
        },
      };

      wsClient.send({
        type: "send_agent_message",
        data: messageData,
      });

      if (onMessageSent) {
        onMessageSent(messageData);
      }

      if (onSendMessage) {
        onSendMessage(input);
      }

      setInput("");
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleChannelClick = (channelId: string) => {
    if (onChannelChange) {
      onChannelChange(channelId);
    }

    // Mark channel as read
    wsClient.send({
      type: "mark_channel_read",
      agent_id: agent.id,
      channel_id: channelId,
    });
  };

  // Mock channels for testing
  const testChannels: Channel[] =
    channels.length > 0
      ? channels
      : [
          {
            id: "channel-1",
            name: "Test Channel 1",
            type: "group",
            participants: [agent.id],
          },
          {
            id: "channel-2",
            name: "Direct Chat",
            type: "direct",
            participants: [agent.id],
          },
        ];

  return (
    <div className="agent-chat">
      <div className="chat-header">
        <h3>{agent.name}</h3>
        <span className="connection-status">{connectionStatus}</span>
      </div>

      <div className="channels">
        {testChannels.map((channel) => (
          <button
            key={channel.id}
            onClick={() => handleChannelClick(channel.id)}
            className={activeChannelId === channel.id ? "active" : ""}
          >
            {channel.name}
          </button>
        ))}
      </div>

      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <span>{msg.content}</span>
          </div>
        ))}
      </div>

      <div className="chat-input">
        <textarea
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}
