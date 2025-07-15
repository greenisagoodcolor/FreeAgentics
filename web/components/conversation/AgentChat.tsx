// Temporary AgentChat component to fix TypeScript errors

import React, { useState } from "react";
import type { Agent } from "@/lib/types";

export interface AgentChatProps {
  agent: Agent;
  onSendMessage?: (message: string) => void;
  messages?: Array<{
    id: string;
    content: string;
    role: "user" | "agent";
    timestamp: string;
  }>;
}

export default function AgentChat({ agent, onSendMessage, messages = [] }: AgentChatProps) {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim() && onSendMessage) {
      onSendMessage(input);
      setInput("");
    }
  };

  return (
    <div className="agent-chat">
      <div className="chat-header">
        <h3>{agent.name}</h3>
      </div>
      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <span>{msg.content}</span>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type a message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}
