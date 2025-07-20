import React, { useState, useCallback, useRef, useEffect } from "react";
import type { Agent } from "@/lib/types";

export interface Message {
  id: string;
  content: string;
  role: "user" | "agent";
  timestamp: string;
}

export interface AccessibleAgentChatProps {
  agent: Agent;
  onSendMessage?: (message: string) => void;
  messages?: Message[];
  isLoading?: boolean;
}

export default function AccessibleAgentChat({
  agent,
  onSendMessage,
  messages = [],
  isLoading = false,
}: AccessibleAgentChatProps) {
  const [input, setInput] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Announce new messages to screen readers
  useEffect(() => {
    if (messages.length > 0) {
      const latestMessage = messages[messages.length - 1];
      setAnnouncement(`New ${latestMessage.role} message: ${latestMessage.content}`);
    }
  }, [messages]);

  const handleSend = useCallback(() => {
    if (input.trim() && onSendMessage && !isLoading) {
      onSendMessage(input);
      setInput("");
      // Keep focus on input after sending
      inputRef.current?.focus();
    }
  }, [input, onSendMessage, isLoading]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div
      className="agent-chat flex flex-col h-full"
      role="region"
      aria-label={`Chat with ${agent.name}`}
    >
      {/* Screen reader announcements */}
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </div>

      {/* Chat header */}
      <header className="chat-header p-4 border-b">
        <h2 className="text-lg font-semibold" id={`chat-title-${agent.id}`}>
          {agent.name}
        </h2>
        <p className="text-sm text-muted-foreground">
          Status: <span className="font-medium">{agent.status}</span>
        </p>
      </header>

      {/* Messages area */}
      <div
        className="chat-messages flex-1 overflow-y-auto p-4 space-y-4"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
        aria-describedby={`chat-title-${agent.id}`}
      >
        {messages.length === 0 ? (
          <p className="text-muted-foreground text-center">
            No messages yet. Start a conversation!
          </p>
        ) : (
          messages.map((msg, index) => (
            <article
              key={msg.id}
              className={`message flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              aria-label={`${msg.role} message ${index + 1} of ${messages.length}`}
            >
              <div
                className={`
                  max-w-[70%] rounded-lg p-3 
                  ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }
                `}
              >
                <p className="break-words">{msg.content}</p>
                <time dateTime={msg.timestamp} className="text-xs opacity-70 mt-1 block">
                  {new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </time>
              </div>
            </article>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="chat-input p-4 border-t"
        aria-label="Message input form"
      >
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            disabled={isLoading}
            aria-label="Type your message"
            aria-describedby={isLoading ? "loading-indicator" : undefined}
            className="flex-1 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
          />
          <button
            type="submit"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            aria-label="Send message"
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <span className="sr-only">Sending...</span>
                <span aria-hidden="true">...</span>
              </>
            ) : (
              "Send"
            )}
          </button>
        </div>
        {isLoading && (
          <p id="loading-indicator" className="text-sm text-muted-foreground mt-2">
            Agent is typing...
          </p>
        )}
      </form>
    </div>
  );
}
