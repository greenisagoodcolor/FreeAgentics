"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  MessageSquare,
  Brain,
  Network,
  Activity,
  ChevronRight,
} from "lucide-react";

// Define DashboardView type locally
export type DashboardView =
  | "ceo-demo"
  | "executive"
  | "technical"
  | "research"
  | "minimal";

interface ConversationPanelProps {
  view: DashboardView;
}

interface ConversationMessage {
  id: string;
  type: "llm" | "gnn" | "pymdp" | "system";
  content: string;
  timestamp: string; // Changed to string for consistent formatting
  metadata?: any;
}

// Use static timestamps with consistent formatting to avoid hydration mismatches
const mockMessages: ConversationMessage[] = [
  {
    id: "1",
    type: "system",
    content: "Agent initialized with goal: Optimize resource allocation",
    timestamp: "12:59:50 PM",
  },
  {
    id: "2",
    type: "llm",
    content: "Analyzing current resource distribution patterns...",
    timestamp: "12:59:51 PM",
  },
  {
    id: "3",
    type: "gnn",
    content: "Graph neural network processing: 847 nodes, 2,341 edges analyzed",
    timestamp: "12:59:52 PM",
  },
  {
    id: "4",
    type: "pymdp",
    content: "Active inference update: Free energy reduced by 23.4%",
    timestamp: "12:59:53 PM",
  },
];

export default function ConversationPanel({ view }: ConversationPanelProps) {
  const [messages, setMessages] = useState<ConversationMessage[]>(mockMessages);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const getMessageIcon = (type: ConversationMessage["type"]) => {
    switch (type) {
      case "llm":
        return <Brain className="w-4 h-4" />;
      case "gnn":
        return <Network className="w-4 h-4" />;
      case "pymdp":
        return <Activity className="w-4 h-4" />;
      case "system":
        return <MessageSquare className="w-4 h-4" />;
    }
  };

  const getMessageColor = (type: ConversationMessage["type"]) => {
    switch (type) {
      case "llm":
        return "text-blue-400 border-blue-400/20 bg-blue-400/5";
      case "gnn":
        return "text-green-400 border-green-400/20 bg-green-400/5";
      case "pymdp":
        return "text-purple-400 border-purple-400/20 bg-purple-400/5";
      case "system":
        return "text-[var(--text-tertiary)] border-[var(--bg-tertiary)] bg-[var(--bg-tertiary)]";
    }
  };

  const getTypeLabel = (type: ConversationMessage["type"]) => {
    switch (type) {
      case "llm":
        return "LLM";
      case "gnn":
        return "GNN";
      case "pymdp":
        return "PyMDP";
      case "system":
        return "SYSTEM";
    }
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-[var(--bg-tertiary)]">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs font-mono text-[var(--text-secondary)]">
              ACTIVE
            </span>
          </div>
          <span className="text-xs font-mono text-[var(--text-tertiary)]">
            Agent Loop
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-blue-400">LLM</span>
          <ChevronRight className="w-3 h-3 text-[var(--text-tertiary)]" />
          <span className="text-green-400">GNN</span>
          <ChevronRight className="w-3 h-3 text-[var(--text-tertiary)]" />
          <span className="text-purple-400">PyMDP</span>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin"
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start gap-3 p-3 rounded-lg border ${getMessageColor(
              message.type,
            )} transition-all hover:border-opacity-40`}
          >
            <div className="flex-shrink-0 mt-0.5">
              {getMessageIcon(message.type)}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-semibold">
                  {getTypeLabel(message.type)}
                </span>
                <span className="text-xs font-mono text-[var(--text-tertiary)]">
                  {message.timestamp}
                </span>
              </div>
              <p className="text-sm text-[var(--text-primary)] break-words">
                {message.content}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Input Area (Future) */}
      <div className="p-3 border-t border-[var(--bg-tertiary)]">
        <div className="flex items-center gap-2 text-xs font-mono text-[var(--text-tertiary)]">
          <Activity className="w-3 h-3" />
          <span>Processing next inference cycle...</span>
        </div>
      </div>
    </div>
  );
}
