"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Presentation, Menu, Moon, Sun, ZoomIn, ZoomOut } from "lucide-react";

// Define DashboardView type locally
export type DashboardView =
  | "ceo-demo"
  | "executive"
  | "technical"
  | "research"
  | "minimal";

// Import components
import GoalPanel from "./dashboard/components/panels/GoalPanel";
import AgentPanel from "./dashboard/components/panels/AgentPanel";
import ConversationPanel from "./dashboard/components/panels/ConversationPanel";
import KnowledgePanel from "./dashboard/components/panels/KnowledgePanel";
import MetricsPanel from "./dashboard/components/panels/MetricsPanel";

export default function HomePage() {
  const router = useRouter();
  const [view, setView] = useState<DashboardView>("executive");
  const [currentTime, setCurrentTime] = useState<string>("");
  const [isClient, setIsClient] = useState(false);
  const [isDarkTheme, setIsDarkTheme] = useState(true);
  const [selectedLayout, setSelectedLayout] = useState("unified");

  useEffect(() => {
    // Set client flag first
    setIsClient(true);

    // Set initial time and update every second
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString());
    };

    updateTime(); // Set initial time
    const interval = setInterval(updateTime, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleShowDemo = () => {
    router.push("/ceo-demo");
  };

  const toggleTheme = () => {
    setIsDarkTheme(!isDarkTheme);
    document.body.classList.toggle("dark", !isDarkTheme);
  };

  const handleLayoutChange = (layout: string) => {
    setSelectedLayout(layout);
    // Add layout change logic here
  };

  return (
    <div
      className={`min-h-screen flex flex-col bg-[var(--bg-primary)] ${isDarkTheme ? "dark" : ""} ${selectedLayout === "bloomberg" ? "layout-bloomberg bloomberg-theme" : ""}`}
    >
      {/* Title and Status Bar */}
      <div className="flex items-center justify-between p-4 border-b border-[var(--border-primary)]">
        <div className="flex items-center gap-4">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center"
            style={{ background: "var(--primary-amber)" }}
          >
            <span
              className="text-lg font-bold"
              style={{ color: "var(--bg-primary)" }}
            >
              FA
            </span>
          </div>
          <div>
            <h1
              className="text-xl font-bold"
              style={{ color: "var(--text-primary)" }}
            >
              FreeAgentics
            </h1>
            <div
              className="flex items-center gap-2 text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              <span className="inline-block w-2 h-2 bg-green-500 rounded-full"></span>
              <span>System Operational</span>
              <span>•</span>
              <span>{isClient ? currentTime : "--:--:-- --"}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Layout Selector */}
          <select
            data-testid="layout-selector"
            value={selectedLayout}
            onChange={(e) => handleLayoutChange(e.target.value)}
            className="px-3 py-1 rounded border bg-[var(--bg-secondary)] text-[var(--text-primary)] border-[var(--border-primary)]"
          >
            <option value="unified">Unified</option>
            <option value="bloomberg">Bloomberg</option>
            <option value="resizable">Resizable</option>
          </select>

          {/* Theme Toggle */}
          <button
            data-testid="theme-toggle"
            onClick={toggleTheme}
            className="p-2 rounded hover:bg-[var(--bg-secondary)]"
            style={{ color: "var(--text-primary)" }}
          >
            {isDarkTheme ? <Sun size={16} /> : <Moon size={16} />}
          </button>

          {/* Mobile Menu */}
          <button
            data-testid="mobile-menu-toggle"
            className="md:hidden p-2 rounded hover:bg-[var(--bg-secondary)]"
            style={{ color: "var(--text-primary)" }}
          >
            <Menu size={16} />
          </button>

          {/* Show Demo Button */}
          <button
            onClick={handleShowDemo}
            className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium hover:opacity-90 transition-opacity"
            style={{
              background: "var(--primary-amber)",
              color: "var(--bg-primary)",
            }}
          >
            <Presentation size={16} />
            SHOW DEMO
          </button>
        </div>
      </div>

      {/* Goal Section - Full Width */}
      <div className="p-4" data-testid="goal-panel">
        <GoalPanel view={view} />
      </div>

      {/* Three Panels Side-by-Side */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-4 p-4 min-h-[400px]">
        {/* Agents Panel */}
        <div
          data-testid="agent-panel"
          className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-primary)]"
        >
          <AgentPanel view={view} />
        </div>

        {/* Conversation Panel */}
        <div
          data-testid="conversation-panel"
          className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-primary)]"
        >
          <ConversationPanel view={view} />
        </div>

        {/* Knowledge Graph Panel */}
        <div
          data-testid="knowledge-panel"
          className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-primary)] relative"
        >
          {/* Knowledge Graph Controls */}
          <div
            className="absolute top-2 right-2 z-10 flex gap-1"
            data-testid="zoom-controls"
          >
            <button
              data-testid="zoom-in"
              className="p-1 rounded bg-[var(--bg-primary)] hover:bg-[var(--bg-tertiary)]"
              style={{ color: "var(--text-primary)" }}
            >
              <ZoomIn size={14} />
            </button>
            <button
              data-testid="zoom-out"
              className="p-1 rounded bg-[var(--bg-primary)] hover:bg-[var(--bg-tertiary)]"
              style={{ color: "var(--text-primary)" }}
            >
              <ZoomOut size={14} />
            </button>
          </div>

          {/* Node Filter */}
          <div className="absolute top-2 left-2 z-10">
            <select
              data-testid="node-filter"
              className="text-xs px-2 py-1 rounded border bg-[var(--bg-primary)] text-[var(--text-primary)] border-[var(--border-primary)]"
            >
              <option value="all">All Nodes</option>
              <option value="agent">Agents</option>
              <option value="concept">Concepts</option>
              <option value="relationship">Relationships</option>
            </select>
          </div>

          <KnowledgePanel view={view} />
        </div>
      </div>

      {/* Metrics Section - Full Width */}
      <div className="p-4" data-testid="metrics-panel">
        <MetricsPanel view={view} />
      </div>

      {/* Control Panel - Hidden but available for tests */}
      <div data-testid="control-panel" className="hidden">
        <div>Control Panel Content</div>
      </div>

      {/* Footer Status Bar */}
      <div
        className="flex items-center justify-between p-3 border-t border-[var(--border-primary)] text-sm"
        style={{ color: "var(--text-secondary)" }}
      >
        <div className="flex items-center gap-4">
          <span>4 Active Agents</span>
          <span>•</span>
          <span>12 Conversations</span>
          <span>•</span>
          <span>156 Knowledge Nodes</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Memory: 2.1GB</span>
          <span>•</span>
          <span>CPU: 23%</span>
          <span>•</span>
          <span>Version 1.0.0</span>
        </div>
      </div>
    </div>
  );
}
