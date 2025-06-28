"use client";

import React from "react";
import { DashboardView } from "../../page";
import AgentPanel from "../components/panels/AgentPanel";
import ConversationPanel from "../components/panels/ConversationPanel";
import AnalyticsPanel from "../components/panels/AnalyticsPanel";
import KnowledgePanel from "../components/panels/KnowledgePanel";

interface ResizableLayoutProps {
  view: DashboardView;
}

export default function ResizableLayout({ view }: ResizableLayoutProps) {
  return (
    <div
      className="resizable-layout h-full"
      style={{ background: "var(--bg-primary)" }}
    >
      {/* Technical Dashboard Layout */}
      <div className="flex h-full gap-1 p-1">
        {/* Left Panel: Agent Management */}
        <div className="w-1/4 flex flex-col gap-1">
          <div className="card h-1/2">
            <div className="card-header">
              <h2
                className="card-title"
                style={{ color: "var(--primary-amber)" }}
              >
                AGENTS
              </h2>
            </div>
            <div className="card-content p-0">
              <AgentPanel view={view} />
            </div>
          </div>

          <div className="card h-1/2">
            <div className="card-header">
              <h2
                className="card-title"
                style={{ color: "var(--primary-amber)" }}
              >
                ANALYTICS
              </h2>
            </div>
            <div className="card-content p-0">
              <AnalyticsPanel view={view} />
            </div>
          </div>
        </div>

        {/* Center Panel: Conversation */}
        <div className="flex-1">
          <div className="card h-full">
            <div
              className="card-header"
              style={{ borderBottom: "1px solid var(--primary-amber)" }}
            >
              <h2
                className="card-title"
                style={{ color: "var(--primary-amber)" }}
              >
                CONVERSATION
              </h2>
              <div className="flex items-center gap-2">
                <div className="status-dot active"></div>
                <span
                  className="text-xs font-mono"
                  style={{ color: "var(--text-secondary)" }}
                >
                  LIVE
                </span>
              </div>
            </div>
            <div className="card-content p-0">
              <ConversationPanel view={view} />
            </div>
          </div>
        </div>

        {/* Right Panel: Knowledge Graph */}
        <div className="w-1/3">
          <div className="card h-full">
            <div
              className="card-header"
              style={{ borderBottom: "1px solid var(--primary-amber)" }}
            >
              <h2
                className="card-title"
                style={{ color: "var(--primary-amber)" }}
              >
                KNOWLEDGE
              </h2>
            </div>
            <div className="card-content p-0">
              <KnowledgePanel view={view} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
