"use client";

import React from "react";
import { DashboardView } from "../../page";
import AgentPanel from "../components/panels/AgentPanel";
import AnalyticsPanel from "../components/panels/AnalyticsPanel";
import KnowledgePanel from "../components/panels/KnowledgePanel";

interface KnowledgeLayoutProps {
  view: DashboardView;
}

export default function KnowledgeLayout({ view }: KnowledgeLayoutProps) {
  return (
    <div className="knowledge-layout h-full bg-primary">
      {/* Research-focused Layout */}
      <div className="h-full flex gap-1 p-1">
        {/* Main Knowledge Graph */}
        <div className="flex-1">
          <div className="card h-full">
            <div className="card-header border-b border-primary-amber">
              <h2 className="card-title text-primary-amber">KNOWLEDGE GRAPH</h2>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="status-dot active"></div>
                  <span className="text-xs font-mono text-text-secondary">
                    LIVE UPDATES
                  </span>
                </div>
                <button className="button button-xs button-secondary">
                  Export
                </button>
              </div>
            </div>
            <div className="card-content p-0">
              <KnowledgePanel view={view} />
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="w-80 flex flex-col gap-1">
          {/* Analytics Panel */}
          <div className="card h-1/2">
            <div className="card-header">
              <h2 className="card-title text-primary-amber">ANALYTICS</h2>
            </div>
            <div className="card-content p-0">
              <AnalyticsPanel view={view} />
            </div>
          </div>

          {/* Agent Control Panel */}
          <div className="card h-1/2">
            <div className="card-header">
              <h2 className="card-title text-primary-amber">RESEARCH AGENTS</h2>
            </div>
            <div className="card-content p-0">
              <AgentPanel view={view} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
