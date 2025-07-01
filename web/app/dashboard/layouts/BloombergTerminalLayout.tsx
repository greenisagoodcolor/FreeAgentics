"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import TiledPanel from "@/components/dashboard/TiledPanel";
import GoalPanel from "../components/panels/GoalPanel";
import ConversationPanel from "../components/panels/ConversationPanel";
import KnowledgePanel from "../components/panels/KnowledgePanel";
import AnalyticsPanel from "../components/panels/AnalyticsPanel";
import { DashboardView } from "../../page";
import { Presentation } from "lucide-react";

interface BloombergTerminalLayoutProps {
  view: DashboardView;
}

export default function BloombergTerminalLayout({
  view,
}: BloombergTerminalLayoutProps) {
  const router = useRouter();
  const [focusedPanel, setFocusedPanel] = useState<string | null>(null);

  return (
    <div className="h-full flex flex-col gap-2 p-2 bg-[var(--bg-primary)]">
      {/* CEO Demo Button - Fixed Position */}
      <div className="absolute top-4 right-4 z-50">
        <button
          onClick={() => router.push("/dashboard?view=ceo-demo")}
          className="px-4 py-2 bg-[var(--bg-secondary)] border border-[var(--primary-amber)] rounded-lg flex items-center gap-2 hover:bg-[var(--bg-tertiary)] transition-colors"
        >
          <Presentation className="w-4 h-4 text-[var(--primary-amber)]" />
          <span className="text-sm font-bold text-[var(--primary-amber)] font-mono">
            CEO DEMO
          </span>
        </button>
      </div>

      {/* Goal Panel - Fixed Height */}
      <div className="h-[100px] flex-shrink-0">
        <TiledPanel
          id="goal"
          title="AGENT GOAL"
          closable={false}
          detachable={false}
          focused={focusedPanel === "goal"}
          onFocus={() => setFocusedPanel("goal")}
          className="h-full"
        >
          <GoalPanel view={view} />
        </TiledPanel>
      </div>

      {/* Main Content Area - Flexible Height */}
      <div className="flex-1 grid grid-cols-2 gap-2 min-h-0">
        {/* Conversation Panel - Left */}
        <TiledPanel
          id="conversation"
          title="AGENT CONVERSATION"
          closable={false}
          detachable={true}
          focused={focusedPanel === "conversation"}
          onFocus={() => setFocusedPanel("conversation")}
          className="h-full min-h-0"
        >
          <div className="h-full overflow-auto">
            <ConversationPanel view={view} />
          </div>
        </TiledPanel>

        {/* Knowledge Graph - Right */}
        <TiledPanel
          id="knowledge"
          title="KNOWLEDGE GRAPH"
          closable={false}
          detachable={true}
          focused={focusedPanel === "knowledge"}
          onFocus={() => setFocusedPanel("knowledge")}
          className="h-full min-h-0"
        >
          <div className="h-full overflow-auto">
            <KnowledgePanel view={view} />
          </div>
        </TiledPanel>
      </div>

      {/* Analytics Panel - Fixed Height */}
      <div className="h-[180px] flex-shrink-0">
        <TiledPanel
          id="analytics"
          title="SYSTEM ANALYTICS"
          closable={false}
          detachable={true}
          focused={focusedPanel === "analytics"}
          onFocus={() => setFocusedPanel("analytics")}
          className="h-full"
        >
          <AnalyticsPanel view={view} />
        </TiledPanel>
      </div>
    </div>
  );
}
