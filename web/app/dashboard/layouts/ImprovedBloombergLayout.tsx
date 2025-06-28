"use client";

import React, { useState } from "react";
import TiledPanel from "@/components/dashboard/TiledPanel";
import GoalPanel from "../components/panels/GoalPanel";
import ConversationPanel from "../components/panels/ConversationPanel";
import KnowledgePanel from "../components/panels/KnowledgePanel";
import AnalyticsPanel from "../components/panels/AnalyticsPanel";
import AgentPanel from "../components/panels/AgentPanel";
import { DashboardView } from "../../page";

interface ImprovedBloombergLayoutProps {
  view: DashboardView;
}

export default function ImprovedBloombergLayout({ view }: ImprovedBloombergLayoutProps) {
  const [focusedPanel, setFocusedPanel] = useState<string | null>(null);

  return (
    <div className="h-full grid grid-rows-[120px_1fr_200px] gap-2 p-2 bg-[var(--bg-primary)]">
      {/* Top Row - Goal Input Panel */}
      <TiledPanel
        id="goal"
        title="AGENT GOAL"
        gridArea={{ rowStart: 1, rowEnd: 2, colStart: 1, colEnd: 13 }}
        closable={false}
        detachable={false}
        focused={focusedPanel === 'goal'}
        onFocus={() => setFocusedPanel('goal')}
        className="h-full"
      >
        <GoalPanel view={view} />
      </TiledPanel>

      {/* Middle Row - Main Content */}
      <div className="grid grid-cols-12 gap-2 h-full">
        {/* Left - Conversation Stream (50% width) */}
        <TiledPanel
          id="conversation"
          title="AGENT CONVERSATION"
          gridArea={{ rowStart: 1, rowEnd: 2, colStart: 1, colEnd: 7 }}
          closable={false}
          detachable={true}
          focused={focusedPanel === 'conversation'}
          onFocus={() => setFocusedPanel('conversation')}
          className="col-span-6 h-full overflow-hidden"
        >
          <ConversationPanel view={view} />
        </TiledPanel>

        {/* Right - Knowledge Graph (50% width) */}
        <TiledPanel
          id="knowledge"
          title="KNOWLEDGE GRAPH"
          gridArea={{ rowStart: 1, rowEnd: 2, colStart: 7, colEnd: 13 }}
          closable={false}
          detachable={true}
          focused={focusedPanel === 'knowledge'}
          onFocus={() => setFocusedPanel('knowledge')}
          className="col-span-6 h-full overflow-hidden"
        >
          <KnowledgePanel view={view} />
        </TiledPanel>
      </div>

      {/* Bottom Row - Analytics */}
      <TiledPanel
        id="analytics"
        title="SYSTEM ANALYTICS"
        gridArea={{ rowStart: 3, rowEnd: 4, colStart: 1, colEnd: 13 }}
        closable={false}
        detachable={true}
        focused={focusedPanel === 'analytics'}
        onFocus={() => setFocusedPanel('analytics')}
        className="h-full"
      >
        <AnalyticsPanel view={view} />
      </TiledPanel>
    </div>
  );
}