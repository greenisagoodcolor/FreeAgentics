"use client";

import React from "react";
import { PromptBar } from "@/components/main/PromptBar";
import { AgentCreatorPanel } from "@/components/main/AgentCreatorPanel";
import { ConversationWindow } from "@/components/main/ConversationWindow";
import { KnowledgeGraphView } from "@/components/main/KnowledgeGraphView";
import { SimulationGrid } from "@/components/main/SimulationGrid";
import { MetricsFooter } from "@/components/main/MetricsFooter";

export default function MainPage() {
  return (
    <div className="main-layout flex flex-col h-screen bg-background">
      {/* Top Bar - height ≤ 64px, sticky */}
      <div className="top-bar sticky top-0 z-50 max-h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <PromptBar />
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Row - 3 columns: Agent Creator, Knowledge Graph, Simulation Grid */}
        <div className="main-row grid grid-cols-1 lg:grid-cols-3 gap-4 p-4 flex-1 overflow-hidden">
          {/* Agent Creator Panel */}
          <div className="h-full overflow-hidden">
            <AgentCreatorPanel />
          </div>

          {/* Knowledge Graph View */}
          <div className="h-full overflow-hidden">
            <KnowledgeGraphView />
          </div>

          {/* Simulation Grid */}
          <div className="h-full overflow-hidden">
            <SimulationGrid />
          </div>
        </div>

        {/* Bottom Row - Conversation Window (bottom third) */}
        <div className="conversation-row w-full p-4 h-1/3">
          <ConversationWindow />
        </div>
      </div>

      {/* Footer Bar - height ≈ 48px */}
      <div className="footer-bar h-12 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <MetricsFooter />
      </div>
    </div>
  );
}
