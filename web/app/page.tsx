"use client";

import React from "react";
import { PromptBar } from "@/components/main/PromptBar";
import { AgentCreatorPanel } from "@/components/main/AgentCreatorPanel";
import { ConversationWindow } from "@/components/main/ConversationWindow";
import { KnowledgeGraphView } from "@/components/main/KnowledgeGraphView";
import { SimulationGrid } from "@/components/main/SimulationGrid";
import { MetricsFooter } from "@/components/main/MetricsFooter";

export default function HomePage() {
  return (
    <div className="main-layout flex flex-col h-screen bg-background">
      {/* Top Bar - height ≤ 64px, sticky */}
      <div className="top-bar sticky top-0 z-50 max-h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex items-center justify-between px-4 h-full">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-foreground">FreeAgentics</h1>
            <span className="text-sm text-muted-foreground">Active Inference Platform</span>
          </div>
          <div className="flex-1 max-w-2xl mx-4">
            <PromptBar />
          </div>
          <div className="w-32"></div> {/* Spacer for balance */}
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Main Row - 3 equal columns on large screens */}
        <div className="main-row grid grid-cols-1 lg:grid-cols-3 gap-4 p-4 flex-1 overflow-hidden">
          {/* Agent Creator Panel */}
          <div className="h-full overflow-hidden">
            <AgentCreatorPanel />
          </div>

          {/* Conversation Window */}
          <div className="h-full overflow-hidden">
            <ConversationWindow />
          </div>

          {/* Knowledge Graph View */}
          <div className="h-full overflow-hidden">
            <KnowledgeGraphView />
          </div>
        </div>

        {/* Second Row - Simulation Grid (100% width) */}
        <div className="simulation-row w-full p-4">
          <SimulationGrid />
        </div>
      </div>

      {/* Footer Bar - height ≈ 48px */}
      <div className="footer-bar h-12 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <MetricsFooter />
      </div>
    </div>
  );
}
