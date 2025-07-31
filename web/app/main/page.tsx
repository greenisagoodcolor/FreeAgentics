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
      {/* Header Section with Title and Prompt */}
      <div className="header-section border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        {/* Title and Subtitle */}
        <div className="px-6 pt-6 pb-4">
          <h1 className="text-3xl font-bold text-foreground mb-2">FreeAgentics</h1>
          <p className="text-lg text-muted-foreground mb-4">
            Multi-agent AI platform implementing Active Inference for autonomous,
            mathematically-principled intelligent systems
          </p>
        </div>

        {/* Prompt Input - Now bigger and more prominent */}
        <div className="px-6 pb-4">
          <PromptBar />
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Row - 3 columns with explanations */}
        <div className="main-row grid gap-4 p-4 flex-1 overflow-hidden auto-rows-max grid-cols-1 lg:grid-cols-12">
          {/* Agent Creator Panel - lg: span 3 */}
          <div className="h-full overflow-hidden lg:col-span-3">
            <div className="mb-2">
              <p className="text-sm text-muted-foreground">
                Create Active Inference agents that minimize free energy through belief updates and
                action selection
              </p>
            </div>
            <AgentCreatorPanel />
          </div>

          {/* Knowledge Graph View - lg: span 5 */}
          <div className="h-full overflow-hidden lg:col-span-5">
            <div className="mb-2">
              <p className="text-sm text-muted-foreground">
                Semantic knowledge representation showing agent beliefs and world model
                relationships
              </p>
            </div>
            <KnowledgeGraphView />
          </div>

          {/* Simulation Grid - lg: span 4 */}
          <div className="h-full overflow-hidden lg:col-span-4">
            <div className="mb-2">
              <p className="text-sm text-muted-foreground">
                Grid world environment where agents demonstrate emergent behavior through
                variational inference
              </p>
            </div>
            <SimulationGrid />
          </div>
        </div>

        {/* Bottom Row - Conversation Window (bottom third) */}
        <div className="conversation-row w-full p-4 h-1/3">
          <div className="mb-2">
            <p className="text-sm text-muted-foreground">
              Real-time conversation with agents showing their reasoning process and decision-making
              via PyMDP
            </p>
          </div>
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
