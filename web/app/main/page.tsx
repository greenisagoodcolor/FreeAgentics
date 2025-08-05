"use client";

import React, { useState, useEffect } from "react";
import { PromptBar } from "@/components/main/PromptBar";
import { AgentCreatorPanel } from "@/components/main/AgentCreatorPanel";
import { ConversationWindow } from "@/components/main/ConversationWindow";
import { KnowledgeGraphView } from "@/components/main/KnowledgeGraphView";
import { SimulationGrid } from "@/components/main/SimulationGrid";
import { MetricsFooter } from "@/components/main/MetricsFooter";
import { SettingsModal } from "@/components/modals/SettingsModal";
import { useOnboarding } from "@/hooks/use-onboarding";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, Key } from "lucide-react";

export default function MainPage() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const { needsOnboarding, markOnboardingComplete, hasValidApiKey } = useOnboarding();

  // Show onboarding modal automatically when needed
  useEffect(() => {
    if (needsOnboarding) {
      console.log("[Onboarding] Showing first-run API key setup");
      setIsSettingsOpen(true);
    }
  }, [needsOnboarding]);

  // Close modal and mark onboarding complete when settings are closed
  const handleSettingsClose = (open: boolean) => {
    setIsSettingsOpen(open);
    if (!open && needsOnboarding) {
      // Mark onboarding complete when user closes settings
      // (whether they configured keys or not)
      markOnboardingComplete();
    }
  };

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

      {/* API Key Warning Banner */}
      {!hasValidApiKey && (
        <div className="fixed top-0 left-0 right-0 z-50 p-4">
          <Alert variant="default" className="bg-yellow-50 border-yellow-200">
            <Key className="h-4 w-4 text-yellow-600" />
            <AlertDescription className="text-yellow-800">
              <strong>API Key Required:</strong> Add your OpenAI or Anthropic API key in Settings to enable agent conversations.
            </AlertDescription>
          </Alert>
        </div>
      )}

      {/* Onboarding Settings Modal */}
      <SettingsModal open={isSettingsOpen} onOpenChange={handleSettingsClose} />
    </div>
  );
}
