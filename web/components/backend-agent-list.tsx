"use client";

import AgentList from "@/components/AgentList";
import type { Agent } from "@/lib/types";
import type { LLMSettings } from "@/lib/llm-settings";

interface BackendAgentListProps {
  filter?: { class?: string };
  onSelectAgent: (agent: Agent | null) => void;
  selectedAgent: Agent | null;
}

export function BackendAgentList({
  filter,
  onSelectAgent,
  selectedAgent,
}: BackendAgentListProps) {
  // This is a wrapper component that will use the existing AgentList
  return (
    <AgentList
      agents={[]}
      selectedAgent={selectedAgent}
      onSelectAgent={onSelectAgent}
      onCreateAgent={() => {}}
      onCreateAgentWithName={() => {}}
      onDeleteAgent={() => {}}
      onAddToConversation={() => {}}
      onRemoveFromConversation={() => {}}
      onUpdateAgentColor={() => {}}
      onToggleAutonomy={() => {}}
      onExportAgents={() => {}}
      onImportAgents={() => {}}
      activeConversation={false}
      llmSettings={
        {
          provider: "openai",
          model: "gpt-4",
          temperature: 0.7,
          maxTokens: 2000,
          topP: 1,
          frequencyPenalty: 0,
          presencePenalty: 0,
        } as LLMSettings
      }
    />
  );
}
