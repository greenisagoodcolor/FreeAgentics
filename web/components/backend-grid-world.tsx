"use client";

import GridWorld from "@/components/gridworld";

interface BackendGridWorldProps {
  onHexClick?: (hexId: string | null) => void;
  selectedHex?: string | null;
  viewMode?: "terrain" | "resources" | "agents";
}

export function BackendGridWorld({ onHexClick, selectedHex, viewMode }: BackendGridWorldProps) {
  // This is a wrapper component that uses the existing GridWorld
  // Props are passed for future implementation
  return <GridWorld agents={[]} onUpdatePosition={() => {}} activeConversation={null} onSelectKnowledgeNode={() => {}} onShowAbout={() => {}} />;
}
