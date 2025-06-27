"use client";

import KnowledgeGraph from "@/components/KnowledgeGraph";

interface KnowledgeGraphVizProps {
  agentId?: string;
  mode?: "individual" | "collective";
  filter?: string;
}

export function KnowledgeGraphViz({
  agentId,
  mode,
  filter,
}: KnowledgeGraphVizProps) {
  // This is a wrapper component that uses the existing KnowledgeGraph
  // Props are passed for future implementation
  return (
    <KnowledgeGraph
      knowledge={[]}
      onSelectEntry={() => {}}
      selectedEntry={null}
    />
  );
}
