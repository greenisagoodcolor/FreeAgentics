"use client";

import React from "react";
import { DashboardView } from "../../../page";
import { Network } from "lucide-react";
import KnowledgeGraphVisualization from "@/components/dashboard/KnowledgeGraphVisualization";

interface KnowledgePanelProps {
  view: DashboardView;
}

export default function KnowledgePanel({ view }: KnowledgePanelProps) {
  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Panel Header */}
      <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-[var(--accent-primary)]" />
          <h3 className="font-semibold text-[var(--text-primary)]">
            Knowledge Graph
          </h3>
        </div>
      </div>

      {/* Knowledge Graph Content - REAL D3.js Implementation */}
      <div className="flex-1 overflow-hidden">
        <KnowledgeGraphVisualization />
      </div>
    </div>
  );
}
