"use client";

import React from "react";
import { Network } from "lucide-react";
import KnowledgeGraphVisualization from "@/components/dashboard/KnowledgeGraphVisualization";

interface KnowledgePanelProps {
  zoom?: number;
}

export default function KnowledgePanel({ zoom = 1 }: KnowledgePanelProps) {
  // Enable test mode for Playwright tests or when NODE_ENV is test
  const isTestMode =
    typeof window !== "undefined" &&
    (window.location.search.includes("testMode=true") ||
      process.env.NODE_ENV === "test");

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Panel Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-gray-100">
            Knowledge Graph
          </h3>
        </div>
        <div className="text-xs text-gray-500">
          {zoom !== 1 && `${Math.round(zoom * 100)}%`}
        </div>
      </div>

      {/* Knowledge Graph Content */}
      <div className="flex-1 overflow-hidden bg-gray-950/50">
        <KnowledgeGraphVisualization testMode={isTestMode} zoom={zoom} />
      </div>
    </div>
  );
}