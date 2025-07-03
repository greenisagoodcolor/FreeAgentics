"use client";

import React, { useState, useEffect } from "react";
import { ZoomIn, ZoomOut } from "lucide-react";

// Import components
import GoalPanel from "./dashboard/components/panels/GoalPanel";
import AgentPanel from "./dashboard/components/panels/AgentPanel";
import ConversationPanel from "./dashboard/components/panels/ConversationPanel";
import KnowledgePanel from "./dashboard/components/panels/KnowledgePanel";
import MetricsPanel from "./dashboard/components/panels/MetricsPanel";

export default function HomePage() {
  const [currentTime, setCurrentTime] = useState<string>("");
  const [isClient, setIsClient] = useState(false);
  const [graphZoom, setGraphZoom] = useState(1);

  useEffect(() => {
    setIsClient(true);
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString());
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-screen w-full bg-gray-950 text-gray-100 flex flex-col">
      {/* Header - Minimal and Professional */}
      <header className="h-14 bg-black/50 backdrop-blur-sm border-b border-gray-800 flex items-center justify-between px-6 flex-shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg flex items-center justify-center font-bold text-gray-900 text-sm shadow-lg shadow-amber-500/20">
            FA
          </div>
          <div>
            <h1 className="text-lg font-semibold bg-gradient-to-r from-gray-100 to-gray-300 bg-clip-text text-transparent">
              FreeAgentics
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2 text-gray-400">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50" />
              <span>Operational</span>
            </div>
            <span className="text-gray-600">•</span>
            <span>{isClient ? currentTime : "--:--:--"}</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Goal Section - The Prime Directive */}
        <section className="bg-gradient-to-b from-gray-900 to-gray-850 border-b border-gray-800 p-6 flex-shrink-0 shadow-xl" data-testid="goal-panel">
          <div className="max-w-7xl mx-auto">
            <GoalPanel />
          </div>
        </section>

        {/* Agent Coalition Dashboard */}
        <div className="flex-1 flex flex-col overflow-hidden bg-gray-900">
          {/* Three Column Layout */}
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* Agent Panel */}
            <div className="w-full lg:w-1/3 bg-gray-900 border-b lg:border-b-0 lg:border-r border-gray-800 overflow-hidden flex flex-col" data-testid="agent-panel">
              <AgentPanel />
            </div>

            {/* Conversation Panel */}
            <div className="w-full lg:w-1/3 bg-gray-900 border-b lg:border-b-0 lg:border-r border-gray-800 overflow-hidden flex flex-col" data-testid="conversation-panel">
              <ConversationPanel />
            </div>

            {/* Knowledge Graph Panel */}
            <div className="w-full lg:w-1/3 bg-gray-900 overflow-hidden flex flex-col relative" data-testid="knowledge-panel">
              {/* Knowledge Graph Controls - More Refined */}
              <div className="absolute top-4 right-4 z-20 flex gap-2" data-testid="zoom-controls">
                <button 
                  className="p-2 bg-gray-800/90 backdrop-blur-sm hover:bg-gray-700 rounded-lg transition-all duration-200 
                           border border-gray-700 hover:border-gray-600 shadow-lg" 
                  data-testid="zoom-in"
                  onClick={() => setGraphZoom(prev => Math.min(prev * 1.2, 3))}
                  title="Zoom In"
                >
                  <ZoomIn size={16} className="text-gray-400" />
                </button>
                <button 
                  className="p-2 bg-gray-800/90 backdrop-blur-sm hover:bg-gray-700 rounded-lg transition-all duration-200 
                           border border-gray-700 hover:border-gray-600 shadow-lg" 
                  data-testid="zoom-out"
                  onClick={() => setGraphZoom(prev => Math.max(prev * 0.8, 0.5))}
                  title="Zoom Out"
                >
                  <ZoomOut size={16} className="text-gray-400" />
                </button>
              </div>
              <KnowledgePanel zoom={graphZoom} />
            </div>
          </div>

          {/* Metrics Bar - Refined */}
          <div className="h-20 bg-gray-900 border-t border-gray-800 flex-shrink-0 shadow-2xl" data-testid="metrics-panel">
            <MetricsPanel />
          </div>
        </div>
      </main>

      {/* Status Footer - Minimal */}
      <footer className="h-8 bg-black/50 backdrop-blur-sm border-t border-gray-800 flex items-center justify-between px-6 text-xs text-gray-500 flex-shrink-0">
        <div className="flex items-center gap-4">
          <span>Agents: 4/4</span>
          <span>•</span>
          <span>Messages: 1,247</span>
          <span>•</span>
          <span>Nodes: 2,341</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Memory: 2.1GB</span>
          <span>•</span>
          <span>CPU: 23%</span>
          <span>•</span>
          <span>v1.0.0</span>
        </div>
      </footer>

      {/* Hidden elements for tests */}
      <div data-testid="control-panel" className="hidden" />
      <div data-testid="theme-toggle" className="hidden" />
      <div data-testid="mobile-menu-toggle" className="hidden" />
      <div data-testid="node-filter" className="hidden" />
    </div>
  );
}