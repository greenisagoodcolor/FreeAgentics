'use client';

import React from 'react';
import { DashboardView } from '../page';

interface PanelConfig {
  id: string;
  component: React.ComponentType<any>;
  title: string;
}

interface KnowledgeLayoutProps {
  view: DashboardView;
  panels: PanelConfig[];
}

export default function KnowledgeLayout({ view, panels }: KnowledgeLayoutProps) {
  const getPanelComponent = (panelId: string) => {
    const panel = panels.find(p => p.id === panelId);
    if (!panel) return null;
    
    const Component = panel.component;
    return (
      <div className="h-full flex flex-col">
        <div className="bg-[var(--bg-tertiary)] px-4 py-2 border-b border-[var(--bg-quaternary)]">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">
            {panel.title}
          </h3>
        </div>
        <div className="flex-1 overflow-hidden">
          <Component view={view} />
        </div>
      </div>
    );
  };

  return (
    <div className="h-full bg-[var(--bg-primary)] knowledge-layout p-1">
      {/* Knowledge-Centric Grid Layout */}
      <div className="h-full grid grid-cols-12 grid-rows-12 gap-1">
        {/* Main Knowledge Graph - Takes center stage */}
        <div className="col-span-8 row-span-10 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden">
          {getPanelComponent('knowledge')}
        </div>
        
        {/* Right Side - Analytics */}
        <div className="col-span-4 row-span-6 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden">
          {getPanelComponent('analytics')}
        </div>
        
        {/* Right Side - Agents */}
        <div className="col-span-4 row-span-4 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden">
          {getPanelComponent('agents')}
        </div>
        
        {/* Bottom Row - Knowledge Insights/Controls */}
        <div className="col-span-12 row-span-2 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded">
          <div className="h-full p-4 flex items-center justify-between text-sm">
            <div className="flex items-center gap-6">
              <span className="text-[var(--text-secondary)]">
                KNOWLEDGE NODES: {/* Will be populated by knowledge panel */}
              </span>
              <span className="text-[var(--text-secondary)]">
                CONNECTIONS: {/* Will be populated by knowledge panel */}
              </span>
              <span className="text-[var(--text-secondary)]">
                CONFIDENCE: {/* Will be populated by knowledge panel */}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <button className="px-3 py-1 bg-[var(--accent-primary)] text-white rounded text-xs">
                Export Graph
              </button>
              <button className="px-3 py-1 bg-[var(--bg-tertiary)] text-[var(--text-primary)] rounded text-xs">
                Filter Nodes
              </button>
              <button className="px-3 py-1 bg-[var(--bg-tertiary)] text-[var(--text-primary)] rounded text-xs">
                Analyze Patterns
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 