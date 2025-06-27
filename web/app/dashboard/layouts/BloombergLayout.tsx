'use client';

import React from 'react';
import { DashboardView } from '../page';

interface PanelConfig {
  id: string;
  component: React.ComponentType<any>;
  title: string;
}

interface BloombergLayoutProps {
  view: DashboardView;
  panels: PanelConfig[];
}

export default function BloombergLayout({ view, panels }: BloombergLayoutProps) {
  const getPanelComponent = (panelId: string) => {
    const panel = panels.find(p => p.id === panelId);
    if (!panel) return null;
    
    const Component = panel.component;
    return (
      <div key={panelId} className="h-full">
        <Component view={view} />
      </div>
    );
  };

  return (
    <div className="h-full bg-[var(--bg-primary)] bloomberg-layout">
      {/* Bloomberg Terminal Grid Layout */}
      <div className="h-full grid grid-cols-12 grid-rows-12 gap-1 p-1">
        {/* Top Row - Key Metrics */}
        <div className="col-span-12 row-span-2 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded">
          {getPanelComponent('metrics')}
        </div>
        
        {/* Left Panel - Agent Management */}
        <div className="col-span-4 row-span-8 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded">
          {getPanelComponent('agents')}
        </div>
        
        {/* Center Panel - Knowledge Graph */}
        <div className="col-span-5 row-span-8 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded">
          {getPanelComponent('knowledge')}
        </div>
        
        {/* Right Panel - Controls */}
        <div className="col-span-3 row-span-8 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded">
          {getPanelComponent('controls')}
        </div>
        
        {/* Bottom Row - Status/Info */}
        <div className="col-span-12 row-span-2 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded">
          <div className="h-full p-4 flex items-center justify-between text-sm font-mono">
            <div className="flex items-center gap-6">
              <span className="text-[var(--text-secondary)]">SYSTEM STATUS: OPERATIONAL</span>
              <span className="text-[var(--text-secondary)]">LATENCY: &lt;50ms</span>
              <span className="text-[var(--text-secondary)]">UPTIME: 99.9%</span>
            </div>
            <div className="flex items-center gap-6">
              <span className="text-[var(--text-secondary)]">CPU: 12%</span>
              <span className="text-[var(--text-secondary)]">MEM: 2.1GB</span>
              <span className="text-[var(--text-secondary)]">NET: 156KB/s</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 