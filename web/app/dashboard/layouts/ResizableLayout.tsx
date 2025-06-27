'use client';

import React, { useState } from 'react';
import { DashboardView } from '../page';

interface PanelConfig {
  id: string;
  component: React.ComponentType<any>;
  title: string;
}

interface ResizableLayoutProps {
  view: DashboardView;
  panels: PanelConfig[];
}

export default function ResizableLayout({ view, panels }: ResizableLayoutProps) {
  const [panelSizes, setPanelSizes] = useState<Record<string, number>>({
    left: 25,
    center: 50,
    right: 25,
  });

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

  // Arrange panels based on view configuration
  const arrangePanels = () => {
    if (view.id === 'minimal') {
      // Two-panel layout for minimal view
      return (
        <div className="h-full flex gap-1">
          <div 
            className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden"
            style={{ width: `${panelSizes.left}%` }}
          >
            {getPanelComponent('conversation')}
          </div>
          <div 
            className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden"
            style={{ width: `${100 - panelSizes.left}%` }}
          >
            {getPanelComponent('agents')}
          </div>
        </div>
      );
    }

    // Three-panel layout for technical view
    return (
      <div className="h-full flex gap-1">
        <div 
          className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden"
          style={{ width: `${panelSizes.left}%` }}
        >
          {getPanelComponent('agents')}
        </div>
        <div 
          className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden"
          style={{ width: `${panelSizes.center}%` }}
        >
          {getPanelComponent('conversation')}
        </div>
        <div className="flex flex-col gap-1" style={{ width: `${panelSizes.right}%` }}>
          <div className="flex-1 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden">
            {getPanelComponent('analytics')}
          </div>
          <div className="flex-1 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded overflow-hidden">
            {getPanelComponent('controls')}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full bg-[var(--bg-primary)] p-1">
      {arrangePanels()}
    </div>
  );
} 