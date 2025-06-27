'use client';

import React from 'react';
import { DashboardView } from '../page';
import AgentPanel from '../components/panels/AgentPanel';
import MetricsPanel from '../components/panels/MetricsPanel';
import ConversationPanel from '../components/panels/ConversationPanel';
import AnalyticsPanel from '../components/panels/AnalyticsPanel';
import KnowledgePanel from '../components/panels/KnowledgePanel';

interface BloombergLayoutProps {
  view: DashboardView;
}

export default function BloombergLayout({ view }: BloombergLayoutProps) {
  return (
    <div className="bloomberg-layout h-full" style={{ background: 'var(--bg-primary)' }}>
      {/* Bloomberg Terminal Grid Layout */}
      <div className="grid grid-cols-12 grid-rows-6 gap-1 h-full p-1">
        
        {/* Top Row: Key Metrics */}
        <div className="col-span-12 row-span-1">
          <div className="card h-full">
            <div className="card-header">
              <h2 className="card-title" style={{ color: 'var(--primary-amber)' }}>SYSTEM METRICS</h2>
            </div>
            <div className="card-content">
              <MetricsPanel view={view} />
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="col-span-8 row-span-4">
          <div className="card h-full">
            <div className="card-header" style={{ borderBottom: '1px solid var(--primary-amber)' }}>
              <h2 className="card-title" style={{ color: 'var(--primary-amber)' }}>KNOWLEDGE GRAPH</h2>
              <div className="flex items-center gap-2">
                <div className="status-dot active"></div>
                <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>LIVE</span>
              </div>
            </div>
            <div className="card-content p-0">
              <KnowledgePanel view={view} />
            </div>
          </div>
        </div>

        {/* Right Sidebar: Agent Management */}
        <div className="col-span-4 row-span-4">
          <div className="card h-full">
            <div className="card-header" style={{ borderBottom: '1px solid var(--primary-amber)' }}>
              <h2 className="card-title" style={{ color: 'var(--primary-amber)' }}>AGENT CONTROL</h2>
              <button className="button button-xs button-primary">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </button>
            </div>
            <div className="card-content p-0">
              <AgentPanel view={view} />
            </div>
          </div>
        </div>

        {/* Bottom Row: Real-time Activity */}
        <div className="col-span-8 row-span-1">
          <div className="card h-full">
            <div className="card-header">
              <h2 className="card-title" style={{ color: 'var(--primary-amber)' }}>CONVERSATION STREAM</h2>
              <div className="flex items-center gap-2">
                <div className="status-dot thinking"></div>
                <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>PROCESSING</span>
              </div>
            </div>
            <div className="card-content p-0">
              <ConversationPanel view={view} />
            </div>
          </div>
        </div>

        {/* Bottom Right: Analytics */}
        <div className="col-span-4 row-span-1">
          <div className="card h-full">
            <div className="card-header">
              <h2 className="card-title" style={{ color: 'var(--primary-amber)' }}>ANALYTICS</h2>
            </div>
            <div className="card-content p-0">
              <AnalyticsPanel view={view} />
            </div>
          </div>
        </div>

      </div>

      {/* Bloomberg-style Footer Status Bar */}
      <div className="status-bar h-6 flex items-center justify-between px-4 text-xs font-mono" style={{
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--primary-amber)'
      }}>
        <div className="flex items-center gap-4">
          <span style={{ color: 'var(--primary-amber)' }}>COGNITICNET TERMINAL</span>
          <span style={{ color: 'var(--text-secondary)' }}>v2.1.0</span>
        </div>
        <div className="flex items-center gap-4">
          <span style={{ color: 'var(--text-secondary)' }}>LATENCY: 12ms</span>
          <span style={{ color: 'var(--text-secondary)' }}>CPU: 23%</span>
          <span style={{ color: 'var(--text-secondary)' }}>MEM: 1.2GB</span>
          <div className="flex items-center gap-1">
            <div className="status-dot active"></div>
            <span style={{ color: 'var(--success)' }}>ONLINE</span>
          </div>
        </div>
      </div>
    </div>
  );
} 