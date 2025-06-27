'use client';

import React, { useState } from 'react';
import { useSearchParams } from 'next/navigation';
import BloombergLayout from './layouts/BloombergLayout';
import ResizableLayout from './layouts/ResizableLayout';
import KnowledgeLayout from './layouts/KnowledgeLayout';

export type DashboardView = 'executive' | 'technical' | 'research' | 'minimal';

interface DashboardPageProps {}

export default function DashboardPage({}: DashboardPageProps) {
  const searchParams = useSearchParams();
  const initialView = (searchParams?.get('view') as DashboardView) || 'executive';
  const [currentView, setCurrentView] = useState<DashboardView>(initialView);

  const renderLayout = () => {
    switch (currentView) {
      case 'executive':
        return <BloombergLayout view={currentView} />;
      case 'technical':
        return <ResizableLayout view={currentView} />;
      case 'research':
        return <KnowledgeLayout view={currentView} />;
      case 'minimal':
        return <BloombergLayout view={currentView} />;
      default:
        return <BloombergLayout view={currentView} />;
    }
  };

  return (
    <div className="dashboard-container bg-primary min-h-screen">
      {/* Bloomberg-style Command Bar */}
      <div className="command-bar bg-secondary border-b border-tertiary h-12 flex items-center justify-between px-6">
        {/* Left: Logo & Navigation */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-primary-amber rounded-md flex items-center justify-center">
              <span className="text-bg-primary font-bold text-sm">CN</span>
            </div>
            <h1 className="text-text-primary font-semibold text-lg">CogniticNet</h1>
          </div>
          
          {/* View Selector */}
          <div className="flex items-center gap-2">
            {(['executive', 'technical', 'research', 'minimal'] as DashboardView[]).map((view) => (
              <button
                key={view}
                onClick={() => setCurrentView(view)}
                className={`button button-sm ${
                  currentView === view 
                    ? 'button-primary' 
                    : 'button-ghost'
                } transition-all duration-fast`}
              >
                {view.charAt(0).toUpperCase() + view.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Center: System Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="status-dot active"></div>
            <span className="text-text-secondary text-sm font-mono">SYSTEM ONLINE</span>
          </div>
          <div className="text-text-tertiary text-sm font-mono">
            {new Date().toLocaleTimeString()}
          </div>
        </div>

        {/* Right: Quick Actions */}
        <div className="flex items-center gap-3">
          <button className="button button-sm button-ghost">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
          <button className="button button-sm button-ghost">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
            </svg>
          </button>
          <button className="button button-sm button-primary">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            New Agent
          </button>
        </div>
      </div>

      {/* Main Dashboard Content */}
      <div className="dashboard-content h-[calc(100vh-48px)]">
        {renderLayout()}
      </div>
    </div>
  );
} 