'use client';

import React from 'react';
import { DashboardView } from '../../../page';
import { BarChart3 } from 'lucide-react';
import AnalyticsWidgetSystem from '@/components/dashboard/AnalyticsWidgetSystem';

interface AnalyticsPanelProps {
  view: DashboardView;
}

export default function AnalyticsPanel({ view }: AnalyticsPanelProps) {
  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Panel Header */}
      <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-[var(--accent-primary)]" />
          <h3 className="font-semibold text-[var(--text-primary)]">
            Analytics Dashboard
          </h3>
        </div>
      </div>

      {/* Analytics Widget System - REAL Drag-and-Drop Implementation */}
      <div className="flex-1 overflow-hidden">
        <AnalyticsWidgetSystem />
      </div>
    </div>
  );
} 