"use client";

import React from "react";
import { DashboardView } from "../../../page";
import { Settings } from "lucide-react";

interface ControlPanelProps {
  view: DashboardView;
}

export default function ControlPanel({ view }: ControlPanelProps) {
  return (
    <div className="h-full bg-[var(--bg-primary)] p-4 flex items-center justify-center">
      <div className="text-center">
        <Settings className="w-12 h-12 text-[var(--text-tertiary)] mx-auto mb-3" />
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">
          Control Panel
        </h3>
        <p className="text-[var(--text-secondary)] text-sm">
          System controls and configuration
        </p>
      </div>
    </div>
  );
}
