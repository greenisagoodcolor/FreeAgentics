"use client";

import React, { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Grid3X3,
  Maximize2,
  Minimize2,
  X,
  Move,
  Settings,
  Square,
  Columns,
  Rows,
  MoreVertical,
} from "lucide-react";

// Layout Configuration Types
interface PanelConfig {
  id: string;
  gridArea: {
    rowStart: number;
    rowEnd: number;
    colStart: number;
    colEnd: number;
  };
  content:
    | "agent-list"
    | "conversation"
    | "knowledge-graph"
    | "analytics"
    | "metrics"
    | "command";
  resizable: boolean;
  closable: boolean;
  detachable: boolean;
  title: string;
}

interface TileLayout {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  grid: {
    rows: number;
    columns: number;
    panels: PanelConfig[];
  };
}

// Predefined Bloomberg-inspired layouts
const defaultLayouts: TileLayout[] = [
  {
    id: "bloomberg",
    name: "Bloomberg Terminal",
    icon: <Grid3X3 className="w-4 h-4" />,
    description: "Professional Bloomberg-style layout",
    grid: {
      rows: 6,
      columns: 12,
      panels: [
        {
          id: "metrics",
          gridArea: { rowStart: 1, rowEnd: 2, colStart: 1, colEnd: 13 },
          content: "metrics",
          resizable: false,
          closable: false,
          detachable: true,
          title: "SYSTEM METRICS",
        },
        {
          id: "knowledge",
          gridArea: { rowStart: 2, rowEnd: 5, colStart: 1, colEnd: 9 },
          content: "knowledge-graph",
          resizable: true,
          closable: false,
          detachable: true,
          title: "KNOWLEDGE GRAPH",
        },
        {
          id: "agents",
          gridArea: { rowStart: 2, rowEnd: 5, colStart: 9, colEnd: 13 },
          content: "agent-list",
          resizable: true,
          closable: false,
          detachable: true,
          title: "AGENT CONTROL",
        },
        {
          id: "conversation",
          gridArea: { rowStart: 5, rowEnd: 6, colStart: 1, colEnd: 9 },
          content: "conversation",
          resizable: true,
          closable: false,
          detachable: true,
          title: "CONVERSATION STREAM",
        },
        {
          id: "analytics",
          gridArea: { rowStart: 5, rowEnd: 6, colStart: 9, colEnd: 13 },
          content: "analytics",
          resizable: true,
          closable: false,
          detachable: true,
          title: "ANALYTICS",
        },
      ],
    },
  },
  {
    id: "quad",
    name: "Quadrant",
    icon: <Square className="w-4 h-4" />,
    description: "Four equal panels",
    grid: {
      rows: 2,
      columns: 2,
      panels: [
        {
          id: "agents",
          gridArea: { rowStart: 1, rowEnd: 2, colStart: 1, colEnd: 2 },
          content: "agent-list",
          resizable: true,
          closable: true,
          detachable: true,
          title: "AGENTS",
        },
        {
          id: "conversation",
          gridArea: { rowStart: 1, rowEnd: 2, colStart: 2, colEnd: 3 },
          content: "conversation",
          resizable: true,
          closable: true,
          detachable: true,
          title: "CONVERSATION",
        },
        {
          id: "knowledge",
          gridArea: { rowStart: 2, rowEnd: 3, colStart: 1, colEnd: 2 },
          content: "knowledge-graph",
          resizable: true,
          closable: true,
          detachable: true,
          title: "KNOWLEDGE",
        },
        {
          id: "analytics",
          gridArea: { rowStart: 2, rowEnd: 3, colStart: 2, colEnd: 3 },
          content: "analytics",
          resizable: true,
          closable: true,
          detachable: true,
          title: "ANALYTICS",
        },
      ],
    },
  },
];

interface TilingWindowManagerProps {
  initialLayout?: string;
  onLayoutChange?: (layoutId: string) => void;
  children?: React.ReactNode;
}

const TilingWindowManager: React.FC<TilingWindowManagerProps> = ({
  initialLayout = "bloomberg",
  onLayoutChange,
  children,
}) => {
  const [currentLayout, setCurrentLayout] = useState<TileLayout>(
    defaultLayouts.find((l) => l.id === initialLayout) || defaultLayouts[0],
  );
  const [focusedPanel, setFocusedPanel] = useState<string | null>(null);
  const [panels, setPanels] = useState<Map<string, PanelConfig>>(
    new Map(currentLayout.grid.panels.map((p) => [p.id, p])),
  );

  // Handle layout switching
  const handleLayoutChange = useCallback(
    (layout: TileLayout) => {
      setCurrentLayout(layout);
      setPanels(new Map(layout.grid.panels.map((p) => [p.id, p])));
      setFocusedPanel(null);
      onLayoutChange?.(layout.id);
    },
    [onLayoutChange],
  );

  // Generate grid template styles
  const getGridTemplateStyle = () => {
    return {
      gridTemplateRows: `repeat(${currentLayout.grid.rows}, 1fr)`,
      gridTemplateColumns: `repeat(${currentLayout.grid.columns}, 1fr)`,
    };
  };

  return (
    <div className="tiling-container">
      {/* Fixed Header Bar */}
      <div className="tiling-header">
        {/* Left: Logo & Layout Switcher */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md flex items-center justify-center bg-[var(--primary-amber)]">
              <span className="font-bold text-sm text-[var(--bg-primary)]">
                CN
              </span>
            </div>
            <h1 className="font-semibold text-lg text-[var(--text-primary)]">
              FreeAgentics Terminal
            </h1>
          </div>

          {/* Layout Switcher */}
          <div className="layout-switcher">
            <div className="quick-layouts">
              {defaultLayouts.map((layout) => (
                <motion.button
                  key={layout.id}
                  className={`layout-btn ${currentLayout.id === layout.id ? "active" : ""}`}
                  title={`${layout.name} - ${layout.description}`}
                  onClick={() => handleLayoutChange(layout)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {layout.icon}
                </motion.button>
              ))}
            </div>
          </div>
        </div>

        {/* Center: Current Layout Info */}
        <div className="flex items-center gap-4">
          <div className="text-center">
            <div className="text-xs font-mono text-[var(--text-secondary)]">
              LAYOUT
            </div>
            <div className="text-sm font-semibold text-[var(--primary-amber)]">
              {currentLayout.name.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Right: System Status */}
        <div className="system-status">
          <div className="flex items-center gap-4">
            <span className="text-[var(--text-secondary)]">LATENCY: 12ms</span>
            <span className="text-[var(--text-secondary)]">CPU: 23%</span>
            <span className="text-[var(--text-secondary)]">MEM: 1.2GB</span>
            <div className="flex items-center gap-1">
              <div className="status-dot active"></div>
              <span className="text-[var(--success)]">ONLINE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tiled Workspace */}
      <div
        className="tiling-workspace"
        data-layout={currentLayout.id}
        style={getGridTemplateStyle()}
      >
        {children}
      </div>
    </div>
  );
};

export default TilingWindowManager;
