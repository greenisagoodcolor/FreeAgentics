"use client";

import type React from "react";

import { useRef, useEffect, useState, useCallback } from "react";
import type { Agent, Conversation, Position } from "@/lib/types";
import { CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ZoomIn,
  ZoomOut,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import GlobalKnowledgeGraph from "@/components/GlobalKnowledgeGraph";

interface GridWorldProps {
  agents: Agent[];
  onUpdatePosition: (agentId: string, position: Position) => void;
  activeConversation: Conversation | null;
  onSelectKnowledgeNode: (
    nodeType: "entry" | "tag",
    nodeId: string,
    nodeTitle: string,
  ) => void;
  onShowAbout: () => void;
}

export default function GridWorld({
  agents,
  onUpdatePosition,
  activeConversation,
  onSelectKnowledgeNode,
  onShowAbout,
}: GridWorldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [gridSize, setGridSize] = useState({ width: 10, height: 10 });
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [cellSize, setCellSize] = useState(40);

  // Navigation state
  const [zoomLevel, setZoomLevel] = useState(1);
  const [viewportOffset, setViewportOffset] = useState({ x: 0, y: 0 });

  // Add state variables for resizing
  const [gridWorldHeight, setGridWorldHeight] = useState<number>(50); // Percentage (0-100)
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const [resizeStartY, setResizeStartY] = useState<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Resize event handlers
  const handleResizeStart = (e: React.MouseEvent) => {
    setIsResizing(true);
    setResizeStartY(e.clientY);
  };

  const handleResizeMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const containerHeight = containerRect.height;
      const mouseY = e.clientY - containerRect.top;

      // Calculate new height percentage with constraints
      const newHeightPercentage = Math.min(
        Math.max(
          (mouseY / containerHeight) * 100,
          20, // Minimum 20%
        ),
        80, // Maximum 80%
      );

      setGridWorldHeight(newHeightPercentage);
    },
    [isResizing],
  );

  const handleResizeEnd = () => {
    setIsResizing(false);
  };

  // Add event listeners for mouse movement and release
  useEffect(() => {
    if (isResizing) {
      window.addEventListener("mousemove", handleResizeMove);
      window.addEventListener("mouseup", handleResizeEnd);

      // Add a class to the body to change cursor during resize
      document.body.classList.add("resize-ns");
    } else {
      window.removeEventListener("mousemove", handleResizeMove);
      window.removeEventListener("mouseup", handleResizeEnd);

      // Remove the cursor class
      document.body.classList.remove("resize-ns");
    }

    return () => {
      window.removeEventListener("mousemove", handleResizeMove);
      window.removeEventListener("mouseup", handleResizeEnd);
      document.body.classList.remove("resize-ns");
    };
  }, [isResizing, handleResizeMove]);

  // Draw the grid and agents
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply zoom and pan transformations
    ctx.save();
    ctx.translate(-viewportOffset.x, -viewportOffset.y);
    ctx.scale(zoomLevel, zoomLevel);

    // Calculate effective cell size with zoom
    const effectiveCellSize = cellSize * zoomLevel;

    // Draw grid
    ctx.strokeStyle = "#6b21a8"; // Purple color for grid lines
    ctx.lineWidth = 1 / zoomLevel; // Adjust line width for zoom

    // Draw vertical lines
    for (let x = 0; x <= gridSize.width; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cellSize, 0);
      ctx.lineTo(x * cellSize, gridSize.height * cellSize);
      ctx.stroke();
    }

    // Draw horizontal lines
    for (let y = 0; y <= gridSize.height; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellSize);
      ctx.lineTo(gridSize.width * cellSize, y * cellSize);
      ctx.stroke();
    }

    // Draw agents
    agents.forEach((agent) => {
      const { x, y } = agent.position;

      // Calculate radius with absolute value to ensure it's positive
      const radius = Math.max(cellSize / 3, 1); // Ensure minimum radius of 1

      // Draw agent circle
      ctx.beginPath();
      ctx.arc(
        x * cellSize + cellSize / 2,
        y * cellSize + cellSize / 2,
        radius,
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = agent.color;
      ctx.fill();

      // Draw selection indicator if selected
      if (agent.id === selectedAgentId) {
        const selectionRadius = Math.max(cellSize / 2, 2); // Ensure minimum radius of 2
        ctx.beginPath();
        ctx.arc(
          x * cellSize + cellSize / 2,
          y * cellSize + cellSize / 2,
          selectionRadius,
          0,
          Math.PI * 2,
        );
        ctx.strokeStyle = "#ffffff"; // White selection indicator
        ctx.lineWidth = 2 / zoomLevel;
        ctx.stroke();
      }

      // Draw conversation indicator
      if (agent.inConversation) {
        const indicatorRadius = Math.max(cellSize / 2, 2); // Ensure minimum radius of 2
        ctx.beginPath();
        ctx.arc(
          x * cellSize + cellSize / 2,
          y * cellSize + cellSize / 2,
          indicatorRadius,
          0,
          Math.PI * 2,
        );
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2 / zoomLevel;
        ctx.stroke();
      }

      // Draw agent name
      ctx.fillStyle = "#ffffff"; // White text
      ctx.font = `${Math.max(10 / zoomLevel, 8)}px Arial`; // Ensure minimum font size
      ctx.textAlign = "center";
      ctx.fillText(
        agent.name,
        x * cellSize + cellSize / 2,
        y * cellSize + cellSize + 5,
      );
    });

    // Draw conversation lines between agents in conversation
    if (activeConversation) {
      const conversationAgents = agents.filter((agent) =>
        activeConversation.participants.includes(agent.id),
      );

      if (conversationAgents.length >= 2) {
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2 / zoomLevel;

        for (let i = 0; i < conversationAgents.length; i++) {
          for (let j = i + 1; j < conversationAgents.length; j++) {
            const agent1 = conversationAgents[i];
            const agent2 = conversationAgents[j];

            ctx.beginPath();
            ctx.moveTo(
              agent1.position.x * cellSize + cellSize / 2,
              agent1.position.y * cellSize + cellSize / 2,
            );
            ctx.lineTo(
              agent2.position.x * cellSize + cellSize / 2,
              agent2.position.y * cellSize + cellSize / 2,
            );
            ctx.stroke();
          }
        }
      }
    }

    // Restore canvas context
    ctx.restore();
  }, [
    agents,
    gridSize,
    cellSize,
    selectedAgentId,
    activeConversation,
    zoomLevel,
    viewportOffset,
  ]);

  // Handle canvas click for agent selection and movement
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();

    // Adjust for zoom and pan
    const adjustedX = (e.clientX - rect.left + viewportOffset.x) / zoomLevel;
    const adjustedY = (e.clientY - rect.top + viewportOffset.y) / zoomLevel;

    const x = Math.floor(adjustedX / cellSize);
    const y = Math.floor(adjustedY / cellSize);

    // Check if clicked on an agent
    const clickedAgent = agents.find(
      (agent) => agent.position.x === x && agent.position.y === y,
    );

    if (clickedAgent) {
      setSelectedAgentId(clickedAgent.id);
    } else if (selectedAgentId) {
      // Move selected agent to new position
      onUpdatePosition(selectedAgentId, { x, y });
      setSelectedAgentId(null);
    }
  };

  // Resize canvas when component mounts or when gridWorldHeight changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const updateCanvasSize = () => {
      const container = canvas.parentElement;
      if (!container) return;

      const width = container.clientWidth;
      const height = container.clientHeight - 60; // Subtract header height

      canvas.width = width;
      canvas.height = height;

      // Calculate cell size based on container size and grid dimensions
      const cellWidth = width / gridSize.width;
      const cellHeight = height / gridSize.height;
      setCellSize(Math.min(cellWidth, cellHeight));
    };

    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    return () => {
      window.removeEventListener("resize", updateCanvasSize);
    };
  }, [gridSize, gridWorldHeight]); // Add gridWorldHeight to dependencies

  // Navigation handlers
  const handleZoomIn = () => {
    setZoomLevel((prev) => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel((prev) => Math.max(prev - 0.2, 0.5));
  };

  const handlePan = (direction: "up" | "down" | "left" | "right") => {
    const panAmount = 50;

    setViewportOffset((prev) => {
      switch (direction) {
        case "up":
          return { ...prev, y: Math.max(prev.y - panAmount, 0) };
        case "down":
          return { ...prev, y: prev.y + panAmount };
        case "left":
          return { ...prev, x: Math.max(prev.x - panAmount, 0) };
        case "right":
          return { ...prev, x: prev.x + panAmount };
        default:
          return prev;
      }
    });
  };

  const handleReset = () => {
    setZoomLevel(1);
    setViewportOffset({ x: 0, y: 0 });
  };

  return (
    <div className="flex flex-col h-full">
      <CardHeader className="pb-2 border-b border-purple-800 bg-gradient-to-r from-purple-900/50 to-indigo-900/50">
        <CardTitle className="text-white">Grid World</CardTitle>
      </CardHeader>

      {/* Split the content area into two sections */}
      <div className="flex-1 flex flex-col" ref={containerRef}>
        {/* Top half: Grid World */}
        <div className="bg-black/20" style={{ height: `${gridWorldHeight}%` }}>
          <div className="h-full flex flex-col">
            <div className="p-2 text-sm text-purple-300">
              Click an agent to select, then click an empty cell to move.
            </div>
            <div className="flex-1 relative">
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                className="absolute inset-0"
              />

              {/* Navigation Controls */}
              <div className="absolute bottom-4 right-4 bg-purple-950/80 backdrop-blur-sm p-2 rounded-lg border border-purple-700 shadow-md">
                <div className="grid grid-cols-3 gap-1">
                  {/* Top row */}
                  <div></div>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handlePan("up")}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <ArrowUp size={16} className="text-white" />
                  </Button>
                  <div></div>

                  {/* Middle row */}
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handlePan("left")}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <ArrowLeft size={16} className="text-white" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleReset}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <RefreshCw size={16} className="text-white" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handlePan("right")}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <ArrowRight size={16} className="text-white" />
                  </Button>

                  {/* Bottom row */}
                  <div></div>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handlePan("down")}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <ArrowDown size={16} className="text-white" />
                  </Button>
                  <div></div>
                </div>

                {/* Zoom controls */}
                <div className="flex justify-center mt-2 gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleZoomOut}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <ZoomOut size={16} className="text-white" />
                  </Button>
                  <span className="flex items-center text-xs text-white">
                    {Math.round(zoomLevel * 100)}%
                  </span>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleZoomIn}
                    className="h-8 w-8 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <ZoomIn size={16} className="text-white" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Resizable divider */}
        <div
          className="h-1 bg-purple-800 cursor-row-resize hover:bg-purple-600 relative"
          onMouseDown={handleResizeStart}
        >
          {/* Visual indicator for draggable area */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-1 bg-purple-400 rounded-full"></div>
        </div>

        {/* Bottom half: Global Knowledge Graph */}
        <div
          className="border-t border-purple-800"
          style={{ height: `${100 - gridWorldHeight}%` }}
        >
          <GlobalKnowledgeGraph
            agents={agents}
            onSelectNode={onSelectKnowledgeNode}
            onShowAbout={onShowAbout}
          />
        </div>
      </div>
    </div>
  );
}
