"use client";

import React, { useRef, useEffect, useState } from "react";
import { Play, Pause, RotateCcw, SkipForward, Settings, Zap, Activity } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useSimulation } from "@/hooks/use-simulation";
import { useAgents } from "@/hooks/use-agents";
import { cn } from "@/lib/utils";

const CELL_SIZE = 30;
const CELL_COLORS = {
  empty: "#f3f4f6",
  wall: "#374151",
  resource: "#10b981",
  agent: "#3b82f6",
};

const AGENT_STATE_COLORS = {
  idle: "#6b7280",
  exploring: "#3b82f6",
  collecting: "#10b981",
  returning: "#f59e0b",
};

interface TooltipData {
  x: number;
  y: number;
  content: React.ReactNode;
}

export function SimulationGrid() {
  const {
    isRunning,
    isPaused,
    currentStep,
    grid,
    agents,
    startSimulation,
    stopSimulation,
    pauseSimulation,
    resumeSimulation,
    stepSimulation,
    resetSimulation,
    speed,
    setSpeed,
    gridSize,
    setGridSize,
    metrics,
  } = useSimulation();

  const { agents: registeredAgents } = useAgents();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [gridDimensions, setGridDimensions] = useState({ width: 20, height: 20 });

  // Draw grid on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    try {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw grid cells
      grid.forEach((row) => {
        row.forEach((cell) => {
          const x = cell.x * CELL_SIZE;
          const y = cell.y * CELL_SIZE;

          // Fill cell based on type
          ctx.fillStyle = CELL_COLORS[cell.type];
          ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

          // Draw resource if present
          if (cell.resource) {
            ctx.fillStyle = CELL_COLORS.resource;
            ctx.beginPath();
            ctx.arc(x + CELL_SIZE / 2, y + CELL_SIZE / 2, CELL_SIZE / 3, 0, Math.PI * 2);
            ctx.fill();
          }

          // Draw grid lines
          ctx.strokeStyle = "#e5e7eb";
          ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
        });
      });

      // Draw agents
      agents.forEach((agent) => {
        const x = agent.position.x * CELL_SIZE;
        const y = agent.position.y * CELL_SIZE;

        // Agent body
        ctx.fillStyle = AGENT_STATE_COLORS[agent.state];
        ctx.beginPath();
        ctx.arc(x + CELL_SIZE / 2, y + CELL_SIZE / 2, CELL_SIZE / 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Energy indicator
        const energyPercent = agent.energy / 100;
        ctx.strokeStyle = energyPercent > 0.3 ? "#10b981" : "#ef4444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(
          x + CELL_SIZE / 2,
          y + CELL_SIZE / 2,
          CELL_SIZE / 2.5 + 3,
          -Math.PI / 2,
          -Math.PI / 2 + Math.PI * 2 * energyPercent,
        );
        ctx.stroke();
      });
    } catch (error) {
      // Canvas operations can fail in test environments
      console.warn("Canvas rendering error:", error);
    }
  }, [grid, agents]);

  // Handle mouse hover for tooltips
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
    const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);

    if (x >= 0 && x < gridSize.width && y >= 0 && y < gridSize.height) {
      const cell = grid[y]?.[x];
      if (!cell) return;

      if (cell.occupant) {
        setTooltip({
          x: e.clientX,
          y: e.clientY,
          content: (
            <div data-testid="agent-tooltip" className="space-y-1">
              <div className="font-semibold">{cell.occupant.name}</div>
              <div className="text-xs">State: {cell.occupant.state}</div>
              <div className="text-xs">Energy: {cell.occupant.energy}</div>
              {cell.occupant.beliefs && (
                <div className="text-xs">
                  Beliefs: {JSON.stringify(cell.occupant.beliefs, null, 2)}
                </div>
              )}
            </div>
          ),
        });
      } else if (cell.resource) {
        setTooltip({
          x: e.clientX,
          y: e.clientY,
          content: (
            <div className="space-y-1">
              <div className="font-semibold">Resource</div>
              <div className="text-xs">Type: {cell.resource.type}</div>
              <div className="text-xs">Amount: {cell.resource.amount}</div>
            </div>
          ),
        });
      } else {
        setTooltip(null);
      }
    }
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  const handlePlayPause = () => {
    if (isRunning) {
      if (isPaused) {
        resumeSimulation();
      } else {
        pauseSimulation();
      }
    } else {
      startSimulation();
    }
  };

  const handleGridSizeChange = (dimension: "width" | "height", value: string) => {
    const size = parseInt(value, 10);
    if (!isNaN(size) && size > 0 && size <= 50) {
      setGridDimensions((prev) => ({ ...prev, [dimension]: size }));
      setGridSize({ ...gridSize, [dimension]: size });
    }
  };

  const hasSimulation = grid.length > 0;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Simulation Grid</CardTitle>
            <CardDescription>
              Step: {currentStep} • {agents.length} agents
              {metrics && (
                <span className="ml-2">
                  • {metrics.fps} FPS • {metrics.avgStepTime}ms/step
                </span>
              )}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {/* Simulation Controls */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handlePlayPause}
                    disabled={registeredAgents.length === 0}
                    aria-label={isRunning && !isPaused ? "Pause" : "Start"}
                  >
                    {isRunning && !isPaused ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{isRunning && !isPaused ? "Pause" : "Start"}</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={stepSimulation}
                    disabled={isRunning && !isPaused}
                    aria-label="Step"
                  >
                    <SkipForward className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Step</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={resetSimulation}
                    aria-label="Reset"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Reset</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {isRunning && (
              <Button variant="destructive" size="sm" onClick={stopSimulation}>
                Stop
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Controls Row */}
        <div className="flex flex-wrap gap-4">
          {/* Speed Control */}
          <div className="flex items-center gap-2">
            <Label htmlFor="speed-slider" className="text-sm">
              <Zap className="h-4 w-4 inline mr-1" />
              Speed
            </Label>
            <Slider
              id="speed-slider"
              value={[speed]}
              onValueChange={([value]) => setSpeed(value)}
              min={0.5}
              max={10}
              step={0.5}
              className="w-24"
              aria-label="Speed"
            />
            <span className="text-sm font-medium w-8">{speed}x</span>
          </div>

          {/* Grid Size Controls */}
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            <Label htmlFor="grid-width" className="text-sm">
              Grid Width
            </Label>
            <Input
              id="grid-width"
              type="number"
              value={gridDimensions.width}
              onChange={(e) => handleGridSizeChange("width", e.target.value)}
              disabled={isRunning}
              className="w-16 h-8"
              min={5}
              max={50}
            />
            <Label htmlFor="grid-height" className="text-sm">
              Height
            </Label>
            <Input
              id="grid-height"
              type="number"
              value={gridDimensions.height}
              onChange={(e) => handleGridSizeChange("height", e.target.value)}
              disabled={isRunning}
              className="w-16 h-8"
              min={5}
              max={50}
            />
          </div>

          {/* Performance Metrics */}
          {metrics && (
            <div className="flex items-center gap-2 ml-auto">
              <Activity className="h-4 w-4" />
              <Badge variant="secondary">{metrics.fps} FPS</Badge>
              <Badge variant="secondary">{metrics.avgStepTime}ms/step</Badge>
            </div>
          )}
        </div>

        {/* Grid Display */}
        <div className="flex-1 relative">
          {!hasSimulation ? (
            <div className="h-full flex items-center justify-center text-center">
              <div className="text-muted-foreground">
                <p className="text-sm font-medium">No active simulation</p>
                <p className="text-xs mt-1">Create agents and start a simulation</p>
              </div>
            </div>
          ) : (
            <div
              ref={containerRef}
              className="h-full overflow-auto bg-muted/10 rounded-lg relative"
            >
              <canvas
                ref={canvasRef}
                data-testid="simulation-canvas"
                width={gridSize.width * CELL_SIZE}
                height={gridSize.height * CELL_SIZE}
                className="border border-border"
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
              />

              {/* Tooltip */}
              {tooltip && (
                <div
                  className="absolute z-10 bg-popover text-popover-foreground p-2 rounded-md shadow-md border text-xs pointer-events-none"
                  style={{
                    left: tooltip.x + 10,
                    top: tooltip.y + 10,
                  }}
                >
                  {tooltip.content}
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
