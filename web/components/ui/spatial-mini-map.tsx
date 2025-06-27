"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Grid, Users, Target, Shuffle, RotateCcw } from "lucide-react";

interface Agent {
  id: string;
  name: string;
  position: { x: number; y: number };
  color: string;
  proximityRadius: number;
  isSelected?: boolean;
}

interface GridSize {
  width: number;
  height: number;
  label: string;
}

interface SpatialMiniMapProps {
  agents: Agent[];
  gridSize: GridSize;
  proximityThreshold: number;
  onAgentMove: (agentId: string, newPosition: { x: number; y: number }) => void;
  onGridSizeChange: (newSize: GridSize) => void;
  onProximityThresholdChange: (threshold: number) => void;
  onAutoArrange: (type: string) => void;
  showMovementTrails?: boolean;
  onAgentSelect?: (agentId: string) => void;
  className?: string;
}

const GRID_SIZES: GridSize[] = [
  { width: 5, height: 5, label: "5x5" },
  { width: 10, height: 10, label: "10x10" },
  { width: 20, height: 20, label: "20x20" },
];

const CELL_SIZE = 15; // Size of each grid cell in pixels
const MAP_PADDING = 20; // Padding around the map

export function SpatialMiniMap({
  agents,
  gridSize,
  proximityThreshold,
  onAgentMove,
  onGridSizeChange,
  onProximityThresholdChange,
  onAutoArrange,
  showMovementTrails = false,
  onAgentSelect,
  className = "",
}: SpatialMiniMapProps) {
  const [draggedAgent, setDraggedAgent] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [hoveredAgent, setHoveredAgent] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const mapRef = useRef<SVGSVGElement>(null);

  const mapWidth = gridSize.width * CELL_SIZE + MAP_PADDING * 2;
  const mapHeight = gridSize.height * CELL_SIZE + MAP_PADDING * 2;

  // Calculate proximity connections
  const proximityConnections = React.useMemo(() => {
    const connections: Array<{
      agent1: Agent;
      agent2: Agent;
      distance: number;
    }> = [];

    for (let i = 0; i < agents.length; i++) {
      for (let j = i + 1; j < agents.length; j++) {
        const agent1 = agents[i];
        const agent2 = agents[j];
        const distance = Math.sqrt(
          Math.pow(agent1.position.x - agent2.position.x, 2) +
            Math.pow(agent1.position.y - agent2.position.y, 2),
        );

        if (distance <= proximityThreshold) {
          connections.push({ agent1, agent2, distance });
        }
      }
    }

    return connections;
  }, [agents, proximityThreshold]);

  // Convert grid coordinates to SVG coordinates
  const gridToSvg = useCallback((gridX: number, gridY: number) => {
    return {
      x: gridX * CELL_SIZE + MAP_PADDING + CELL_SIZE / 2,
      y: gridY * CELL_SIZE + MAP_PADDING + CELL_SIZE / 2,
    };
  }, []);

  // Convert SVG coordinates to grid coordinates
  const svgToGrid = useCallback(
    (svgX: number, svgY: number) => {
      const gridX = Math.round(
        (svgX - MAP_PADDING - CELL_SIZE / 2) / CELL_SIZE,
      );
      const gridY = Math.round(
        (svgY - MAP_PADDING - CELL_SIZE / 2) / CELL_SIZE,
      );

      return {
        x: Math.max(0, Math.min(gridSize.width - 1, gridX)),
        y: Math.max(0, Math.min(gridSize.height - 1, gridY)),
      };
    },
    [gridSize],
  );

  // Handle mouse down on agent
  const handleAgentMouseDown = useCallback(
    (event: React.MouseEvent, agentId: string) => {
      event.preventDefault();
      const agent = agents.find((a) => a.id === agentId);
      if (!agent) return;

      const rect = mapRef.current?.getBoundingClientRect();
      if (!rect) return;

      const svgPos = gridToSvg(agent.position.x, agent.position.y);
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      setDraggedAgent(agentId);
      setDragOffset({
        x: mouseX - svgPos.x,
        y: mouseY - svgPos.y,
      });

      setSelectedAgent(agentId);
      onAgentSelect?.(agentId);
    },
    [agents, gridToSvg, onAgentSelect],
  );

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (event: React.MouseEvent) => {
      if (!draggedAgent) return;

      const rect = mapRef.current?.getBoundingClientRect();
      if (!rect) return;

      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      const svgX = mouseX - dragOffset.x;
      const svgY = mouseY - dragOffset.y;

      const gridPos = svgToGrid(svgX, svgY);

      // Update agent position temporarily for visual feedback
      const agentIndex = agents.findIndex((a) => a.id === draggedAgent);
      if (agentIndex !== -1) {
        agents[agentIndex].position = gridPos;
      }
    },
    [draggedAgent, dragOffset, svgToGrid, agents],
  );

  // Handle mouse up to complete drag
  const handleMouseUp = useCallback(
    (event: React.MouseEvent) => {
      if (!draggedAgent) return;

      const rect = mapRef.current?.getBoundingClientRect();
      if (!rect) return;

      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      const svgX = mouseX - dragOffset.x;
      const svgY = mouseY - dragOffset.y;

      const gridPos = svgToGrid(svgX, svgY);
      onAgentMove(draggedAgent, gridPos);

      setDraggedAgent(null);
      setDragOffset({ x: 0, y: 0 });
    },
    [draggedAgent, dragOffset, svgToGrid, onAgentMove],
  );

  // Handle grid size change
  const handleGridSizeChange = (sizeLabel: string) => {
    const newSize = GRID_SIZES.find((s) => s.label === sizeLabel);
    if (newSize) {
      onGridSizeChange(newSize);
    }
  };

  // Render grid lines
  const renderGridLines = () => {
    const lines = [];

    // Vertical lines
    for (let i = 0; i <= gridSize.width; i++) {
      const x = i * CELL_SIZE + MAP_PADDING;
      lines.push(
        <line
          key={`v-${i}`}
          x1={x}
          y1={MAP_PADDING}
          x2={x}
          y2={gridSize.height * CELL_SIZE + MAP_PADDING}
          stroke="#e5e7eb"
          strokeWidth="1"
        />,
      );
    }

    // Horizontal lines
    for (let i = 0; i <= gridSize.height; i++) {
      const y = i * CELL_SIZE + MAP_PADDING;
      lines.push(
        <line
          key={`h-${i}`}
          x1={MAP_PADDING}
          y1={y}
          x2={gridSize.width * CELL_SIZE + MAP_PADDING}
          y2={y}
          stroke="#e5e7eb"
          strokeWidth="1"
        />,
      );
    }

    return lines;
  };

  // Render proximity circles
  const renderProximityCircles = () => {
    return agents.map((agent) => {
      if (agent.id === hoveredAgent || agent.id === selectedAgent) {
        const svgPos = gridToSvg(agent.position.x, agent.position.y);
        const radius = agent.proximityRadius * CELL_SIZE;

        return (
          <circle
            key={`proximity-${agent.id}`}
            cx={svgPos.x}
            cy={svgPos.y}
            r={radius}
            fill={agent.color}
            fillOpacity="0.15"
            stroke={agent.color}
            strokeWidth="1"
            strokeDasharray="3,3"
          />
        );
      }
      return null;
    });
  };

  // Render connection lines
  const renderConnections = () => {
    return proximityConnections.map(({ agent1, agent2, distance }, index) => {
      const pos1 = gridToSvg(agent1.position.x, agent1.position.y);
      const pos2 = gridToSvg(agent2.position.x, agent2.position.y);

      // Calculate opacity based on distance (closer = more opaque)
      const opacity = Math.max(0.2, 1 - distance / proximityThreshold);

      return (
        <line
          key={`connection-${index}`}
          x1={pos1.x}
          y1={pos1.y}
          x2={pos2.x}
          y2={pos2.y}
          stroke="#3b82f6"
          strokeWidth="2"
          strokeOpacity={opacity}
        />
      );
    });
  };

  // Render agents
  const renderAgents = () => {
    return agents.map((agent) => {
      const svgPos = gridToSvg(agent.position.x, agent.position.y);
      const isSelected = agent.id === selectedAgent;
      const isDragged = agent.id === draggedAgent;
      const isHovered = agent.id === hoveredAgent;

      return (
        <g key={agent.id}>
          {/* Agent circle */}
          <circle
            cx={svgPos.x}
            cy={svgPos.y}
            r={8}
            fill={agent.color}
            stroke={isSelected ? "#000" : "#fff"}
            strokeWidth={isSelected ? 3 : 2}
            className={`cursor-pointer transition-all ${isDragged ? "scale-110" : ""} ${isHovered ? "scale-105" : ""}`}
            onMouseDown={(e) => handleAgentMouseDown(e, agent.id)}
            onMouseEnter={() => setHoveredAgent(agent.id)}
            onMouseLeave={() => setHoveredAgent(null)}
          />

          {/* Agent label */}
          <text
            x={svgPos.x}
            y={svgPos.y + 20}
            textAnchor="middle"
            fontSize="10"
            fill="#374151"
            className="pointer-events-none select-none"
          >
            {agent.name.split(" ")[0]}
          </text>
        </g>
      );
    });
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Grid className="h-5 w-5" />
          Spatial Mini-Map
          <Badge variant="outline" className="ml-auto">
            {agents.length} agents
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Grid Size:</label>
            <Select value={gridSize.label} onValueChange={handleGridSizeChange}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {GRID_SIZES.map((size) => (
                  <SelectItem key={size.label} value={size.label}>
                    {size.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2 min-w-32">
            <label className="text-sm font-medium">Proximity:</label>
            <Slider
              value={[proximityThreshold]}
              onValueChange={([value]) => onProximityThresholdChange(value)}
              min={1}
              max={5}
              step={1}
              className="flex-1"
            />
            <span className="text-sm text-muted-foreground w-4">
              {proximityThreshold}
            </span>
          </div>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAutoArrange("grid")}
            >
              <Target className="h-4 w-4 mr-1" />
              Grid
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAutoArrange("circle")}
            >
              <Users className="h-4 w-4 mr-1" />
              Circle
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAutoArrange("random")}
            >
              <Shuffle className="h-4 w-4 mr-1" />
              Random
            </Button>
          </div>
        </div>

        {/* Map */}
        <div className="relative bg-muted/30 rounded-lg overflow-hidden">
          <svg
            ref={mapRef}
            width={mapWidth}
            height={mapHeight}
            className="border rounded"
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => {
              setDraggedAgent(null);
              setHoveredAgent(null);
            }}
          >
            {/* Grid lines */}
            {renderGridLines()}

            {/* Proximity circles */}
            {renderProximityCircles()}

            {/* Connection lines */}
            {renderConnections()}

            {/* Agents */}
            {renderAgents()}
          </svg>

          {/* Info overlay */}
          {proximityConnections.length > 0 && (
            <div className="absolute top-2 right-2 bg-white/90 rounded px-2 py-1 text-xs">
              {proximityConnections.length} connection
              {proximityConnections.length !== 1 ? "s" : ""}
            </div>
          )}
        </div>

        {/* Status */}
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div>Click and drag agents to move them</div>
          <div className="flex items-center gap-4">
            <span>
              Grid: {gridSize.width}×{gridSize.height}
            </span>
            <span>Proximity: {proximityThreshold} cells</span>
            <span>Connections: {proximityConnections.length}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
