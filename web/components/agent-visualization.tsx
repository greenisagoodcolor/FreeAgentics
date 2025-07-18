"use client";

import React, { useEffect, useRef, useState } from "react";
import { Badge } from "./ui/badge";
import { Card } from "./ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";

interface Agent {
  id: string;
  name: string;
  status: string;
  type?: string;
  thinking?: boolean;
  position?: { x: number; y: number };
  connections?: string[];
  metrics?: {
    free_energy?: number;
    uncertainty?: number;
    actions_taken?: number;
  };
}

interface AgentVisualizationProps {
  agents: Agent[];
  className?: string;
}

export function AgentVisualization({ agents, className = "" }: AgentVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [view, setView] = useState<"network" | "list" | "metrics">("network");
  const animationRef = useRef<number>();

  // Initialize agent positions
  useEffect(() => {
    agents.forEach((agent, index) => {
      if (!agent.position) {
        const angle = (index / agents.length) * Math.PI * 2;
        const radius = 150;
        agent.position = {
          x: Math.cos(angle) * radius + 200,
          y: Math.sin(angle) * radius + 200,
        };
      }
    });
  }, [agents]);

  // Canvas rendering for network view
  useEffect(() => {
    if (view !== "network" || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const updateCanvasSize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connections
      agents.forEach((agent) => {
        if (agent.connections) {
          agent.connections.forEach((targetId) => {
            const target = agents.find((a) => a.id === targetId);
            if (target && agent.position && target.position) {
              ctx.beginPath();
              ctx.moveTo(agent.position.x, agent.position.y);
              ctx.lineTo(target.position.x, target.position.y);
              ctx.strokeStyle = "#e5e7eb";
              ctx.lineWidth = 1;
              ctx.stroke();
            }
          });
        }
      });

      // Draw agents
      agents.forEach((agent) => {
        if (!agent.position) return;

        const { x, y } = agent.position;
        const radius = 30;

        // Agent circle
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = getAgentColor(agent.status);
        ctx.fill();
        ctx.strokeStyle = selectedAgent?.id === agent.id ? "#3b82f6" : "#d1d5db";
        ctx.lineWidth = selectedAgent?.id === agent.id ? 3 : 1;
        ctx.stroke();

        // Thinking indicator
        if (agent.thinking) {
          ctx.beginPath();
          ctx.arc(x, y, radius + 10, 0, Math.PI * 2);
          ctx.strokeStyle = "#3b82f6";
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Agent name
        ctx.fillStyle = "#1f2937";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(agent.name, x, y + radius + 5);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", updateCanvasSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [agents, selectedAgent, view]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find clicked agent
    const clickedAgent = agents.find((agent) => {
      if (!agent.position) return false;
      const dist = Math.sqrt(Math.pow(x - agent.position.x, 2) + Math.pow(y - agent.position.y, 2));
      return dist <= 30;
    });

    setSelectedAgent(clickedAgent || null);
  };

  const getAgentColor = (status: string) => {
    switch (status) {
      case "active":
        return "#10b981";
      case "thinking":
        return "#3b82f6";
      case "idle":
        return "#f59e0b";
      case "error":
        return "#ef4444";
      default:
        return "#6b7280";
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "active":
        return "success";
      case "thinking":
        return "info";
      case "idle":
        return "warning";
      case "error":
        return "destructive";
      default:
        return "secondary";
    }
  };

  return (
    <div className={`h-full flex flex-col ${className}`}>
      <Tabs value={view} onValueChange={(v) => setView(v as any)}>
        <TabsList className="mb-4">
          <TabsTrigger value="network">Network View</TabsTrigger>
          <TabsTrigger value="list">List View</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="network" className="flex-1">
          <div className="relative h-full">
            <canvas
              ref={canvasRef}
              className="w-full h-full cursor-pointer"
              onClick={handleCanvasClick}
              style={{ minHeight: "400px" }}
            />

            {selectedAgent && (
              <Card className="absolute bottom-4 right-4 p-4 max-w-xs">
                <h3 className="font-semibold mb-2">{selectedAgent.name}</h3>
                <Badge variant={getStatusBadgeVariant(selectedAgent.status)}>
                  {selectedAgent.status}
                </Badge>
                {selectedAgent.metrics && (
                  <div className="mt-2 text-sm text-gray-600">
                    <div>Free Energy: {selectedAgent.metrics.free_energy?.toFixed(2)}</div>
                    <div>Uncertainty: {selectedAgent.metrics.uncertainty?.toFixed(2)}</div>
                    <div>Actions: {selectedAgent.metrics.actions_taken}</div>
                  </div>
                )}
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="list" className="flex-1 overflow-y-auto">
          <div className="space-y-2">
            {agents.map((agent) => (
              <Card
                key={agent.id}
                className="p-4 cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedAgent(agent)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">{agent.name}</h3>
                    <p className="text-sm text-gray-600">{agent.type || "Standard"}</p>
                  </div>
                  <Badge variant={getStatusBadgeVariant(agent.status)}>{agent.status}</Badge>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="flex-1">
          <div className="grid grid-cols-2 gap-4">
            {agents.map((agent) => (
              <Card key={agent.id} className="p-4">
                <h4 className="font-semibold mb-2">{agent.name}</h4>
                {agent.metrics ? (
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span>Free Energy:</span>
                      <span>{agent.metrics.free_energy?.toFixed(3) || "N/A"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Uncertainty:</span>
                      <span>{agent.metrics.uncertainty?.toFixed(3) || "N/A"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Actions:</span>
                      <span>{agent.metrics.actions_taken || 0}</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No metrics available</p>
                )}
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
