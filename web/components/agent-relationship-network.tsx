"use client";

import { CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Agent } from "@/lib/types";
import { Users } from "lucide-react";
import { useEffect, useRef, useCallback } from "react";

interface AgentRelationshipNetworkProps {
  agents: Agent[];
}

interface INetworkNode {
  id: string;
  name: string;
  x: number;
  y: number;
  color: string;
}

interface INetworkLink {
  source: string;
  target: string;
  strength: number;
}

export default function AgentRelationshipNetwork({
  agents,
}: AgentRelationshipNetworkProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate mock relationship data
  const generateRelationships = useCallback((): INetworkLink[] => {
    const links: INetworkLink[] = [];

    // Create some random relationships between agents
    agents.forEach((agent, i) => {
      // Each agent has 1-3 relationships
      const numRelationships = Math.floor(Math.random() * 3) + 1;

      for (let j = 0; j < numRelationships; j++) {
        const targetIndex = Math.floor(Math.random() * agents.length);
        if (targetIndex !== i) {
          const existingLink = links.find(
            (l) =>
              (l.source === agent.id && l.target === agents[targetIndex].id) ||
              (l.target === agent.id && l.source === agents[targetIndex].id),
          );

          if (!existingLink) {
            links.push({
              source: agent.id,
              target: agents[targetIndex].id,
              strength: Math.random(),
            });
          }
        }
      }
    });

    return links;
  }, [agents]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 50;

    // Position nodes in a circle
    const nodes: INetworkNode[] = agents.map((agent, index) => {
      const angle = (index / agents.length) * 2 * Math.PI;
      return {
        id: agent.id,
        name: agent.name,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        color: agent.color,
      };
    });

    const links = generateRelationships();

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw links
    links.forEach((link) => {
      const sourceNode = nodes.find((n) => n.id === link.source);
      const targetNode = nodes.find((n) => n.id === link.target);

      if (sourceNode && targetNode) {
        ctx.beginPath();
        ctx.moveTo(sourceNode.x, sourceNode.y);
        ctx.lineTo(targetNode.x, targetNode.y);
        ctx.strokeStyle = `rgba(147, 51, 234, ${link.strength * 0.5})`; // Purple with varying opacity
        ctx.lineWidth = link.strength * 3;
        ctx.stroke();
      }
    });

    // Draw nodes
    nodes.forEach((node) => {
      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, 20, 0, 2 * Math.PI);
      ctx.fillStyle = node.color;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Node label
      ctx.fillStyle = "#ffffff";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.fillText(node.name, node.x, node.y + 35);
    });

    // Draw legend
    ctx.fillStyle = "#ffffff";
    ctx.font = "14px Arial";
    ctx.textAlign = "left";
    ctx.fillText("Relationship Strength", 20, 30);

    // Legend gradient
    const gradient = ctx.createLinearGradient(20, 40, 120, 40);
    gradient.addColorStop(0, "rgba(147, 51, 234, 0.1)");
    gradient.addColorStop(1, "rgba(147, 51, 234, 0.5)");
    ctx.fillStyle = gradient;
    ctx.fillRect(20, 40, 100, 10);

    ctx.fillStyle = "#ffffff";
    ctx.font = "10px Arial";
    ctx.fillText("Weak", 20, 65);
    ctx.fillText("Strong", 85, 65);
  }, [agents, generateRelationships]);

  return (
    <div className="h-full flex flex-col">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5" />
          <CardTitle>Agent Relationships</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="flex-1 relative">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        <div className="absolute bottom-4 left-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg p-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Total Agents</div>
              <div className="font-medium">{agents.length}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Active Relationships</div>
              <div className="font-medium">
                {generateRelationships().length}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </div>
  );
}
