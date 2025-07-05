import React, { useMemo, useRef, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Network } from "lucide-react";
import type { Agent } from "@/lib/types";
import type { KnowledgeEntry } from "@/types/memory-viewer";

interface GraphViewProps {
  selectedAgent: Agent | null;
  agents: Agent[];
  knowledge: KnowledgeEntry[];
}

interface GraphNode {
  id: string;
  label: string;
  type: "agent" | "knowledge" | "tag";
  x?: number;
  y?: number;
  color?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  type: "knows" | "tagged" | "related";
}

export function GraphView({ selectedAgent, knowledge }: GraphViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // Build graph data
  const { nodes, edges } = useMemo(() => {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const tagNodes = new Map<string, GraphNode>();

    // Add agent node
    if (selectedAgent) {
      nodes.push({
        id: selectedAgent.id,
        label: selectedAgent.name,
        type: "agent",
        color: selectedAgent.color || "#3b82f6",
      });

      // Add knowledge nodes
      knowledge.forEach((k) => {
        nodes.push({
          id: k.id,
          label: k.title,
          type: "knowledge",
          color: "#10b981",
        });

        // Connect agent to knowledge
        edges.push({
          source: selectedAgent.id,
          target: k.id,
          type: "knows",
        });

        // Add tag nodes and connections
        k.tags.forEach((tag) => {
          if (!tagNodes.has(tag)) {
            const tagNode: GraphNode = {
              id: `tag-${tag}`,
              label: tag,
              type: "tag",
              color: "#f59e0b",
            };
            tagNodes.set(tag, tagNode);
            nodes.push(tagNode);
          }

          edges.push({
            source: k.id,
            target: `tag-${tag}`,
            type: "tagged",
          });
        });

        // Add related agent connections
        if (k.relatedAgents) {
          k.relatedAgents.forEach((agentId) => {
            edges.push({
              source: k.id,
              target: agentId,
              type: "related",
            });
          });
        }
      });
    }

    // Initialize node positions
    const centerX = 400;
    const centerY = 300;
    const radius = 200;

    nodes.forEach((node, index) => {
      const angle = (index / nodes.length) * Math.PI * 2;
      node.x = centerX + Math.cos(angle) * radius;
      node.y = centerY + Math.sin(angle) * radius;
    });

    return { nodes, edges };
  }, [selectedAgent, knowledge]);

  // Force-directed layout simulation
  useEffect(() => {
    if (!canvasRef.current || nodes.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    let mouseX = 0;
    let mouseY = 0;
    let hoveredNode: GraphNode | null = null;

    // Mouse tracking
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;

      // Check for hovered node
      hoveredNode = null;
      nodes.forEach((node) => {
        if (!node.x || !node.y) return;
        const dx = mouseX - node.x!;
        const dy = mouseY - node.y!;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 20) {
          hoveredNode = node;
          canvas.style.cursor = "pointer";
          return;
        }
      });

      if (!hoveredNode) {
        canvas.style.cursor = "default";
      }
    };

    canvas.addEventListener("mousemove", handleMouseMove);

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Apply forces
      nodes.forEach((node, i) => {
        if (!node.x || !node.y) return;

        // Repulsion between nodes
        nodes.forEach((other, j) => {
          if (i === j || !other.x || !other.y) return;

          const dx = node.x! - other.x!;
          const dy = node.y! - other.y!;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100 && distance > 0) {
            const force = 50 / distance;
            node.x! += (dx / distance) * force * 0.05;
            node.y! += (dy / distance) * force * 0.05;
          }
        });

        // Attraction to center
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const dx = centerX - node.x!;
        const dy = centerY - node.y!;
        node.x! += dx * 0.001;
        node.y! += dy * 0.001;

        // Keep nodes within bounds
        node.x = Math.max(30, Math.min(canvas.width - 30, node.x!));
        node.y = Math.max(30, Math.min(canvas.height - 30, node.y!));
      });

      // Draw edges
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      edges.forEach((edge) => {
        const source = nodes.find((n) => n.id === edge.source);
        const target = nodes.find((n) => n.id === edge.target);

        if (source?.x && source?.y && target?.x && target?.y) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);

          // Different styles for different edge types
          if (edge.type === "related") {
            ctx.setLineDash([5, 5]);
          } else {
            ctx.setLineDash([]);
          }

          ctx.stroke();
        }
      });

      // Draw nodes
      nodes.forEach((node) => {
        if (!node.x || !node.y) return;

        // Node circle
        ctx.beginPath();
        ctx.arc(
          node.x!,
          node.y!,
          hoveredNode === node ? 25 : 20,
          0,
          Math.PI * 2,
        );
        ctx.fillStyle = node.color || "#6b7280";
        ctx.fill();

        if (hoveredNode === node) {
          ctx.strokeStyle = "#1f2937";
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Node icon
        ctx.fillStyle = "white";
        ctx.font = "16px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        const icon =
          node.type === "agent"
            ? "👤"
            : node.type === "knowledge"
              ? "📄"
              : "🏷️";
        ctx.fillText(icon, node.x!, node.y!);

        // Node label (on hover)
        if (hoveredNode === node) {
          ctx.fillStyle = "#1f2937";
          ctx.font = "12px Arial";
          ctx.fillText(node.label, node.x!, node.y! + 35);
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      canvas.removeEventListener("mousemove", handleMouseMove);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, edges]);

  if (!selectedAgent) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent>
          <p className="text-center text-muted-foreground">
            Select an agent to view their knowledge graph
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="h-full space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Network className="h-5 w-5" />
          Knowledge Graph
        </h3>
        <div className="flex gap-2">
          <Badge variant="outline">
            <span className="w-3 h-3 rounded-full bg-blue-500 mr-2" />
            Agent
          </Badge>
          <Badge variant="outline">
            <span className="w-3 h-3 rounded-full bg-green-500 mr-2" />
            Knowledge
          </Badge>
          <Badge variant="outline">
            <span className="w-3 h-3 rounded-full bg-amber-500 mr-2" />
            Tags
          </Badge>
        </div>
      </div>

      <Card className="flex-1">
        <CardContent className="p-0 h-[500px]">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ background: "#f9fafb" }}
          />
        </CardContent>
      </Card>

      <div className="text-sm text-muted-foreground">
        <p>
          Interactive knowledge graph visualization. Hover over nodes to see
          details.
        </p>
      </div>
    </div>
  );
}
