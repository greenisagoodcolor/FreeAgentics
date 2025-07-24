"use client";

import React, { useEffect, useRef, useState } from "react";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";

interface KnowledgeGraphNode {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, unknown>;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface KnowledgeGraphEdge {
  source: string;
  target: string;
  relationship: string;
}

interface KnowledgeGraph {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
}

interface KnowledgeGraphViewProps {
  graph: KnowledgeGraph;
  className?: string;
}

export function KnowledgeGraphView({ graph, className = "" }: KnowledgeGraphViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedNode, setSelectedNode] = useState<KnowledgeGraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredGraph, setFilteredGraph] = useState<KnowledgeGraph>(graph);
  const [view, setView] = useState<"graph" | "list" | "stats">("graph");
  const animationRef = useRef<number>();
  const simulationRef = useRef<(() => void) | null>(null);

  // Filter graph based on search
  useEffect(() => {
    if (!searchQuery) {
      setFilteredGraph(graph);
      return;
    }

    const query = searchQuery.toLowerCase();
    const matchingNodes = graph.nodes.filter(
      (node) => node.label.toLowerCase().includes(query) || node.type.toLowerCase().includes(query),
    );

    const matchingNodeIds = new Set(matchingNodes.map((n) => n.id));

    // Include edges connected to matching nodes
    const matchingEdges = graph.edges.filter(
      (edge) => matchingNodeIds.has(edge.source) || matchingNodeIds.has(edge.target),
    );

    // Include nodes connected by matching edges
    matchingEdges.forEach((edge) => {
      matchingNodeIds.add(edge.source);
      matchingNodeIds.add(edge.target);
    });

    setFilteredGraph({
      nodes: graph.nodes.filter((n) => matchingNodeIds.has(n.id)),
      edges: matchingEdges,
    });
  }, [graph, searchQuery]);

  // Initialize node positions with force simulation
  useEffect(() => {
    if (view !== "graph") return;

    // Initialize positions
    filteredGraph.nodes.forEach((node, i) => {
      if (!node.x || !node.y) {
        const angle = (i / filteredGraph.nodes.length) * Math.PI * 2;
        const radius = 100;
        node.x = Math.cos(angle) * radius + 200;
        node.y = Math.sin(angle) * radius + 200;
        node.vx = 0;
        node.vy = 0;
      }
    });

    // Simple force simulation
    const simulate = () => {
      const alpha = 0.1;
      const centerX = 200;
      const centerY = 200;

      // Apply forces
      filteredGraph.nodes.forEach((node) => {
        if (!node.x || !node.y) return;

        // Repulsion between nodes
        filteredGraph.nodes.forEach((other) => {
          if (node.id === other.id || !node.x || !node.y || !other.x || !other.y) return;

          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100 && distance > 0) {
            const force = ((100 - distance) / distance) * 0.5;
            node.vx! += dx * force;
            node.vy! += dy * force;
          }
        });

        // Attraction along edges
        filteredGraph.edges.forEach((edge) => {
          let other: KnowledgeGraphNode | undefined;
          if (edge.source === node.id) {
            other = filteredGraph.nodes.find((n) => n.id === edge.target);
          } else if (edge.target === node.id) {
            other = filteredGraph.nodes.find((n) => n.id === edge.source);
          }

          if (other && other.x && other.y && node.x && node.y) {
            const dx = other.x - node.x;
            const dy = other.y - node.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance > 50) {
              const force = ((distance - 50) / distance) * 0.1;
              node.vx! += dx * force;
              node.vy! += dy * force;
            }
          }
        });

        // Center gravity
        if (!node.x || !node.y) return;
        const dx = centerX - node.x;
        const dy = centerY - node.y;
        node.vx! += dx * 0.01;
        node.vy! += dy * 0.01;

        // Apply velocity with damping
        if (!node.vx || !node.vy || !node.x || !node.y) return;
        node.vx *= 0.9;
        node.vy *= 0.9;
        node.x += node.vx * alpha;
        node.y += node.vy * alpha;

        // Keep within bounds
        node.x = Math.max(30, Math.min(370, node.x));
        node.y = Math.max(30, Math.min(370, node.y));
      });
    };

    simulationRef.current = simulate;
  }, [filteredGraph, view]);

  // Canvas rendering
  useEffect(() => {
    if (view !== "graph" || !canvasRef.current) return;

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

      // Run simulation
      if (simulationRef.current) {
        simulationRef.current();
      }

      // Draw edges
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      filteredGraph.edges.forEach((edge) => {
        const source = filteredGraph.nodes.find((n) => n.id === edge.source);
        const target = filteredGraph.nodes.find((n) => n.id === edge.target);

        if (source?.x && source?.y && target?.x && target?.y) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();

          // Draw relationship label
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          ctx.fillStyle = "#6b7280";
          ctx.font = "10px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(edge.relationship, midX, midY);
        }
      });

      // Draw nodes
      filteredGraph.nodes.forEach((node) => {
        if (!node.x || !node.y) return;

        const radius = 20;

        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = getNodeColor(node.type);
        ctx.fill();
        ctx.strokeStyle = selectedNode?.id === node.id ? "#3b82f6" : "#d1d5db";
        ctx.lineWidth = selectedNode?.id === node.id ? 3 : 1;
        ctx.stroke();

        // Node label
        ctx.fillStyle = "#1f2937";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const label = node.label.length > 15 ? node.label.substring(0, 15) + "..." : node.label;
        ctx.fillText(label, node.x, node.y + radius + 5);
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
  }, [filteredGraph, selectedNode, view]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find clicked node
    const clickedNode = filteredGraph.nodes.find((node) => {
      if (!node.x || !node.y) return false;
      const dist = Math.sqrt(Math.pow(x - node.x, 2) + Math.pow(y - node.y, 2));
      return dist <= 20;
    });

    setSelectedNode(clickedNode || null);
  };

  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      concept: "#3b82f6",
      entity: "#10b981",
      action: "#f59e0b",
      property: "#8b5cf6",
      relationship: "#ef4444",
    };
    return colors[type] || "#6b7280";
  };

  const getNodeStats = () => {
    const typeCount: Record<string, number> = {};
    filteredGraph.nodes.forEach((node) => {
      typeCount[node.type] = (typeCount[node.type] || 0) + 1;
    });
    return typeCount;
  };

  return (
    <div className={`h-full flex flex-col ${className}`}>
      {/* Search Bar */}
      <div className="mb-4">
        <Input
          type="search"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full"
        />
      </div>

      <Tabs value={view} onValueChange={(v) => setView(v as "graph" | "list" | "stats")}>
        <TabsList className="mb-4">
          <TabsTrigger value="graph">Graph View</TabsTrigger>
          <TabsTrigger value="list">Node List</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
        </TabsList>

        <TabsContent value="graph" className="flex-1">
          <div className="relative h-full">
            <canvas
              ref={canvasRef}
              className="w-full h-full cursor-pointer border rounded-lg"
              onClick={handleCanvasClick}
              style={{ minHeight: "400px" }}
            />

            {selectedNode && (
              <Card className="absolute bottom-4 right-4 p-4 max-w-xs">
                <h3 className="font-semibold mb-2">{selectedNode.label}</h3>
                <Badge
                  className="mb-2"
                  style={{ backgroundColor: getNodeColor(selectedNode.type) }}
                >
                  {selectedNode.type}
                </Badge>
                {selectedNode.properties && (
                  <div className="mt-2 text-sm text-gray-600">
                    {Object.entries(selectedNode.properties).map(([key, value]) => (
                      <div key={key}>
                        <span className="font-medium">{key}:</span> {String(value)}
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="list" className="flex-1 overflow-y-auto">
          <div className="space-y-2">
            {filteredGraph.nodes.map((node) => (
              <Card
                key={node.id}
                className="p-3 cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedNode(node)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium">{node.label}</h4>
                    <p className="text-sm text-gray-600">ID: {node.id}</p>
                  </div>
                  <Badge style={{ backgroundColor: getNodeColor(node.type) }}>{node.type}</Badge>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="stats" className="flex-1">
          <Card className="p-4">
            <h3 className="font-semibold mb-4">Graph Statistics</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Total Nodes:</span>
                <span className="font-medium">{filteredGraph.nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Edges:</span>
                <span className="font-medium">{filteredGraph.edges.length}</span>
              </div>
              <div className="mt-4">
                <h4 className="font-medium mb-2">Node Types:</h4>
                {Object.entries(getNodeStats()).map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getNodeColor(type) }}
                      />
                      <span className="capitalize">{type}</span>
                    </div>
                    <span className="font-medium">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
