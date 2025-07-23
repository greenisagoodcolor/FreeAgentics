"use client";

import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  Trash2,
  Loader2,
  AlertCircle,
  GitBranch,
  Grid3x3,
  Circle,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  useKnowledgeGraph,
  type NodeType,
  type GraphNode,
  type GraphEdge,
} from "@/hooks/use-knowledge-graph";
import { cn } from "@/lib/utils";

type LayoutType = "force" | "hierarchical" | "circular";

const NODE_COLORS: Record<NodeType, string> = {
  agent: "#3b82f6", // blue
  belief: "#10b981", // green
  goal: "#f59e0b", // amber
  observation: "#8b5cf6", // purple
  action: "#ef4444", // red
};

const NODE_RADIUS = 20;

export function KnowledgeGraphView() {
  const { nodes, edges, isLoading, error, clearGraph } = useKnowledgeGraph();

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);

  const [selectedLayout, setSelectedLayout] = useState<LayoutType>("force");
  const [filters, setFilters] = useState<Record<NodeType, boolean>>({
    agent: true,
    belief: true,
    goal: true,
    observation: true,
    action: true,
  });

  // Filter nodes and edges based on active filters
  const filteredNodes = nodes.filter((node) => filters[node.type]);
  const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
  const filteredEdges = edges.filter(
    (edge) => filteredNodeIds.has(edge.source) && filteredNodeIds.has(edge.target),
  );

  // Initialize and update D3 visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;
    if (filteredNodes.length === 0) return;

    try {
      const container = containerRef.current;
      const width = container.clientWidth || 600;
      const height = container.clientHeight || 400;

      // Clear previous content
      d3.select(svgRef.current).selectAll("*").remove();

      const svg = d3.select(svgRef.current).attr("width", width).attr("height", height);

      // Create zoom behavior
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
          g.attr("transform", event.transform.toString());
        });

      if (svg.call) {
        svg.call(zoom);
      }

      // Create main group for transformations
      const g = svg.append("g");

      // Create arrow markers for directed edges
      svg
        .append("defs")
        .selectAll("marker")
        .data(["end"])
        .enter()
        .append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", NODE_RADIUS + 10)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#999");

      // Create force simulation
      const simulation = d3
        .forceSimulation<GraphNode>(filteredNodes)
        .force(
          "link",
          d3
            .forceLink<GraphNode, GraphEdge>(filteredEdges)
            .id((d) => d.id)
            .distance(100),
        )
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(NODE_RADIUS + 10));

      simulationRef.current = simulation;

      // Drag functions
      const dragstarted = (event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>, d: GraphNode) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.x = event.x;
        d.y = event.y;
      };

      const dragged = (event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>, d: GraphNode) => {
        d.x = event.x;
        d.y = event.y;
      };

      const dragended = (event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>, _d: GraphNode) => {
        if (!event.active) simulation.alphaTarget(0);
      };

      // Create edges
      const link = g
        .append("g")
        .selectAll("line")
        .data(filteredEdges)
        .enter()
        .append("line")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 2)
        .attr("marker-end", "url(#arrow)");

      // Create nodes
      const node = g
        .append("g")
        .selectAll("circle")
        .data(filteredNodes)
        .enter()
        .append("circle")
        .attr("r", NODE_RADIUS)
        .attr("fill", (d) => NODE_COLORS[d.type])
        .attr("stroke", "#fff")
        .attr("stroke-width", 2)
        .call(
          d3
            .drag<SVGCircleElement, GraphNode>()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended),
        );

      // Add labels
      const label = g
        .append("g")
        .selectAll("text")
        .data(filteredNodes)
        .enter()
        .append("text")
        .text((d) => d.label)
        .attr("font-size", 12)
        .attr("text-anchor", "middle")
        .attr("dy", -NODE_RADIUS - 5);

      // Add tooltips
      node.append("title").text((d) => `${d.type}: ${d.label}`);

      // Update positions on tick
      simulation.on("tick", () => {
        link
          .attr("x1", (d: d3.SimulationLinkDatum<GraphNode>) => (d.source as GraphNode).x!)
          .attr("y1", (d: d3.SimulationLinkDatum<GraphNode>) => (d.source as GraphNode).y!)
          .attr("x2", (d: d3.SimulationLinkDatum<GraphNode>) => (d.target as GraphNode).x!)
          .attr("y2", (d: d3.SimulationLinkDatum<GraphNode>) => (d.target as GraphNode).y!);

        node.attr("cx", (d: GraphNode) => d.x!).attr("cy", (d: GraphNode) => d.y!);

        label.attr("x", (d: GraphNode) => d.x!).attr("y", (d: GraphNode) => d.y!);
      });

      // Apply layout
      if (selectedLayout === "hierarchical") {
        // Simple hierarchical layout
        const agents = filteredNodes.filter((n) => n.type === "agent");
        const beliefs = filteredNodes.filter((n) => n.type === "belief");
        const goals = filteredNodes.filter((n) => n.type === "goal");
        const others = filteredNodes.filter((n) => !["agent", "belief", "goal"].includes(n.type));

        [...agents, ...beliefs, ...goals, ...others].forEach((node, i) => {
          const row =
            node.type === "agent" ? 0 : node.type === "belief" ? 1 : node.type === "goal" ? 2 : 3;
          const col = i % Math.ceil(filteredNodes.length / 4);
          node.x = 100 + col * 150;
          node.y = 100 + row * 150;
        });

        simulation.alpha(0.3).restart();
      } else if (selectedLayout === "circular") {
        // Circular layout
        const radius = Math.min(width, height) / 3;
        filteredNodes.forEach((node, i) => {
          const angle = (i / filteredNodes.length) * 2 * Math.PI;
          node.x = width / 2 + radius * Math.cos(angle);
          node.y = height / 2 + radius * Math.sin(angle);
        });

        simulation.alpha(0.3).restart();
      }

      // Cleanup
      return () => {
        simulation.stop();
      };
    } catch (error) {
      // D3 operations can fail in test environments
      console.warn("D3 visualization error:", error);
    }
  }, [filteredNodes, filteredEdges, selectedLayout]);

  const handleZoom = (direction: "in" | "out" | "reset") => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>();

    if (direction === "reset") {
      svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
    } else {
      const scale = direction === "in" ? 1.2 : 0.8;
      svg.transition().duration(300).call(zoom.scaleBy, scale);
    }
  };

  const handleExport = () => {
    const data = {
      nodes: filteredNodes,
      edges: filteredEdges,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `knowledge-graph-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleFilter = (type: NodeType) => {
    setFilters((prev) => ({ ...prev, [type]: !prev[type] }));
  };

  if (isLoading) {
    return (
      <Card className="h-full flex items-center justify-center">
        <div data-testid="loading-spinner" className="flex items-center gap-2">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span>Loading graph...</span>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="h-full flex items-center justify-center">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error.message}</AlertDescription>
        </Alert>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Knowledge Graph</CardTitle>
            <CardDescription>
              {filteredNodes.length} nodes, {filteredEdges.length} edges
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleExport}
              disabled={filteredNodes.length === 0}
            >
              <Download className="h-4 w-4" />
              Export graph
            </Button>
            <Button variant="outline" size="sm" onClick={clearGraph} disabled={nodes.length === 0}>
              <Trash2 className="h-4 w-4" />
              Clear graph
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Controls */}
        <div className="space-y-4">
          {/* Layout Selection */}
          <div className="flex gap-2">
            <Button
              variant={selectedLayout === "force" ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedLayout("force")}
            >
              <GitBranch className="h-4 w-4 mr-1" />
              Force Layout
            </Button>
            <Button
              variant={selectedLayout === "hierarchical" ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedLayout("hierarchical")}
            >
              <Grid3x3 className="h-4 w-4 mr-1" />
              Hierarchical
            </Button>
            <Button
              variant={selectedLayout === "circular" ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedLayout("circular")}
            >
              <Circle className="h-4 w-4 mr-1" />
              Circular
            </Button>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-4">
            {Object.entries(filters).map(([type, enabled]) => (
              <div key={type} className="flex items-center space-x-2">
                <Switch
                  id={`filter-${type}`}
                  checked={enabled}
                  onCheckedChange={() => toggleFilter(type as NodeType)}
                  aria-label={`Show ${type}s`}
                />
                <Label htmlFor={`filter-${type}`} className="text-sm">
                  <Badge
                    variant="outline"
                    style={{ backgroundColor: enabled ? NODE_COLORS[type as NodeType] : undefined }}
                    className={cn(!enabled && "opacity-50")}
                  >
                    {type}s
                  </Badge>
                </Label>
              </div>
            ))}
          </div>
        </div>

        {/* Graph Container */}
        <div className="flex-1 relative">
          {filteredNodes.length === 0 ? (
            <div className="h-full flex items-center justify-center text-center">
              <div className="text-muted-foreground">
                <p className="text-sm font-medium">No knowledge graph data</p>
                <p className="text-xs mt-1">Interact with agents to build the graph</p>
              </div>
            </div>
          ) : (
            <>
              <div
                ref={containerRef}
                data-testid="graph-container"
                className="absolute inset-0 bg-muted/10 rounded-lg"
              >
                <svg ref={svgRef} className="w-full h-full" />
              </div>

              {/* Zoom Controls */}
              <div className="absolute bottom-4 right-4 flex flex-col gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => handleZoom("in")}
                  aria-label="Zoom in"
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => handleZoom("out")}
                  aria-label="Zoom out"
                >
                  <ZoomOut className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => handleZoom("reset")}
                  aria-label="Reset zoom"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
