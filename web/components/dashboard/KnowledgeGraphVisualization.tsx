"use client";

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import { motion } from "framer-motion";
import { useAppSelector } from "@/store/hooks";
import { KnowledgeNode, KnowledgeEdge } from "@/store/slices/knowledgeSlice";
import { ZoomIn, ZoomOut, RotateCcw, Download, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: "belief" | "fact" | "hypothesis";
  confidence: number;
  agents: string[];
  radius: number;
  color: string;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  id: string;
  type: "supports" | "contradicts" | "related";
  strength: number;
  strokeWidth: number;
  strokeDasharray?: string;
}

interface KnowledgeGraphVisualizationProps {
  testMode?: boolean;
}

const KnowledgeGraphVisualization: React.FC<
  KnowledgeGraphVisualizationProps
> = ({ testMode = false }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(
    null,
  );

  // Use fixed dimensions in test mode to prevent layout shifts
  // Responsive dimensions for different viewports in test mode
  const getTestModeDimensions = () => {
    if (!testMode) return { width: 800, height: 600 };

    // Check if we're in a mobile viewport (rough approximation)
    if (typeof window !== "undefined" && window.innerWidth < 768) {
      return { width: 350, height: 250 }; // Mobile test dimensions
    }
    return { width: 1280, height: 960 }; // Desktop test dimensions
  };

  const [dimensions, setDimensions] = useState(getTestModeDimensions());
  const [zoom, setZoom] = useState(1);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);
  const [showControls, setShowControls] = useState(false);

  // Redux state
  const knowledgeGraph = useAppSelector((state) => state.knowledge.graph);
  const agents = useAppSelector((state) => state.agents.agents);
  const selectedAgentId = useAppSelector(
    (state) => state.agents.selectedAgentId,
  );

  // Color schemes (memoized to prevent unnecessary re-renders)
  const typeColors = useMemo(
    () => ({
      belief: "#4F46E5",
      fact: "#10B981",
      hypothesis: "#F59E0B",
    }),
    [],
  );

  const edgeStyles = useMemo(
    () => ({
      supports: { strokeDasharray: "none", opacity: 0.8 },
      contradicts: { strokeDasharray: "5,5", opacity: 0.6 },
      related: { strokeDasharray: "2,3", opacity: 0.4 },
    }),
    [],
  );

  // Process data for D3
  const processGraphData = useCallback(() => {
    const nodes: GraphNode[] = Object.values(knowledgeGraph.nodes)
      .filter((node) => node.confidence >= confidenceThreshold)
      .map((node, index) => ({
        ...node,
        radius: Math.sqrt(node.agents.length) * 8 + 12,
        color: typeColors[node.type],
        // Use fixed positions in test mode to prevent movement
        x: testMode
          ? 200 + (index % 5) * 200
          : node.position?.x || Math.random() * dimensions.width,
        y: testMode
          ? 200 + Math.floor(index / 5) * 150
          : node.position?.y || Math.random() * dimensions.height,
      }));

    const links: GraphLink[] = Object.values(knowledgeGraph.edges)
      .filter(
        (edge) =>
          nodes.find((n) => n.id === edge.source) &&
          nodes.find((n) => n.id === edge.target),
      )
      .map((edge) => ({
        ...edge,
        source: edge.source,
        target: edge.target,
        strokeWidth: edge.strength * 4 + 1,
        strokeDasharray: edgeStyles[edge.type].strokeDasharray,
      }));

    return { nodes, links };
  }, [
    knowledgeGraph,
    confidenceThreshold,
    dimensions,
    typeColors,
    edgeStyles,
    testMode,
  ]);

  // Initialize and update D3 simulation
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const { nodes, links } = processGraphData();

    // Clear previous content
    svg.selectAll("*").remove();

    // Create main group for zoom/pan
    const g = svg.append("g").attr("class", "main-group");

    // Initialize simulation - disable in test mode for stability
    const simulation = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance(100),
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force(
        "center",
        d3.forceCenter(dimensions.width / 2, dimensions.height / 2),
      )
      .force(
        "collision",
        d3.forceCollide<GraphNode>().radius((d) => d.radius + 5),
      );

    // Stop simulation immediately in test mode for stable positioning
    if (testMode) {
      simulation.stop();
    }

    simulationRef.current = simulation;

    // Create edges
    const linkSelection = g
      .selectAll(".link")
      .data(links)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", "#666")
      .attr("stroke-width", (d) => d.strokeWidth)
      .attr("stroke-dasharray", (d) => d.strokeDasharray || "none")
      .attr("opacity", (d) => edgeStyles[d.type].opacity);

    // Create nodes
    const nodeSelection = g
      .selectAll(".node")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .style("cursor", "pointer");

    // Node circles
    nodeSelection
      .append("circle")
      .attr("r", (d) => d.radius)
      .attr("fill", (d) => {
        if (selectedAgentId && !d.agents.includes(selectedAgentId)) {
          return d.color + "30"; // Faded if agent not selected
        }
        return d.color;
      })
      .attr("stroke", (d) => (selectedNode?.id === d.id ? "#fff" : "none"))
      .attr("stroke-width", 3)
      .style("filter", (d) =>
        selectedNode?.id === d.id
          ? "drop-shadow(0 0 10px rgba(255,255,255,0.8))"
          : "none",
      );

    // Node labels
    nodeSelection
      .append("text")
      .text((d) =>
        d.label.length > 20 ? d.label.slice(0, 17) + "..." : d.label,
      )
      .attr("text-anchor", "middle")
      .attr("dy", (d) => d.radius + 15)
      .attr("fill", "#fff")
      .attr("font-size", "12px")
      .attr("font-family", "Inter, system-ui");

    // Confidence indicators
    nodeSelection
      .append("circle")
      .attr("r", 4)
      .attr("cx", (d) => d.radius - 6)
      .attr("cy", (d) => -d.radius + 6)
      .attr("fill", (d) => {
        if (d.confidence > 0.7) return "#10B981";
        if (d.confidence > 0.4) return "#F59E0B";
        return "#EF4444";
      })
      .attr("stroke", "#000")
      .attr("stroke-width", 1);

    // Agent count badges
    nodeSelection
      .append("circle")
      .attr("r", 8)
      .attr("cx", (d) => -d.radius + 8)
      .attr("cy", (d) => -d.radius + 8)
      .attr("fill", "#4F46E5")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);

    nodeSelection
      .append("text")
      .text((d) => d.agents.length)
      .attr("x", (d) => -d.radius + 8)
      .attr("y", (d) => -d.radius + 8)
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .attr("fill", "#fff")
      .attr("font-size", "10px")
      .attr("font-weight", "bold");

    // Node interactions - disable in test mode
    if (!testMode) {
      nodeSelection
        .on("click", (event, d) => {
          setSelectedNode(d);
          event.stopPropagation();
        })
        .on("mouseover", (event, d) => {
          // Highlight connected nodes
          const connectedNodeIds = new Set();
          links.forEach((link) => {
            if (
              link.source === d ||
              (typeof link.source === "object" && link.source.id === d.id)
            ) {
              connectedNodeIds.add(
                typeof link.target === "object" ? link.target.id : link.target,
              );
            }
            if (
              link.target === d ||
              (typeof link.target === "object" && link.target.id === d.id)
            ) {
              connectedNodeIds.add(
                typeof link.source === "object" ? link.source.id : link.source,
              );
            }
          });

          nodeSelection.style("opacity", (n) =>
            n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3,
          );

          linkSelection.style("opacity", (l) => {
            const sourceId =
              typeof l.source === "object" ? l.source.id : l.source;
            const targetId =
              typeof l.target === "object" ? l.target.id : l.target;
            return sourceId === d.id || targetId === d.id ? 1 : 0.1;
          });
        })
        .on("mouseout", () => {
          nodeSelection.style("opacity", 1);
          linkSelection.style("opacity", (d) => edgeStyles[d.type].opacity);
        });

      // Drag behavior - disable in test mode
      const drag = d3
        .drag<SVGGElement, GraphNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      nodeSelection.call(drag);
    }

    // Zoom behavior - disable in test mode
    if (!testMode) {
      const zoomBehavior = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 10])
        .on("zoom", (event) => {
          g.attr("transform", event.transform);
          setZoom(event.transform.k);
        });

      svg.call(zoomBehavior);

      // Click to deselect
      svg.on("click", () => setSelectedNode(null));
    }

    // Update positions on simulation tick - disable in test mode
    if (!testMode) {
      simulation.on("tick", () => {
        linkSelection
          .attr("x1", (d) => (d.source as GraphNode).x!)
          .attr("y1", (d) => (d.source as GraphNode).y!)
          .attr("x2", (d) => (d.target as GraphNode).x!)
          .attr("y2", (d) => (d.target as GraphNode).y!);

        nodeSelection.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });
    } else {
      // In test mode, set positions immediately without animation
      linkSelection
        .attr("x1", (d) => (d.source as GraphNode).x!)
        .attr("y1", (d) => (d.source as GraphNode).y!)
        .attr("x2", (d) => (d.target as GraphNode).x!)
        .attr("y2", (d) => (d.target as GraphNode).y!);

      nodeSelection.attr("transform", (d) => `translate(${d.x},${d.y})`);
    }

    return () => {
      simulation.stop();
    };
  }, [
    processGraphData,
    dimensions,
    selectedNode,
    selectedAgentId,
    edgeStyles,
    testMode,
  ]);

  // Handle container resize - disable in test mode
  useEffect(() => {
    if (testMode) return; // Skip resize handling in test mode

    const handleResize = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width: width - 40, height: height - 40 });
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [testMode]);

  // Control functions
  const handleZoomIn = () => {
    if (svgRef.current && !testMode) {
      d3.select(svgRef.current)
        .transition()
        .call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1.5);
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current && !testMode) {
      d3.select(svgRef.current)
        .transition()
        .call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 0.67);
    }
  };

  const handleReset = () => {
    if (svgRef.current && simulationRef.current && !testMode) {
      d3.select(svgRef.current)
        .transition()
        .call(
          d3.zoom<SVGSVGElement, unknown>().transform as any,
          d3.zoomIdentity,
        );
      simulationRef.current.alpha(1).restart();
    }
  };

  const handleExport = () => {
    if (svgRef.current) {
      const svgData = new XMLSerializer().serializeToString(svgRef.current);
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d")!;
      const img = new Image();

      canvas.width = dimensions.width;
      canvas.height = dimensions.height;

      img.onload = () => {
        ctx.fillStyle = "#0A0A0B";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        const link = document.createElement("a");
        link.download = "knowledge-graph.png";
        link.href = canvas.toDataURL();
        link.click();
      };

      img.src = "data:image/svg+xml;base64," + btoa(svgData);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
        <div>
          <h3 className="font-ui text-lg font-semibold text-[var(--text-primary)]">
            Knowledge Graph
          </h3>
          <p className="text-sm text-[var(--text-secondary)]">
            {Object.keys(knowledgeGraph.nodes).length} nodes,{" "}
            {Object.keys(knowledgeGraph.edges).length} edges
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowControls(!showControls)}
            className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
          >
            <Settings className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleZoomIn}
            className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
          >
            <ZoomIn className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleZoomOut}
            className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
          >
            <ZoomOut className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
          >
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Main Graph Area */}
        <div
          ref={containerRef}
          className="flex-1 relative knowledge-graph-container"
        >
          <svg
            ref={svgRef}
            width={dimensions.width}
            height={dimensions.height}
            className="knowledge-graph-svg"
            data-testid="knowledge-graph-svg"
            style={{
              display: "block",
              visibility: "visible",
              opacity: 1,
              zIndex: 1,
            }}
          />

          {/* Zoom indicator */}
          <div className="absolute bottom-4 left-4 bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded px-2 py-1">
            <span className="text-xs font-mono text-[var(--text-secondary)]">
              {(zoom * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Controls Panel - use regular div in test mode instead of motion.div */}
        {showControls &&
          (testMode ? (
            <div className="w-[300px] bg-[var(--bg-secondary)] border-l border-[var(--bg-tertiary)] p-4 space-y-4">
              <div>
                <Label className="text-sm text-[var(--text-primary)]">
                  Confidence Threshold: {confidenceThreshold.toFixed(2)}
                </Label>
                <Slider
                  value={[confidenceThreshold]}
                  onValueChange={([value]) => setConfidenceThreshold(value)}
                  max={1}
                  min={0}
                  step={0.05}
                  className="mt-2"
                />
              </div>

              {/* Legend */}
              <div>
                <Label className="text-sm text-[var(--text-primary)] mb-2 block">
                  Node Types
                </Label>
                <div className="space-y-2">
                  {Object.entries(typeColors).map(([type, color]) => (
                    <div key={type} className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-xs text-[var(--text-secondary)] capitalize">
                        {type}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Edge Types */}
              <div>
                <Label className="text-sm text-[var(--text-primary)] mb-2 block">
                  Relationships
                </Label>
                <div className="space-y-2">
                  {Object.entries(edgeStyles).map(([type, style]) => (
                    <div key={type} className="flex items-center gap-2">
                      <svg width="20" height="2">
                        <line
                          x1="0"
                          y1="1"
                          x2="20"
                          y2="1"
                          stroke="#666"
                          strokeWidth="2"
                          strokeDasharray={style.strokeDasharray}
                        />
                      </svg>
                      <span className="text-xs text-[var(--text-secondary)] capitalize">
                        {type}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 300, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              className="bg-[var(--bg-secondary)] border-l border-[var(--bg-tertiary)] p-4 space-y-4"
            >
              <div>
                <Label className="text-sm text-[var(--text-primary)]">
                  Confidence Threshold: {confidenceThreshold.toFixed(2)}
                </Label>
                <Slider
                  value={[confidenceThreshold]}
                  onValueChange={([value]) => setConfidenceThreshold(value)}
                  max={1}
                  min={0}
                  step={0.05}
                  className="mt-2"
                />
              </div>

              {/* Legend */}
              <div>
                <Label className="text-sm text-[var(--text-primary)] mb-2 block">
                  Node Types
                </Label>
                <div className="space-y-2">
                  {Object.entries(typeColors).map(([type, color]) => (
                    <div key={type} className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-xs text-[var(--text-secondary)] capitalize">
                        {type}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Edge Types */}
              <div>
                <Label className="text-sm text-[var(--text-primary)] mb-2 block">
                  Relationships
                </Label>
                <div className="space-y-2">
                  {Object.entries(edgeStyles).map(([type, style]) => (
                    <div key={type} className="flex items-center gap-2">
                      <svg width="20" height="2">
                        <line
                          x1="0"
                          y1="1"
                          x2="20"
                          y2="1"
                          stroke="#666"
                          strokeWidth="2"
                          strokeDasharray={style.strokeDasharray}
                        />
                      </svg>
                      <span className="text-xs text-[var(--text-secondary)] capitalize">
                        {type}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
      </div>

      {/* Selected Node Details - use regular div in test mode */}
      {selectedNode &&
        (testMode ? (
          <div className="border-t border-[var(--bg-tertiary)] bg-[var(--bg-secondary)] p-4">
            <Card className="bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)]">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-[var(--text-primary)]">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: selectedNode.color }}
                  />
                  {selectedNode.label}
                  <Badge
                    variant="secondary"
                    style={{
                      backgroundColor: selectedNode.color + "20",
                      color: selectedNode.color,
                    }}
                  >
                    {selectedNode.type}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-[var(--text-secondary)]">
                    Confidence:{" "}
                    <span className="font-mono">
                      {selectedNode.confidence.toFixed(3)}
                    </span>
                  </span>
                  <span className="text-[var(--text-secondary)]">
                    Agents:{" "}
                    <span className="font-mono">
                      {selectedNode.agents.length}
                    </span>
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.agents.map((agentId) => (
                    <Badge
                      key={agentId}
                      variant="outline"
                      className="text-xs bg-[var(--bg-secondary)] border-[var(--bg-secondary)] text-[var(--text-secondary)]"
                    >
                      {agents[agentId]?.name || agentId}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-[var(--bg-tertiary)] bg-[var(--bg-secondary)] p-4"
          >
            <Card className="bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)]">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-[var(--text-primary)]">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: selectedNode.color }}
                  />
                  {selectedNode.label}
                  <Badge
                    variant="secondary"
                    style={{
                      backgroundColor: selectedNode.color + "20",
                      color: selectedNode.color,
                    }}
                  >
                    {selectedNode.type}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-[var(--text-secondary)]">
                    Confidence:{" "}
                    <span className="font-mono">
                      {selectedNode.confidence.toFixed(3)}
                    </span>
                  </span>
                  <span className="text-[var(--text-secondary)]">
                    Agents:{" "}
                    <span className="font-mono">
                      {selectedNode.agents.length}
                    </span>
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.agents.map((agentId) => (
                    <Badge
                      key={agentId}
                      variant="outline"
                      className="text-xs bg-[var(--bg-secondary)] border-[var(--bg-secondary)] text-[var(--text-secondary)]"
                    >
                      {agents[agentId]?.name || agentId}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
    </div>
  );
};

export default KnowledgeGraphVisualization;
