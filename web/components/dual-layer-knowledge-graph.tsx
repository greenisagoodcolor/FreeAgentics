"use client";

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Play,
  Pause,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Download,
  Eye,
  EyeOff,
  Settings,
  Layers,
  Filter,
  Search,
} from "lucide-react";
import {
  KnowledgeGraph,
  KnowledgeNode,
  KnowledgeEdge,
  KnowledgeGraphLayer,
  KnowledgeGraphFilters,
} from "@/lib/types";
import { knowledgeGraphApi } from "@/lib/api/knowledge-graph";

// Dual-Layer Knowledge Graph Visualization Component
// Implements ADR inference engine integration, WebSocket communication, and canonical structure

interface DualLayerKnowledgeGraphProps {
  graphId?: string;
  agentIds?: string[];
  width?: number;
  height?: number;
  onNodeClick?: (node: KnowledgeNode) => void;
  onEdgeClick?: (edge: KnowledgeEdge) => void;
  onNodeHover?: (node: KnowledgeNode | null) => void;
  className?: string;
}

interface D3Node extends KnowledgeNode {
  // D3 simulation properties
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
  index?: number;
  // Additional properties for visualization
  radius: number; // Required by KnowledgeNode
  layerId?: string;
  layerType?: string;
  layerOpacity?: number;
  layerColor?: string;
}

interface D3Edge extends Omit<KnowledgeEdge, "source" | "target"> {
  source: D3Node;
  target: D3Node;
  index?: number;
  // Additional properties for visualization
  layerId?: string;
  layerOpacity?: number;
}

interface LayerSettings {
  visible: boolean;
  opacity: number;
  color: string;
  nodeScale: number;
  edgeScale: number;
}

export default function DualLayerKnowledgeGraph({
  graphId,
  agentIds = [],
  width = 800,
  height = 600,
  onNodeClick,
  onEdgeClick,
  onNodeHover,
  className = "",
}: DualLayerKnowledgeGraphProps) {
  // Refs for D3 elements
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<D3Node, D3Edge> | null>(null);

  // State management
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph | null>(
    null,
  );
  const [isSimulationRunning, setIsSimulationRunning] = useState(true);
  const [selectedNode, setSelectedNode] = useState<KnowledgeNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<KnowledgeEdge | null>(null);
  const [hoveredNode, setHoveredNode] = useState<KnowledgeNode | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Layer management state
  const [layerSettings, setLayerSettings] = useState<
    Record<string, LayerSettings>
  >({});
  const [activeLayer, setActiveLayer] = useState<string>("all");

  // Simulation settings
  const [simulationSettings, setSimulationSettings] = useState({
    linkStrength: 0.3,
    chargeStrength: -300,
    centerForce: 0.05,
    collideRadius: 20,
    alphaDecay: 0.01,
    velocityDecay: 0.4,
  });

  // Zoom and pan state
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  // Performance settings
  const [performanceMode, setPerformanceMode] = useState(false);
  const [maxNodes, setMaxNodes] = useState(500);

  // Computed data
  const processedData = useMemo(() => {
    if (!knowledgeGraph) return { nodes: [], edges: [], layers: [] };

    let allNodes: D3Node[] = [];
    let allEdges: D3Edge[] = [];

    // Process each layer
    knowledgeGraph.layers.forEach((layer: any) => {
      const layerSetting = layerSettings[layer.id];
      if (
        !layerSetting?.visible &&
        activeLayer !== "all" &&
        activeLayer !== layer.id
      ) {
        return; // Skip invisible layers
      }

      // Add nodes with layer context
      const layerNodes: D3Node[] = layer.nodes.map((node) => ({
        ...node,
        layerId: layer.id,
        layerType: layer.type,
        layerOpacity: layerSetting?.opacity || 1.0,
        layerColor: layerSetting?.color || layer.color || node.color,
      }));

      // Add edges with layer context
      const layerEdges: D3Edge[] = layer.edges
        .map((edge) => {
          const sourceNode = layerNodes.find((n) => n.id === edge.source);
          const targetNode = layerNodes.find((n) => n.id === edge.target);

          if (!sourceNode || !targetNode) {
            console.warn(`Edge ${edge.id} references missing nodes`);
            return null;
          }

          return {
            ...edge,
            source: sourceNode,
            target: targetNode,
            layerId: layer.id,
            layerOpacity: layerSetting?.opacity || 1.0,
          };
        })
        .filter(Boolean) as D3Edge[];

      allNodes.push(...layerNodes);
      allEdges.push(...layerEdges);
    });

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      allNodes = allNodes.filter(
        (node) =>
          node.title.toLowerCase().includes(query) ||
          node.content?.toLowerCase().includes(query) ||
          node.tags?.some((tag) => tag.toLowerCase().includes(query)),
      );

      // Filter edges to only include those with both nodes visible
      const visibleNodeIds = new Set(allNodes.map((n) => n.id));
      allEdges = allEdges.filter(
        (edge) =>
          visibleNodeIds.has(edge.source.id) &&
          visibleNodeIds.has(edge.target.id),
      );
    }

    // Apply performance limits
    if (performanceMode && allNodes.length > maxNodes) {
      // Sort by importance and take top nodes
      allNodes.sort((a, b) => (b.importance || 0) - (a.importance || 0));
      allNodes = allNodes.slice(0, maxNodes);

      const visibleNodeIds = new Set(allNodes.map((n) => n.id));
      allEdges = allEdges.filter(
        (edge) =>
          visibleNodeIds.has(edge.source.id) &&
          visibleNodeIds.has(edge.target.id),
      );
    }

    return {
      nodes: allNodes,
      edges: allEdges,
      layers: knowledgeGraph.layers,
    };
  }, [
    knowledgeGraph,
    layerSettings,
    activeLayer,
    searchQuery,
    performanceMode,
    maxNodes,
  ]);

  // Initialize layer settings when graph changes
  useEffect(() => {
    if (knowledgeGraph) {
      const newLayerSettings: Record<string, LayerSettings> = {};

      knowledgeGraph.layers.forEach((layer) => {
        newLayerSettings[layer.id] = {
          visible: layer.isVisible,
          opacity: layer.opacity,
          color: layer.color || "#3b82f6",
          nodeScale: 1.0,
          edgeScale: 1.0,
        };
      });

      setLayerSettings(newLayerSettings);
    }
  }, [knowledgeGraph]);

  // Load knowledge graph data
  useEffect(() => {
    const loadKnowledgeGraph = async () => {
      try {
        const response = await knowledgeGraphApi.getKnowledgeGraphs({
          agentId: agentIds[0], // Use first agent for individual graphs
          includeMetadata: true,
          limit: 1,
        });

        if (response.success && response.data && response.data.length > 0) {
          setKnowledgeGraph(response.data[0]);
        } else {
          // Create mock data for demonstration
          const mockGraph: KnowledgeGraph = {
            id: "demo-graph",
            name: "Demo Knowledge Graph",
            description: "Demonstration dual-layer knowledge graph",
            layers: [
              {
                id: "collective-layer",
                name: "Collective Knowledge",
                type: "collective",
                nodes: [
                  {
                    id: "concept-1",
                    title: "Resource Management",
                    type: "concept",
                    content:
                      "Collective understanding of resource allocation strategies",
                    x: 200,
                    y: 200,
                    radius: 20,
                    color: "#3b82f6",
                    ownerType: "collective",
                    confidence: 0.9,
                    importance: 0.8,
                    lastUpdated: new Date(),
                    createdAt: new Date(),
                    tags: ["resources", "strategy", "collective"],
                  },
                  {
                    id: "fact-1",
                    title: "Trading Post Alpha",
                    type: "fact",
                    content: "Verified trading location with high activity",
                    x: 300,
                    y: 150,
                    radius: 15,
                    color: "#10b981",
                    ownerType: "collective",
                    confidence: 0.95,
                    importance: 0.7,
                    lastUpdated: new Date(),
                    createdAt: new Date(),
                    tags: ["trading", "location", "verified"],
                  },
                ],
                edges: [
                  {
                    id: "edge-1",
                    source: "concept-1",
                    target: "fact-1",
                    type: "relates_to",
                    strength: 0.8,
                    confidence: 0.85,
                    color: "#6366f1",
                    createdAt: new Date(),
                    lastUpdated: new Date(),
                  },
                ],
                isVisible: true,
                opacity: 1.0,
                color: "#3b82f6",
              },
              {
                id: "individual-layer",
                name: "Individual Beliefs",
                type: "individual",
                agentId: agentIds[0] || "agent-1",
                nodes: [
                  {
                    id: "belief-1",
                    title: "Market Opportunity",
                    type: "belief",
                    content: "Personal belief about emerging market trends",
                    x: 250,
                    y: 300,
                    radius: 12,
                    color: "#f59e0b",
                    agentId: agentIds[0] || "agent-1",
                    ownerType: "individual",
                    confidence: 0.75,
                    importance: 0.6,
                    lastUpdated: new Date(),
                    createdAt: new Date(),
                    tags: ["market", "opportunity", "personal"],
                  },
                ],
                edges: [],
                isVisible: true,
                opacity: 0.8,
                color: "#f59e0b",
              },
            ],
            createdAt: new Date(),
            lastUpdated: new Date(),
            version: "1.0.0",
            layout: "force-directed",
            renderer: "d3",
            maxNodes: 1000,
            lodEnabled: true,
            clusteringEnabled: false,
            filters: {
              nodeTypes: ["concept", "fact", "belief"],
              confidenceRange: [0.0, 1.0],
              importanceRange: [0.0, 1.0],
              agentIds: agentIds,
              tags: [],
              edgeTypes: ["relates_to", "supports", "contradicts"],
              strengthRange: [0.0, 1.0],
              showOnlyConnected: false,
              hideIsolatedNodes: false,
            },
            selectedNodes: [],
            selectedEdges: [],
            zoom: 1.0,
            pan: { x: 0, y: 0 },
          };

          setKnowledgeGraph(mockGraph);
        }
      } catch (error) {
        console.error("Failed to load knowledge graph:", error);
      }
    };

    loadKnowledgeGraph();
  }, [graphId, agentIds]);

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current || !processedData.nodes.length) return;

    const svg = d3.select(svgRef.current);
    const container = svg.select(".graph-container");

    // Clear existing content
    container.selectAll("*").remove();

    // Create groups for different elements
    const edgeGroup = container.append("g").attr("class", "edges");
    const nodeGroup = container.append("g").attr("class", "nodes");
    const labelGroup = container.append("g").attr("class", "labels");

    // Initialize force simulation
    const simulation = d3
      .forceSimulation<D3Node>(processedData.nodes)
      .force(
        "link",
        d3
          .forceLink<D3Node, D3Edge>(processedData.edges)
          .id((d) => d.id)
          .strength(simulationSettings.linkStrength),
      )
      .force(
        "charge",
        d3.forceManyBody().strength(simulationSettings.chargeStrength),
      )
      .force(
        "center",
        d3
          .forceCenter(width / 2, height / 2)
          .strength(simulationSettings.centerForce),
      )
      .force(
        "collision",
        d3
          .forceCollide<D3Node>()
          .radius((d) => (d.radius || 10) + simulationSettings.collideRadius),
      )
      .alphaDecay(simulationSettings.alphaDecay)
      .velocityDecay(simulationSettings.velocityDecay);

    simulationRef.current = simulation;

    // Create edges
    const edges = edgeGroup
      .selectAll(".edge")
      .data(processedData.edges)
      .enter()
      .append("line")
      .attr("class", "edge")
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", (d) => Math.max(1, (d.strength || 0.5) * 3))
      .attr("stroke-opacity", (d) => (d.layerOpacity || 1) * 0.6)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        const edge: KnowledgeEdge = {
          ...d,
          source: typeof d.source === "object" ? d.source.id : d.source,
          target: typeof d.target === "object" ? d.target.id : d.target,
        };
        setSelectedEdge(edge);
        onEdgeClick?.(edge);
      });

    // Create nodes
    const nodes = nodeGroup
      .selectAll(".node")
      .data(processedData.nodes)
      .enter()
      .append("circle")
      .attr("class", "node")
      .attr("r", (d) => d.radius || 10)
      .attr("fill", (d) => d.layerColor || d.color)
      .attr("fill-opacity", (d) => d.layerOpacity || 1)
      .attr("stroke", (d) => (selectedNode?.id === d.id ? "#000" : "none"))
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
        onNodeClick?.(d);
      })
      .on("mouseenter", (event, d) => {
        setHoveredNode(d);
        onNodeHover?.(d);
      })
      .on("mouseleave", () => {
        setHoveredNode(null);
        onNodeHover?.(null);
      });

    // Add drag behavior
    const drag = d3
      .drag<SVGCircleElement, D3Node>()
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

    nodes.call(drag);

    // Create labels
    const labels = labelGroup
      .selectAll(".label")
      .data(processedData.nodes)
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("text-anchor", "middle")
      .attr("dy", ".35em")
      .attr("font-size", "12px")
      .attr("fill", "#333")
      .attr("pointer-events", "none")
      .text((d) =>
        d.title.length > 15 ? d.title.slice(0, 15) + "..." : d.title,
      );

    // Update positions on simulation tick
    simulation.on("tick", () => {
      edges
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      nodes.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

      labels
        .attr("x", (d) => d.x)
        .attr("y", (d) => d.y + (d.radius || 10) + 15);
    });

    // Stop simulation if not running
    if (!isSimulationRunning) {
      simulation.stop();
    }

    return () => {
      simulation.stop();
    };
  }, [
    processedData,
    simulationSettings,
    width,
    height,
    selectedNode,
    isSimulationRunning,
    onNodeClick,
    onEdgeClick,
    onNodeHover,
  ]);

  // Setup zoom behavior
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = svg.select(".graph-container");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => {
        const { transform } = event;
        setTransform(transform);
        container.attr("transform", transform);
      });

    svg.call(zoom);

    return () => {
      svg.on(".zoom", null);
    };
  }, []);

  // WebSocket integration for real-time updates
  useEffect(() => {
    if (!graphId) return;

    const connectWebSocket = async () => {
      try {
        await knowledgeGraphApi.connectWebSocket(graphId);

        // Subscribe to updates
        knowledgeGraphApi.subscribe("node_added", (update) => {
          console.log("Node added:", update);
          // Handle node addition
        });

        knowledgeGraphApi.subscribe("node_updated", (update) => {
          console.log("Node updated:", update);
          // Handle node update
        });

        knowledgeGraphApi.subscribe("edge_added", (update) => {
          console.log("Edge added:", update);
          // Handle edge addition
        });
      } catch (error) {
        console.error("Failed to connect WebSocket:", error);
      }
    };

    connectWebSocket();

    return () => {
      knowledgeGraphApi.disconnectWebSocket();
    };
  }, [graphId]);

  // Control functions
  const toggleSimulation = useCallback(() => {
    if (simulationRef.current) {
      if (isSimulationRunning) {
        simulationRef.current.stop();
      } else {
        simulationRef.current.restart();
      }
      setIsSimulationRunning(!isSimulationRunning);
    }
  }, [isSimulationRunning]);

  const resetSimulation = useCallback(() => {
    if (simulationRef.current) {
      simulationRef.current.alpha(1).restart();
      setIsSimulationRunning(true);
    }
  }, []);

  const zoomIn = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg
        .transition()
        .call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1.5);
    }
  }, []);

  const zoomOut = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg
        .transition()
        .call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1 / 1.5);
    }
  }, []);

  const exportGraph = useCallback(async () => {
    if (!knowledgeGraph) return;

    try {
      const exportConfig = {
        format: "svg" as const,
        includeMetadata: true,
        includeFilters: false,
        includeAllLayers: true,
        includeAllElements: true,
        includeLabels: true,
      };

      const response = await knowledgeGraphApi.exportKnowledgeGraph(
        knowledgeGraph.id,
        exportConfig,
      );

      if (response.success && response.data) {
        const url = URL.createObjectURL(response.data);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${knowledgeGraph.name}.svg`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error("Failed to export graph:", error);
    }
  }, [knowledgeGraph]);

  const toggleLayerVisibility = useCallback((layerId: string) => {
    setLayerSettings((prev) => ({
      ...prev,
      [layerId]: {
        ...prev[layerId],
        visible: !prev[layerId]?.visible,
      },
    }));
  }, []);

  const updateLayerOpacity = useCallback((layerId: string, opacity: number) => {
    setLayerSettings((prev) => ({
      ...prev,
      [layerId]: {
        ...prev[layerId],
        opacity,
      },
    }));
  }, []);

  return (
    <div
      className={`dual-layer-knowledge-graph ${className}`}
      ref={containerRef}
    >
      <Card className="w-full">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold">
              {knowledgeGraph?.name || "Knowledge Graph"}
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="outline">
                {processedData.nodes.length} nodes
              </Badge>
              <Badge variant="outline">
                {processedData.edges.length} edges
              </Badge>
              <Badge variant="outline">
                {processedData.layers.length} layers
              </Badge>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={toggleSimulation}>
                {isSimulationRunning ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
              <Button variant="outline" size="sm" onClick={resetSimulation}>
                <RotateCcw className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={zoomIn}>
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={zoomOut}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={exportGraph}>
                <Download className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowFilters(!showFilters)}
              >
                <Filter className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Search */}
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 px-3 py-1 text-sm border rounded-md"
            />
          </div>
        </CardHeader>

        <CardContent className="p-0">
          <div className="flex">
            {/* Main visualization */}
            <div className="flex-1">
              <svg
                ref={svgRef}
                width={width}
                height={height}
                className="border rounded-lg"
              >
                <g className="graph-container" />
              </svg>
            </div>

            {/* Side panels */}
            <div className="w-80 border-l">
              <Tabs defaultValue="layers" className="h-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="layers">Layers</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                  <TabsTrigger value="details">Details</TabsTrigger>
                </TabsList>

                <TabsContent value="layers" className="p-4 space-y-4">
                  <div className="space-y-3">
                    {processedData.layers.map((layer) => (
                      <div key={layer.id} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => toggleLayerVisibility(layer.id)}
                            >
                              {layerSettings[layer.id]?.visible ? (
                                <Eye className="h-4 w-4" />
                              ) : (
                                <EyeOff className="h-4 w-4" />
                              )}
                            </Button>
                            <span className="font-medium">{layer.name}</span>
                          </div>
                          <Badge variant="secondary">{layer.type}</Badge>
                        </div>

                        <div className="ml-6 space-y-2">
                          <div className="flex items-center gap-2">
                            <Label className="text-xs">Opacity</Label>
                            <Slider
                              value={[layerSettings[layer.id]?.opacity || 1]}
                              onValueChange={([value]) =>
                                updateLayerOpacity(layer.id, value)
                              }
                              max={1}
                              min={0}
                              step={0.1}
                              className="flex-1"
                            />
                          </div>

                          <div className="text-xs text-muted-foreground">
                            {layer.nodes.length} nodes, {layer.edges.length}{" "}
                            edges
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="settings" className="p-4 space-y-4">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Simulation</Label>

                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <Label className="text-xs">Link Strength</Label>
                          <Slider
                            value={[simulationSettings.linkStrength]}
                            onValueChange={([value]) =>
                              setSimulationSettings((prev) => ({
                                ...prev,
                                linkStrength: value,
                              }))
                            }
                            max={1}
                            min={0}
                            step={0.1}
                            className="w-24"
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <Label className="text-xs">Charge Strength</Label>
                          <Slider
                            value={[
                              Math.abs(simulationSettings.chargeStrength),
                            ]}
                            onValueChange={([value]) =>
                              setSimulationSettings((prev) => ({
                                ...prev,
                                chargeStrength: -value,
                              }))
                            }
                            max={1000}
                            min={0}
                            step={50}
                            className="w-24"
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <Label className="text-xs">Collision Radius</Label>
                          <Slider
                            value={[simulationSettings.collideRadius]}
                            onValueChange={([value]) =>
                              setSimulationSettings((prev) => ({
                                ...prev,
                                collideRadius: value,
                              }))
                            }
                            max={50}
                            min={0}
                            step={5}
                            className="w-24"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Performance</Label>

                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Performance Mode</Label>
                        <Switch
                          checked={performanceMode}
                          onCheckedChange={setPerformanceMode}
                        />
                      </div>

                      {performanceMode && (
                        <div className="flex items-center justify-between">
                          <Label className="text-xs">Max Nodes</Label>
                          <Slider
                            value={[maxNodes]}
                            onValueChange={([value]) => setMaxNodes(value)}
                            max={1000}
                            min={50}
                            step={50}
                            className="w-24"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="details" className="p-4 space-y-4">
                  {selectedNode ? (
                    <div className="space-y-3">
                      <h3 className="font-medium">{selectedNode.title}</h3>
                      <div className="space-y-2 text-sm">
                        <div>
                          <strong>Type:</strong> {selectedNode.type}
                        </div>
                        <div>
                          <strong>Confidence:</strong>{" "}
                          {(selectedNode.confidence * 100).toFixed(1)}%
                        </div>
                        <div>
                          <strong>Importance:</strong>{" "}
                          {(selectedNode.importance * 100).toFixed(1)}%
                        </div>
                        {selectedNode.content && (
                          <div>
                            <strong>Content:</strong> {selectedNode.content}
                          </div>
                        )}
                        {selectedNode.tags && selectedNode.tags.length > 0 && (
                          <div>
                            <strong>Tags:</strong>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {selectedNode.tags.map((tag) => (
                                <Badge
                                  key={tag}
                                  variant="outline"
                                  className="text-xs"
                                >
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : hoveredNode ? (
                    <div className="space-y-2">
                      <h3 className="font-medium">{hoveredNode.title}</h3>
                      <div className="text-sm text-muted-foreground">
                        {hoveredNode.type} •{" "}
                        {(hoveredNode.confidence * 100).toFixed(1)}% confidence
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      Click or hover on a node to see details
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
