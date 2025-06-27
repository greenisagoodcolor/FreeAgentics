import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

// Knowledge Graph API endpoint for dual-layer visualization
// Implements ADR-008 WebSocket Communication patterns for REST endpoints

const KnowledgeNodeSchema = z.object({
  id: z.string(),
  title: z.string(),
  type: z.enum([
    "concept",
    "fact",
    "belief",
    "agent",
    "entity",
    "relationship",
    "pattern",
  ]),
  content: z.string().optional(),
  x: z.number(),
  y: z.number(),
  radius: z.number(),
  color: z.string(),
  agentId: z.string().optional(),
  agentIds: z.array(z.string()).optional(),
  ownerType: z.enum(["individual", "collective", "shared"]),
  confidence: z.number().min(0).max(1),
  importance: z.number().min(0).max(1),
  lastUpdated: z.string(),
  createdAt: z.string(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional(),
});

const KnowledgeEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  type: z.enum([
    "supports",
    "contradicts",
    "relates_to",
    "causes",
    "prevents",
    "similar_to",
    "derived_from",
    "contains",
    "depends_on",
  ]),
  strength: z.number().min(0).max(1),
  confidence: z.number().min(0).max(1),
  color: z.string(),
  agentId: z.string().optional(),
  agentIds: z.array(z.string()).optional(),
  createdAt: z.string(),
  lastUpdated: z.string(),
  metadata: z.record(z.any()).optional(),
});

const KnowledgeGraphLayerSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.enum(["individual", "collective"]),
  agentId: z.string().optional(),
  nodes: z.array(KnowledgeNodeSchema),
  edges: z.array(KnowledgeEdgeSchema),
  isVisible: z.boolean(),
  opacity: z.number().min(0).max(1),
  color: z.string().optional(),
});

const KnowledgeGraphFiltersSchema = z.object({
  nodeTypes: z.array(z.string()),
  confidenceRange: z.tuple([z.number(), z.number()]),
  importanceRange: z.tuple([z.number(), z.number()]),
  timeRange: z.tuple([z.string(), z.string()]).optional(),
  agentIds: z.array(z.string()),
  tags: z.array(z.string()),
  edgeTypes: z.array(z.string()),
  strengthRange: z.tuple([z.number(), z.number()]),
  searchQuery: z.string().optional(),
  showOnlyConnected: z.boolean(),
  hideIsolatedNodes: z.boolean(),
  maxConnections: z.number().optional(),
});

const KnowledgeGraphSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  layers: z.array(KnowledgeGraphLayerSchema),
  createdAt: z.string(),
  lastUpdated: z.string(),
  version: z.string(),
  layout: z.enum(["force-directed", "hierarchical", "circular", "grid"]),
  renderer: z.enum(["d3", "threejs", "auto"]),
  maxNodes: z.number(),
  lodEnabled: z.boolean(),
  clusteringEnabled: z.boolean(),
  filters: KnowledgeGraphFiltersSchema,
  selectedNodes: z.array(z.string()),
  selectedEdges: z.array(z.string()),
  zoom: z.number(),
  pan: z.object({ x: z.number(), y: z.number() }),
  metadata: z.record(z.any()).optional(),
});

// GET /api/knowledge - Get knowledge graphs with filtering
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Parse query parameters
    const agentId = searchParams.get("agentId");
    const layerType = searchParams.get("layerType") as
      | "individual"
      | "collective"
      | null;
    const includeMetadata = searchParams.get("includeMetadata") === "true";
    const limit = parseInt(searchParams.get("limit") || "10");
    const offset = parseInt(searchParams.get("offset") || "0");

    // Mock data for now - would integrate with actual knowledge systems
    const mockKnowledgeGraphs: any[] = [
      {
        id: "kg-collective-001",
        name: "Collective Knowledge Graph",
        description: "Shared knowledge across all agents",
        layers: [
          {
            id: "layer-collective",
            name: "Collective Knowledge",
            type: "collective" as const,
            nodes: [
              {
                id: "concept-1",
                title: "Resource Location",
                type: "concept" as const,
                content:
                  "Knowledge about resource locations in the environment",
                x: 100,
                y: 100,
                radius: 15,
                color: "#3b82f6",
                ownerType: "collective" as const,
                confidence: 0.85,
                importance: 0.9,
                lastUpdated: new Date().toISOString(),
                createdAt: new Date(Date.now() - 86400000).toISOString(),
                tags: ["resources", "location", "environment"],
                metadata: { verified: true, source: "multiple_agents" },
              },
              {
                id: "fact-1",
                title: "Trading Post Alpha",
                type: "fact" as const,
                content: "Trading post located at coordinates (50, 75)",
                x: 200,
                y: 150,
                radius: 12,
                color: "#10b981",
                ownerType: "collective" as const,
                confidence: 0.95,
                importance: 0.8,
                lastUpdated: new Date().toISOString(),
                createdAt: new Date(Date.now() - 172800000).toISOString(),
                tags: ["trading", "location", "commerce"],
                metadata: { verified: true, coordinates: [50, 75] },
              },
            ],
            edges: [
              {
                id: "edge-1",
                source: "concept-1",
                target: "fact-1",
                type: "contains" as const,
                strength: 0.8,
                confidence: 0.9,
                color: "#6366f1",
                createdAt: new Date().toISOString(),
                lastUpdated: new Date().toISOString(),
                metadata: { relationship_strength: "strong" },
              },
            ],
            isVisible: true,
            opacity: 1.0,
            color: "#3b82f6",
          },
        ],
        createdAt: new Date(Date.now() - 604800000).toISOString(),
        lastUpdated: new Date().toISOString(),
        version: "1.0.0",
        layout: "force-directed" as const,
        renderer: "auto" as const,
        maxNodes: 1000,
        lodEnabled: true,
        clusteringEnabled: true,
        filters: {
          nodeTypes: ["concept", "fact", "belief"],
          confidenceRange: [0.0, 1.0] as [number, number],
          importanceRange: [0.0, 1.0] as [number, number],
          agentIds: [],
          tags: [],
          edgeTypes: ["supports", "relates_to", "contains"],
          strengthRange: [0.0, 1.0] as [number, number],
          showOnlyConnected: true,
          hideIsolatedNodes: false,
        },
        selectedNodes: [],
        selectedEdges: [],
        zoom: 1.0,
        pan: { x: 0, y: 0 },
        metadata: {
          totalInteractions: 156,
          lastAccessed: new Date().toISOString(),
        },
      },
    ];

    // Add individual agent knowledge graphs if agentId specified
    if (agentId) {
      const individualGraph = {
        id: `kg-individual-${agentId}`,
        name: `Agent ${agentId} Knowledge`,
        description: `Individual knowledge graph for agent ${agentId}`,
        layers: [
          {
            id: `layer-individual-${agentId}`,
            name: `Agent ${agentId} Knowledge`,
            type: "individual" as const,
            agentId: agentId,
            nodes: [
              {
                id: `belief-${agentId}-1`,
                title: "Market Opportunity",
                type: "belief" as const,
                content:
                  "Believes there is a trading opportunity in the northern sector",
                x: 150,
                y: 200,
                radius: 10,
                color: "#f59e0b",
                agentId: agentId,
                ownerType: "individual" as const,
                confidence: 0.75,
                importance: 0.6,
                lastUpdated: new Date().toISOString(),
                createdAt: new Date(Date.now() - 43200000).toISOString(),
                tags: ["trading", "opportunity", "belief"],
                metadata: {
                  source: "observation",
                  reasoning: "pattern_recognition",
                },
              },
            ],
            edges: [],
            isVisible: true,
            opacity: 0.8,
            color: "#f59e0b",
          },
        ],
        createdAt: new Date(Date.now() - 259200000).toISOString(),
        lastUpdated: new Date().toISOString(),
        version: "1.0.0",
        layout: "force-directed" as const,
        renderer: "auto" as const,
        maxNodes: 500,
        lodEnabled: true,
        clusteringEnabled: false,
        filters: {
          nodeTypes: ["belief", "fact"],
          confidenceRange: [0.0, 1.0] as [number, number],
          importanceRange: [0.0, 1.0] as [number, number],
          agentIds: [agentId],
          tags: [],
          edgeTypes: ["supports", "contradicts"],
          strengthRange: [0.0, 1.0] as [number, number],
          showOnlyConnected: false,
          hideIsolatedNodes: true,
        },
        selectedNodes: [],
        selectedEdges: [],
        zoom: 1.0,
        pan: { x: 0, y: 0 },
        metadata: { agentSpecific: true, personalizedView: true },
      };

      mockKnowledgeGraphs.push(individualGraph);
    }

    // Apply filters
    let filteredGraphs = mockKnowledgeGraphs;

    if (layerType) {
      filteredGraphs = filteredGraphs.filter((graph) =>
        graph.layers.some((layer) => layer.type === layerType),
      );
    }

    // Apply pagination
    const paginatedGraphs = filteredGraphs.slice(offset, offset + limit);

    // Remove metadata if not requested
    if (!includeMetadata) {
      paginatedGraphs.forEach((graph) => {
        if (graph.metadata) delete graph.metadata;
        graph.layers.forEach((layer: any) => {
          layer.nodes.forEach((node: any) => {
            if (node.metadata) delete node.metadata;
          });
          layer.edges.forEach((edge: any) => {
            if (edge.metadata) delete edge.metadata;
          });
        });
      });
    }

    return NextResponse.json({
      success: true,
      data: paginatedGraphs,
      pagination: {
        total: filteredGraphs.length,
        limit,
        offset,
        hasMore: offset + limit < filteredGraphs.length,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Knowledge graph API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
        message: "Failed to retrieve knowledge graphs",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
}

// POST /api/knowledge - Create or update knowledge graph
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate request body
    const validatedData = KnowledgeGraphSchema.parse(body);

    // Mock creation/update - would integrate with actual knowledge systems
    const timestamp = new Date().toISOString();
    const createdGraph = {
      ...validatedData,
      lastUpdated: timestamp,
      version: "1.0.0",
    };

    return NextResponse.json(
      {
        success: true,
        data: createdGraph,
        message: "Knowledge graph created successfully",
        timestamp,
      },
      { status: 201 },
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        {
          success: false,
          error: "Validation error",
          details: error.errors,
          timestamp: new Date().toISOString(),
        },
        { status: 400 },
      );
    }

    console.error("Knowledge graph creation error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
        message: "Failed to create knowledge graph",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
}

// PUT /api/knowledge - Update existing knowledge graph
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate request body
    const validatedData = KnowledgeGraphSchema.parse(body);

    // Mock update - would integrate with actual knowledge systems
    const timestamp = new Date().toISOString();
    const updatedGraph = {
      ...validatedData,
      lastUpdated: timestamp,
      version: "1.0.1", // Increment version
    };

    return NextResponse.json({
      success: true,
      data: updatedGraph,
      message: "Knowledge graph updated successfully",
      timestamp,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        {
          success: false,
          error: "Validation error",
          details: error.errors,
          timestamp: new Date().toISOString(),
        },
        { status: 400 },
      );
    }

    console.error("Knowledge graph update error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
        message: "Failed to update knowledge graph",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
}

// DELETE /api/knowledge - Delete knowledge graph
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const graphId = searchParams.get("id");

    if (!graphId) {
      return NextResponse.json(
        {
          success: false,
          error: "Bad request",
          message: "Graph ID is required",
          timestamp: new Date().toISOString(),
        },
        { status: 400 },
      );
    }

    // Mock deletion - would integrate with actual knowledge systems
    return NextResponse.json({
      success: true,
      message: `Knowledge graph ${graphId} deleted successfully`,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Knowledge graph deletion error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Internal server error",
        message: "Failed to delete knowledge graph",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
}
