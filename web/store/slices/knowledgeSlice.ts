import { createSlice, PayloadAction } from "@reduxjs/toolkit";

// Types from PRD
export interface KnowledgeNode {
  id: string;
  label: string;
  type: "belief" | "fact" | "hypothesis";
  confidence: number; // 0-1
  agents: string[]; // Agent IDs who hold this knowledge
  createdAt: number;
  lastModified: number;
  position?: {
    x: number;
    y: number;
    z?: number; // For 3D visualization
  };
  metadata?: {
    source?: string;
    category?: string;
    tags?: string[];
  };
}

export interface KnowledgeEdge {
  id: string;
  source: string; // Node ID
  target: string; // Node ID
  type: "supports" | "contradicts" | "related";
  strength: number; // 0-1
  agents: string[]; // Agents who established this relationship
  createdAt: number;
}

export interface KnowledgeGraph {
  nodes: Record<string, KnowledgeNode>;
  edges: Record<string, KnowledgeEdge>;
  layout: "2d" | "3d";
  physics: {
    enabled: boolean;
    charge: number;
    linkDistance: number;
    linkStrength: number;
  };
}

export interface KnowledgeFilters {
  confidenceThreshold: number;
  types: ("belief" | "fact" | "hypothesis")[];
  agentIds: string[];
  timeRange?: {
    start: number;
    end: number;
  };
  searchQuery: string;
}

interface KnowledgeState {
  graph: KnowledgeGraph;
  filters: KnowledgeFilters;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  hoveredNodeId: string | null;
  viewMode: "collective" | "individual" | "comparison";
  focusedAgentId: string | null; // For individual view
  comparisonAgentIds: string[]; // For comparison view
  renderEngine: "svg" | "canvas" | "webgl";
  zoom: number;
  center: { x: number; y: number };
}

// Demo data for CEO presentation
const demoNodes: Record<string, KnowledgeNode> = {
  "node-1": {
    id: "node-1",
    label: "Multi-Agent Systems",
    type: "fact",
    confidence: 0.95,
    agents: ["agent-1", "agent-2"],
    createdAt: Date.now() - 86400000,
    lastModified: Date.now() - 3600000,
    position: { x: 400, y: 300 },
    metadata: { category: "AI Research", tags: ["systems", "agents"] },
  },
  "node-2": {
    id: "node-2",
    label: "Active Inference Framework",
    type: "hypothesis",
    confidence: 0.87,
    agents: ["agent-1", "agent-3"],
    createdAt: Date.now() - 72000000,
    lastModified: Date.now() - 1800000,
    position: { x: 600, y: 200 },
    metadata: {
      category: "Cognitive Science",
      tags: ["inference", "bayesian"],
    },
  },
  "node-3": {
    id: "node-3",
    label: "Belief Propagation Networks",
    type: "belief",
    confidence: 0.78,
    agents: ["agent-2", "agent-3"],
    createdAt: Date.now() - 43200000,
    lastModified: Date.now() - 900000,
    position: { x: 200, y: 400 },
    metadata: {
      category: "Machine Learning",
      tags: ["networks", "propagation"],
    },
  },
  "node-4": {
    id: "node-4",
    label: "Coalition Formation Theory",
    type: "fact",
    confidence: 0.92,
    agents: ["agent-1", "agent-4"],
    createdAt: Date.now() - 21600000,
    lastModified: Date.now() - 300000,
    position: { x: 500, y: 500 },
    metadata: { category: "Game Theory", tags: ["coalitions", "cooperation"] },
  },
  "node-5": {
    id: "node-5",
    label: "Emergent Communication Protocols",
    type: "hypothesis",
    confidence: 0.71,
    agents: ["agent-3", "agent-4"],
    createdAt: Date.now() - 10800000,
    lastModified: Date.now() - 150000,
    position: { x: 350, y: 150 },
    metadata: { category: "Communication", tags: ["emergence", "protocols"] },
  },
};

const demoEdges: Record<string, KnowledgeEdge> = {
  "edge-1": {
    id: "edge-1",
    source: "node-1",
    target: "node-2",
    type: "supports",
    strength: 0.8,
    agents: ["agent-1"],
    createdAt: Date.now() - 36000000,
  },
  "edge-2": {
    id: "edge-2",
    source: "node-2",
    target: "node-3",
    type: "related",
    strength: 0.6,
    agents: ["agent-3"],
    createdAt: Date.now() - 18000000,
  },
  "edge-3": {
    id: "edge-3",
    source: "node-1",
    target: "node-4",
    type: "supports",
    strength: 0.9,
    agents: ["agent-1", "agent-4"],
    createdAt: Date.now() - 9000000,
  },
  "edge-4": {
    id: "edge-4",
    source: "node-3",
    target: "node-5",
    type: "contradicts",
    strength: 0.4,
    agents: ["agent-3"],
    createdAt: Date.now() - 4500000,
  },
  "edge-5": {
    id: "edge-5",
    source: "node-4",
    target: "node-5",
    type: "supports",
    strength: 0.7,
    agents: ["agent-4"],
    createdAt: Date.now() - 2700000,
  },
};

const initialState: KnowledgeState = {
  graph: {
    nodes: demoNodes, // ← NOW HAS DEMO DATA
    edges: demoEdges, // ← NOW HAS DEMO DATA
    layout: "2d",
    physics: {
      enabled: true,
      charge: -300,
      linkDistance: 100,
      linkStrength: 1,
    },
  },
  filters: {
    confidenceThreshold: 0,
    types: ["belief", "fact", "hypothesis"],
    agentIds: [],
    searchQuery: "",
  },
  selectedNodeId: null,
  selectedEdgeId: null,
  hoveredNodeId: null,
  viewMode: "collective",
  focusedAgentId: null,
  comparisonAgentIds: [],
  renderEngine: "svg",
  zoom: 1,
  center: { x: 0, y: 0 },
};

const knowledgeSlice = createSlice({
  name: "knowledge",
  initialState,
  reducers: {
    // Node management
    addKnowledgeNode: (
      state,
      action: PayloadAction<Omit<KnowledgeNode, "createdAt" | "lastModified">>,
    ) => {
      const node: KnowledgeNode = {
        ...action.payload,
        createdAt: Date.now(),
        lastModified: Date.now(),
      };
      state.graph.nodes[node.id] = node;
    },

    updateKnowledgeNode: (
      state,
      action: PayloadAction<{
        id: string;
        updates: Partial<KnowledgeNode>;
      }>,
    ) => {
      const { id, updates } = action.payload;
      if (state.graph.nodes[id]) {
        state.graph.nodes[id] = {
          ...state.graph.nodes[id],
          ...updates,
          lastModified: Date.now(),
        };
      }
    },

    removeKnowledgeNode: (state, action: PayloadAction<string>) => {
      const nodeId = action.payload;
      delete state.graph.nodes[nodeId];

      // Remove edges connected to this node
      Object.keys(state.graph.edges).forEach((edgeId) => {
        const edge = state.graph.edges[edgeId];
        if (edge.source === nodeId || edge.target === nodeId) {
          delete state.graph.edges[edgeId];
        }
      });
    },

    // Edge management
    addKnowledgeEdge: (
      state,
      action: PayloadAction<Omit<KnowledgeEdge, "id" | "createdAt">>,
    ) => {
      const edgeId = `${action.payload.source}-${action.payload.target}`;
      const edge: KnowledgeEdge = {
        ...action.payload,
        id: edgeId,
        createdAt: Date.now(),
      };
      state.graph.edges[edgeId] = edge;
    },

    updateKnowledgeEdge: (
      state,
      action: PayloadAction<{
        id: string;
        updates: Partial<KnowledgeEdge>;
      }>,
    ) => {
      const { id, updates } = action.payload;
      if (state.graph.edges[id]) {
        state.graph.edges[id] = {
          ...state.graph.edges[id],
          ...updates,
        };
      }
    },

    removeKnowledgeEdge: (state, action: PayloadAction<string>) => {
      delete state.graph.edges[action.payload];
    },

    // Selection
    selectNode: (state, action: PayloadAction<string | null>) => {
      state.selectedNodeId = action.payload;
      state.selectedEdgeId = null;
    },

    selectEdge: (state, action: PayloadAction<string | null>) => {
      state.selectedEdgeId = action.payload;
      state.selectedNodeId = null;
    },

    hoverNode: (state, action: PayloadAction<string | null>) => {
      state.hoveredNodeId = action.payload;
    },

    // View management
    setViewMode: (state, action: PayloadAction<KnowledgeState["viewMode"]>) => {
      state.viewMode = action.payload;
    },

    setFocusedAgent: (state, action: PayloadAction<string | null>) => {
      state.focusedAgentId = action.payload;
      if (action.payload) {
        state.viewMode = "individual";
      }
    },

    setComparisonAgents: (state, action: PayloadAction<string[]>) => {
      state.comparisonAgentIds = action.payload;
      if (action.payload.length > 0) {
        state.viewMode = "comparison";
      }
    },

    // Filters
    updateFilters: (
      state,
      action: PayloadAction<Partial<KnowledgeFilters>>,
    ) => {
      state.filters = {
        ...state.filters,
        ...action.payload,
      };
    },

    // Layout and rendering
    setLayout: (state, action: PayloadAction<"2d" | "3d">) => {
      state.graph.layout = action.payload;
      // Switch render engine based on layout
      state.renderEngine =
        action.payload === "3d"
          ? "webgl"
          : Object.keys(state.graph.nodes).length > 100
            ? "canvas"
            : "svg";
    },

    togglePhysics: (state) => {
      state.graph.physics.enabled = !state.graph.physics.enabled;
    },

    updatePhysics: (
      state,
      action: PayloadAction<Partial<KnowledgeGraph["physics"]>>,
    ) => {
      state.graph.physics = {
        ...state.graph.physics,
        ...action.payload,
      };
    },

    setRenderEngine: (
      state,
      action: PayloadAction<KnowledgeState["renderEngine"]>,
    ) => {
      state.renderEngine = action.payload;
    },

    // Zoom and pan
    setZoom: (state, action: PayloadAction<number>) => {
      state.zoom = Math.max(0.1, Math.min(10, action.payload));
    },

    setCenter: (state, action: PayloadAction<{ x: number; y: number }>) => {
      state.center = action.payload;
    },

    // Batch operations
    batchAddNodes: (state, action: PayloadAction<KnowledgeNode[]>) => {
      action.payload.forEach((node) => {
        state.graph.nodes[node.id] = node;
      });
    },

    batchAddEdges: (state, action: PayloadAction<KnowledgeEdge[]>) => {
      action.payload.forEach((edge) => {
        state.graph.edges[edge.id] = edge;
      });
    },

    // Clear graph
    clearGraph: (state) => {
      state.graph.nodes = {};
      state.graph.edges = {};
      state.selectedNodeId = null;
      state.selectedEdgeId = null;
      state.hoveredNodeId = null;
    },

    // Agent knowledge update
    updateAgentKnowledge: (
      state,
      action: PayloadAction<{
        agentId: string;
        nodeIds: string[];
        operation: "add" | "remove";
      }>,
    ) => {
      const { agentId, nodeIds, operation } = action.payload;

      nodeIds.forEach((nodeId) => {
        if (state.graph.nodes[nodeId]) {
          if (
            operation === "add" &&
            !state.graph.nodes[nodeId].agents.includes(agentId)
          ) {
            state.graph.nodes[nodeId].agents.push(agentId);
          } else if (operation === "remove") {
            state.graph.nodes[nodeId].agents = state.graph.nodes[
              nodeId
            ].agents.filter((id) => id !== agentId);
          }
        }
      });
    },
  },
});

export const {
  addKnowledgeNode,
  updateKnowledgeNode,
  removeKnowledgeNode,
  addKnowledgeEdge,
  updateKnowledgeEdge,
  removeKnowledgeEdge,
  selectNode,
  selectEdge,
  hoverNode,
  setViewMode,
  setFocusedAgent,
  setComparisonAgents,
  updateFilters,
  setLayout,
  togglePhysics,
  updatePhysics,
  setRenderEngine,
  setZoom,
  setCenter,
  batchAddNodes,
  batchAddEdges,
  clearGraph,
  updateAgentKnowledge,
} = knowledgeSlice.actions;

export default knowledgeSlice.reducer;
