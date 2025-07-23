import { useState, useCallback, useEffect } from "react";
import { useWebSocket } from "./use-websocket";

export type NodeType = "agent" | "belief" | "goal" | "observation" | "action";
export type EdgeType = "has_belief" | "has_goal" | "observes" | "performs" | "influences";

export interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  x?: number;
  y?: number;
  metadata?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
  metadata?: Record<string, unknown>;
}

export interface KnowledgeGraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  isLoading: boolean;
  error: Error | null;
  addNode: (node: Omit<GraphNode, "id">) => void;
  updateNode: (id: string, updates: Partial<GraphNode>) => void;
  removeNode: (id: string) => void;
  addEdge: (edge: Omit<GraphEdge, "id">) => void;
  removeEdge: (id: string) => void;
  clearGraph: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useKnowledgeGraph(): KnowledgeGraphState {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const { lastMessage, sendMessage } = useWebSocket();

  // Handle WebSocket updates
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "knowledge_graph_update") {
      const { nodes: newNodes, edges: newEdges, operation } = lastMessage.data;

      if (operation === "replace") {
        setNodes(newNodes || []);
        setEdges(newEdges || []);
      } else if (operation === "add_node" && newNodes?.[0]) {
        setNodes((prev) => [...prev, newNodes[0]]);
      } else if (operation === "update_node" && newNodes?.[0]) {
        const updatedNode = newNodes[0];
        setNodes((prev) =>
          prev.map((node) => (node.id === updatedNode.id ? { ...node, ...updatedNode } : node)),
        );
      } else if (operation === "remove_node") {
        const nodeId = lastMessage.data.nodeId;
        setNodes((prev) => prev.filter((node) => node.id !== nodeId));
        setEdges((prev) => prev.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
      } else if (operation === "add_edge" && newEdges?.[0]) {
        setEdges((prev) => [...prev, newEdges[0]]);
      } else if (operation === "remove_edge") {
        const edgeId = lastMessage.data.edgeId;
        setEdges((prev) => prev.filter((edge) => edge.id !== edgeId));
      }
    }
  }, [lastMessage]);

  // Fetch initial graph data
  useEffect(() => {
    fetchGraph();
  }, []);

  const fetchGraph = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/knowledge-graph`);
      if (!response.ok) {
        throw new Error("Failed to fetch knowledge graph");
      }

      const data = await response.json();
      setNodes(data.nodes || []);
      setEdges(data.edges || []);
    } catch (err) {
      setError(err as Error);
      console.error("Failed to fetch knowledge graph:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const addNode = useCallback(
    (node: Omit<GraphNode, "id">) => {
      const newNode: GraphNode = {
        ...node,
        id: `node-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      };

      // Optimistically update
      setNodes((prev) => [...prev, newNode]);

      // Send to server
      sendMessage({
        type: "add_graph_node",
        data: newNode,
      });
    },
    [sendMessage],
  );

  const updateNode = useCallback(
    (id: string, updates: Partial<GraphNode>) => {
      // Optimistically update
      setNodes((prev) => prev.map((node) => (node.id === id ? { ...node, ...updates } : node)));

      // Send to server
      sendMessage({
        type: "update_graph_node",
        data: { id, updates },
      });
    },
    [sendMessage],
  );

  const removeNode = useCallback(
    (id: string) => {
      // Optimistically update
      setNodes((prev) => prev.filter((node) => node.id !== id));
      setEdges((prev) => prev.filter((edge) => edge.source !== id && edge.target !== id));

      // Send to server
      sendMessage({
        type: "remove_graph_node",
        data: { id },
      });
    },
    [sendMessage],
  );

  const addEdge = useCallback(
    (edge: Omit<GraphEdge, "id">) => {
      const newEdge: GraphEdge = {
        ...edge,
        id: `edge-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      };

      // Optimistically update
      setEdges((prev) => [...prev, newEdge]);

      // Send to server
      sendMessage({
        type: "add_graph_edge",
        data: newEdge,
      });
    },
    [sendMessage],
  );

  const removeEdge = useCallback(
    (id: string) => {
      // Optimistically update
      setEdges((prev) => prev.filter((edge) => edge.id !== id));

      // Send to server
      sendMessage({
        type: "remove_graph_edge",
        data: { id },
      });
    },
    [sendMessage],
  );

  const clearGraph = useCallback(() => {
    // Optimistically update
    setNodes([]);
    setEdges([]);

    // Send to server
    sendMessage({
      type: "clear_graph",
      data: {},
    });
  }, [sendMessage]);

  return {
    nodes,
    edges,
    isLoading,
    error,
    addNode,
    updateNode,
    removeNode,
    addEdge,
    removeEdge,
    clearGraph,
  };
}
