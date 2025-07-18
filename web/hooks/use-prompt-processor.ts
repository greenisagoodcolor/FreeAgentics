import { useState, useEffect, useCallback, useRef } from "react";
import { getApiClient } from "../lib/api-client";

interface Agent {
  id: string;
  name: string;
  status: string;
  type?: string;
  thinking?: boolean;
}

interface KnowledgeGraphNode {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, any>;
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

interface IterationContext {
  iteration_number: number;
  total_agents: number;
  kg_nodes: number;
  conversation_summary: {
    iteration_count: number;
    belief_evolution: {
      trend: string;
      stability: number;
    };
    prompt_themes?: string[];
    suggestion_patterns?: {
      diversity: number;
      common_themes: string[];
    };
  };
}

interface PromptResponse {
  id: string;
  prompt: string;
  status: string;
  agents: Agent[];
  knowledge_graph?: KnowledgeGraph;
  next_suggestions?: string[];
  iteration_context?: IterationContext;
}

export function usePromptProcessor() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [currentPromptId, setCurrentPromptId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [iterationContext, setIterationContext] = useState<IterationContext | null>(null);

  const apiClient = getApiClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const suggestionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const initWebSocket = () => {
      try {
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
        const ws = new WebSocket(wsUrl);

        ws.addEventListener("open", () => {
          console.log("WebSocket connected");
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
          }
        });

        ws.addEventListener("message", (event) => {
          try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
          } catch (err) {
            console.error("Failed to parse WebSocket message:", err);
          }
        });

        ws.addEventListener("close", () => {
          console.log("WebSocket disconnected");
          wsRef.current = null;

          // Attempt reconnection after 3 seconds
          reconnectTimeoutRef.current = setTimeout(() => {
            initWebSocket();
          }, 3000);
        });

        ws.addEventListener("error", (error) => {
          console.error("WebSocket error:", error);
        });

        wsRef.current = ws;
      } catch (err) {
        console.error("Failed to initialize WebSocket:", err);
      }
    };

    initWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (suggestionTimeoutRef.current) {
        clearTimeout(suggestionTimeoutRef.current);
      }
    };
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback(
    (data: any) => {
      switch (data.type) {
        case "agent_update":
          setAgents((prev) => {
            const index = prev.findIndex((a) => a.id === data.agent.id);
            if (index >= 0) {
              const updated = [...prev];
              updated[index] = { ...updated[index], ...data.agent };
              return updated;
            }
            return [...prev, data.agent];
          });
          break;

        case "knowledge_update":
          if (data.knowledge_graph) {
            setKnowledgeGraph(data.knowledge_graph);
          }
          break;

        case "prompt_complete":
          if (data.prompt_id === currentPromptId) {
            setIsLoading(false);
            if (data.error) {
              setError(data.error);
            }
          }
          break;

        case "pipeline:pipeline_completed":
          if (data.suggestions) {
            setSuggestions(data.suggestions);
          }
          if (data.conversation_summary) {
            setIterationContext((prev) => ({
              iteration_number: data.iteration_number || (prev?.iteration_number || 0) + 1,
              total_agents: data.total_agents || prev?.total_agents || 1,
              kg_nodes: data.kg_nodes || prev?.kg_nodes || 0,
              conversation_summary: data.conversation_summary,
            }));
          }
          break;

        case "pipeline:pipeline_started":
          if (data.iteration_number && data.conversation_summary) {
            setIterationContext({
              iteration_number: data.iteration_number,
              total_agents: data.conversation_summary.total_agents || 0,
              kg_nodes: data.conversation_summary.kg_nodes || 0,
              conversation_summary: data.conversation_summary,
            });
          }
          break;
      }
    },
    [currentPromptId],
  );

  // Submit prompt to API
  const submitPrompt = useCallback(
    async (prompt: string, useConversation: boolean = true) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await apiClient.submitPrompt(prompt);

        setCurrentPromptId(response.id);

        // Update agents and knowledge graph
        if (response.agents) {
          setAgents(response.agents);
        }

        if (response.knowledge_graph) {
          setKnowledgeGraph(response.knowledge_graph);
        }

        // Generate a conversation ID if needed and not already set
        if (!conversationId && useConversation) {
          const newConversationId = `conv-${Date.now()}`;
          setConversationId(newConversationId);
        }

        // Send to WebSocket for real-time updates
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: "prompt_submitted",
              prompt_id: response.id,
              prompt,
              conversation_id: conversationId,
            }),
          );
        }

        // If status is not completed, we'll wait for WebSocket updates
        if (response.status === "completed") {
          setIsLoading(false);
        }
      } catch (err: any) {
        setError(err.detail || "Failed to process prompt");
        setIsLoading(false);
      }
    },
    [apiClient, conversationId],
  );

  // Retry last prompt
  const retry = useCallback(() => {
    if (currentPromptId) {
      // Implementation would retry the last prompt
      setError(null);
    }
  }, [currentPromptId]);

  // Fetch suggestions (debounced)
  const fetchSuggestions = useCallback(
    (query: string) => {
      // Clear previous timeout
      if (suggestionTimeoutRef.current) {
        clearTimeout(suggestionTimeoutRef.current);
      }

      // Debounce suggestions request
      suggestionTimeoutRef.current = setTimeout(async () => {
        try {
          const response = await apiClient.getPromptSuggestions(query);
          setSuggestions(response.suggestions || []);
        } catch (err) {
          // Fallback to local suggestions
          const localSuggestions = [
            "How can I optimize my agent's performance?",
            "How do agents form coalitions?",
            "How does active inference work?",
            "Show me the current agent network",
            "What is the free energy principle?",
          ].filter((s) => s.toLowerCase().includes(query.toLowerCase()));

          setSuggestions(localSuggestions);
        }
      }, 300); // 300ms debounce
    },
    [apiClient],
  );

  // Reset conversation to start fresh
  const resetConversation = useCallback(() => {
    setConversationId(null);
    setIterationContext(null);
    setAgents([]);
    setKnowledgeGraph(null);
    setSuggestions([]);
    setCurrentPromptId(null);
    setError(null);
  }, []);

  return {
    isLoading,
    error,
    agents,
    knowledgeGraph,
    suggestions,
    submitPrompt,
    retry,
    fetchSuggestions,
    conversationId,
    iterationContext,
    resetConversation,
  };
}
