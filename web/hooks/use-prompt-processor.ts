import { useState, useEffect, useCallback, useRef } from "react";
import { apiClient } from "@/lib/api-client";

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
  properties?: Record<string, unknown>;
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

export function usePromptProcessor() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph>({ nodes: [], edges: [] });
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [currentPromptId, setCurrentPromptId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [iterationContext, setIterationContext] = useState<IterationContext | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const suggestionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Handle WebSocket messages - defined before useEffect to avoid dependency issues
  const handleWebSocketMessage = useCallback(
    (data: unknown) => {
      const message = data as Record<string, unknown>;
      switch (message.type) {
        case "agent_update":
          setAgents((prev) => {
            const agent = message.agent as Agent;
            const index = prev.findIndex((a) => a.id === agent.id);
            if (index >= 0) {
              const updated = [...prev];
              updated[index] = { ...updated[index], ...agent };
              return updated;
            }
            return [...prev, agent];
          });
          break;

        case "knowledge_update":
          if (message.knowledge_graph) {
            setKnowledgeGraph(message.knowledge_graph as KnowledgeGraph);
          }
          break;

        case "prompt_complete":
          if (message.prompt_id === currentPromptId) {
            setIsLoading(false);
            if (message.error) {
              setError(message.error as string);
            }
          }
          break;

        case "pipeline:pipeline_completed":
          if (message.suggestions) {
            setSuggestions(message.suggestions as string[]);
          }
          if (message.conversation_summary) {
            setIterationContext((prev) => ({
              iteration_number:
                (message.iteration_number as number) || (prev?.iteration_number || 0) + 1,
              total_agents: (message.total_agents as number) || prev?.total_agents || 1,
              kg_nodes: (message.kg_nodes as number) || prev?.kg_nodes || 0,
              conversation_summary:
                message.conversation_summary as IterationContext["conversation_summary"],
            }));
          }
          break;

        case "pipeline:pipeline_started":
          if (message.iteration_number && message.conversation_summary) {
            const summary = message.conversation_summary as Record<string, unknown>;
            setIterationContext({
              iteration_number: message.iteration_number as number,
              total_agents: (summary.total_agents as number) || 0,
              kg_nodes: (summary.kg_nodes as number) || 0,
              conversation_summary: summary as IterationContext["conversation_summary"],
            });
          }
          break;
      }
    },
    [currentPromptId],
  );

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
  }, [handleWebSocketMessage]);

  // Submit prompt to API
  const submitPrompt = useCallback(
    async (prompt: string, useConversation: boolean = true) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await apiClient.processPrompt({
          prompt,
          conversationId: useConversation ? conversationId || undefined : undefined,
        });

        if (!response.success) {
          setCurrentPromptId(`prompt-${Date.now()}`); // Set prompt ID so retry can work
          setError(response.error || "Failed to process prompt");
          setIsLoading(false);
          return;
        }

        if (!response.data) {
          setError("No data received from server");
          setIsLoading(false);
          return;
        }

        const {
          agents: newAgents,
          knowledgeGraph: newKnowledgeGraph,
          conversationId: responseConvId,
        } = response.data;

        // Update agents and knowledge graph
        if (newAgents) {
          setAgents(newAgents);
        }

        if (newKnowledgeGraph) {
          setKnowledgeGraph(newKnowledgeGraph);
        }

        // Set current prompt ID and generate a conversation ID if needed and not already set
        setCurrentPromptId(responseConvId || `prompt-${Date.now()}`);
        if (!conversationId && useConversation) {
          const newConversationId = responseConvId || `conv-${Date.now()}`;
          setConversationId(newConversationId);
        }

        // Send to WebSocket for real-time updates
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: "prompt_submitted",
              prompt_id: responseConvId || "unknown",
              prompt,
              conversation_id: conversationId,
            }),
          );
        }

        // If we got a response, assume it's completed
        setIsLoading(false);
      } catch (err) {
        const error = err as Error;
        setError(`Failed to process prompt: ${error.message}`);
        setIsLoading(false);
      }
    },
    [conversationId],
  );

  // Retry last prompt
  const retry = useCallback(() => {
    if (currentPromptId) {
      // Implementation would retry the last prompt
      setError(null);
    }
  }, [currentPromptId]);

  // Fetch suggestions (debounced)
  const fetchSuggestions = useCallback((query: string) => {
    // Clear previous timeout
    if (suggestionTimeoutRef.current) {
      clearTimeout(suggestionTimeoutRef.current);
    }

    // Don't fetch suggestions for empty queries
    if (!query.trim()) {
      setSuggestions([]);
      return;
    }

    // Debounce suggestions request
    suggestionTimeoutRef.current = setTimeout(async () => {
      try {
        const response = await apiClient.getSuggestions(query);
        if (response.success && response.data) {
          setSuggestions(response.data);
        }
      } catch (err) {
        console.error("Error fetching suggestions:", err);
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
  }, []);

  // Reset conversation to start fresh
  const resetConversation = useCallback(() => {
    setConversationId(null);
    setIterationContext(null);
    setAgents([]);
    setKnowledgeGraph({ nodes: [], edges: [] });
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
