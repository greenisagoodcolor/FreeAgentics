import { useState, useEffect, useCallback, useRef } from "react";
import { apiClient } from "@/lib/api-client";
import { PromptAgent } from "@/types/agent";
import { getWebSocketUrl } from "../utils/websocket-url";

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
  const [agents, setAgents] = useState<PromptAgent[]>([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph>({ nodes: [], edges: [] });
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [currentPromptId, setCurrentPromptId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [iterationContext, setIterationContext] = useState<IterationContext | null>(null);
  const [lastPrompt, setLastPrompt] = useState<string>("");

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const suggestionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  // Handle WebSocket messages - defined before useEffect to avoid dependency issues
  const handleWebSocketMessage = useCallback(
    (data: unknown) => {
      const message = data as Record<string, unknown>;
      switch (message.type) {
        case "agent_update":
          setAgents((prev) => {
            const agent = message.agent as PromptAgent;
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
      // Don't attempt connection if we've exceeded max attempts
      if (reconnectAttempts.current >= maxReconnectAttempts) {
        console.warn(`Max WebSocket reconnection attempts (${maxReconnectAttempts}) reached. Giving up.`);
        setConnectionState('error');
        setConnectionError(`Failed to connect after ${maxReconnectAttempts} attempts. Please check if the server is running.`);
        return;
      }

      try {
        setConnectionState('connecting');
        setConnectionError(null);
        
        // Use centralized WebSocket URL construction
        const wsUrl = getWebSocketUrl('dev');
        console.log(`Attempting WebSocket connection to: ${wsUrl} (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
        
        const ws = new WebSocket(wsUrl);

        ws.addEventListener("open", () => {
          console.log("✅ WebSocket connected successfully");
          setConnectionState('connected');
          setConnectionError(null);
          reconnectAttempts.current = 0; // Reset on successful connection
          
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

        ws.addEventListener("close", (event) => {
          console.log(`WebSocket disconnected (code: ${event.code}, reason: ${event.reason})`);
          wsRef.current = null;
          setConnectionState('disconnected');

          // Only attempt reconnection if we haven't exceeded max attempts
          if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current++;
            const backoffDelay = Math.min(1000 * Math.pow(2, reconnectAttempts.current - 1), 30000); // Exponential backoff, max 30s
            
            console.log(`Scheduling WebSocket reconnection in ${backoffDelay}ms (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
            setConnectionError(`Connection lost. Reconnecting in ${Math.ceil(backoffDelay / 1000)}s...`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              initWebSocket();
            }, backoffDelay);
          } else {
            setConnectionState('error');
            setConnectionError('Connection lost. Max reconnection attempts reached.');
          }
        });

        ws.addEventListener("error", (error) => {
          console.error("WebSocket error:", error);
          setConnectionState('error');
          setConnectionError('WebSocket connection failed. Please check if the server is running on http://localhost:8000');
        });

        wsRef.current = ws;
      } catch (err) {
        console.error("Failed to initialize WebSocket:", err);
        setConnectionState('error');
        setConnectionError(`Failed to initialize WebSocket: ${err instanceof Error ? err.message : String(err)}`);
        reconnectAttempts.current++;
      }
    };

    // Start connection
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
      setLastPrompt(prompt); // Store prompt for retry functionality

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
    if (lastPrompt && !isLoading) {
      // Clear error and retry the last prompt
      setError(null);
      submitPrompt(lastPrompt);
    }
  }, [lastPrompt, isLoading, submitPrompt]);

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
    connectionState,
    connectionError,
  };
}
