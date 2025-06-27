import { useEffect, useRef, useState, useCallback } from "react";
import { KnowledgeGraphUpdate } from "@/lib/types";

// WebSocket Hook for Knowledge Graph Real-time Updates
// Implements ADR-008 WebSocket Communication patterns

export interface UseKnowledgeGraphWebSocketOptions {
  graphId?: string;
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  onUpdate?: (update: KnowledgeGraphUpdate) => void;
}

export interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastUpdate: KnowledgeGraphUpdate | null;
  connectionAttempts: number;
}

export interface UseKnowledgeGraphWebSocketReturn {
  state: WebSocketState;
  connect: () => Promise<boolean>;
  disconnect: () => void;
  sendMessage: (message: any) => boolean;
  subscribe: (
    eventType: string,
    callback: (update: KnowledgeGraphUpdate) => void,
  ) => () => void;
}

export function useKnowledgeGraphWebSocket(
  options: UseKnowledgeGraphWebSocketOptions = {},
): UseKnowledgeGraphWebSocketReturn {
  const {
    graphId,
    autoConnect = true,
    reconnectAttempts = 3,
    reconnectDelay = 1000,
    onConnect,
    onDisconnect,
    onError,
    onUpdate,
  } = options;

  // WebSocket connection state
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastUpdate: null,
    connectionAttempts: 0,
  });

  // Refs for WebSocket and event listeners
  const wsRef = useRef<WebSocket | null>(null);
  const eventListenersRef = useRef<
    Map<string, Set<(update: KnowledgeGraphUpdate) => void>>
  >(new Map());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Generate WebSocket URL
  const getWebSocketUrl = useCallback((graphId?: string): string => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const path = graphId ? `/ws/knowledge/${graphId}` : "/ws/knowledge";
    return `${protocol}//${host}${path}`;
  }, []);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const update: KnowledgeGraphUpdate = JSON.parse(event.data);

        // Update state with latest update
        setState((prev) => ({
          ...prev,
          lastUpdate: update,
          error: null,
        }));

        // Call general update handler
        onUpdate?.(update);

        // Emit to specific event type listeners
        const listeners = eventListenersRef.current.get(update.type);
        if (listeners) {
          listeners.forEach((callback) => {
            try {
              callback(update);
            } catch (error) {
              console.error("Error in WebSocket update callback:", error);
            }
          });
        }

        // Emit to general update listeners
        const generalListeners = eventListenersRef.current.get("update");
        if (generalListeners) {
          generalListeners.forEach((callback) => {
            try {
              callback(update);
            } catch (error) {
              console.error(
                "Error in general WebSocket update callback:",
                error,
              );
            }
          });
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
        setState((prev) => ({
          ...prev,
          error: "Failed to parse message",
        }));
      }
    },
    [onUpdate],
  );

  // Handle WebSocket connection open
  const handleOpen = useCallback(() => {
    console.log("Knowledge graph WebSocket connected");

    setState((prev) => ({
      ...prev,
      isConnected: true,
      isConnecting: false,
      error: null,
      connectionAttempts: 0,
    }));

    // Start heartbeat
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000); // 30 seconds

    onConnect?.();
  }, [onConnect]);

  // Handle WebSocket connection close
  const handleClose = useCallback(
    (event: CloseEvent) => {
      console.log(
        "Knowledge graph WebSocket disconnected:",
        event.code,
        event.reason,
      );

      setState((prev) => ({
        ...prev,
        isConnected: false,
        isConnecting: false,
      }));

      // Clear heartbeat
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }

      onDisconnect?.();

      // Attempt reconnection if not manually closed
      if (event.code !== 1000 && state.connectionAttempts < reconnectAttempts) {
        setState((prev) => ({
          ...prev,
          connectionAttempts: prev.connectionAttempts + 1,
        }));

        reconnectTimeoutRef.current = setTimeout(
          () => {
            connect();
          },
          reconnectDelay * Math.pow(2, state.connectionAttempts),
        ); // Exponential backoff
      }
    },
    [onDisconnect, state.connectionAttempts, reconnectAttempts, reconnectDelay],
  );

  // Handle WebSocket errors
  const handleError = useCallback(
    (event: Event) => {
      console.error("Knowledge graph WebSocket error:", event);

      setState((prev) => ({
        ...prev,
        error: "WebSocket connection error",
        isConnecting: false,
      }));

      onError?.(event);
    },
    [onError],
  );

  // Connect to WebSocket
  const connect = useCallback(async (): Promise<boolean> => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return true; // Already connected
    }

    if (state.isConnecting) {
      return false; // Already connecting
    }

    setState((prev) => ({
      ...prev,
      isConnecting: true,
      error: null,
    }));

    try {
      const url = getWebSocketUrl(graphId);
      const ws = new WebSocket(url);

      ws.onopen = handleOpen;
      ws.onmessage = handleMessage;
      ws.onclose = handleClose;
      ws.onerror = handleError;

      wsRef.current = ws;

      return new Promise((resolve) => {
        const checkConnection = () => {
          if (ws.readyState === WebSocket.OPEN) {
            resolve(true);
          } else if (
            ws.readyState === WebSocket.CLOSED ||
            ws.readyState === WebSocket.CLOSING
          ) {
            resolve(false);
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
      });
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
      setState((prev) => ({
        ...prev,
        isConnecting: false,
        error: "Failed to create connection",
      }));
      return false;
    }
  }, [
    graphId,
    state.isConnecting,
    getWebSocketUrl,
    handleOpen,
    handleMessage,
    handleClose,
    handleError,
  ]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    // Clear reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Clear heartbeat
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }

    // Close WebSocket connection
    if (wsRef.current) {
      wsRef.current.close(1000, "Manual disconnect");
      wsRef.current = null;
    }

    setState((prev) => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
      connectionAttempts: 0,
    }));
  }, []);

  // Send message via WebSocket
  const sendMessage = useCallback((message: any): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error("Failed to send WebSocket message:", error);
        setState((prev) => ({
          ...prev,
          error: "Failed to send message",
        }));
        return false;
      }
    }
    return false;
  }, []);

  // Subscribe to specific event types
  const subscribe = useCallback(
    (
      eventType: string,
      callback: (update: KnowledgeGraphUpdate) => void,
    ): (() => void) => {
      if (!eventListenersRef.current.has(eventType)) {
        eventListenersRef.current.set(eventType, new Set());
      }

      const listeners = eventListenersRef.current.get(eventType)!;
      listeners.add(callback);

      // Return unsubscribe function
      return () => {
        listeners.delete(callback);
        if (listeners.size === 0) {
          eventListenersRef.current.delete(eventType);
        }
      };
    },
    [],
  );

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Reconnect when graphId changes
  useEffect(() => {
    if (state.isConnected && graphId) {
      disconnect();
      setTimeout(() => connect(), 100);
    }
  }, [graphId]);

  // Handle page visibility changes to reconnect when page becomes visible
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (
        document.visibilityState === "visible" &&
        !state.isConnected &&
        !state.isConnecting
      ) {
        connect();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [state.isConnected, state.isConnecting, connect]);

  return {
    state,
    connect,
    disconnect,
    sendMessage,
    subscribe,
  };
}
