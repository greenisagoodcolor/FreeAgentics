import { useState, useEffect, useCallback, useRef } from "react";
import { useAuth } from "./use-auth";

export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

export interface WebSocketMessage {
  type: string;
  data: unknown;
  timestamp?: number;
}

export interface WebSocketState {
  isConnected: boolean;
  sendMessage: (message: WebSocketMessage) => void;
  lastMessage: WebSocketMessage | null;
  connectionState: ConnectionState;
  error: Error | null;
}

// Use dev endpoint for development
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/v1/ws/dev";
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export function useWebSocket(): WebSocketState {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isConnectingRef = useRef(false);

  const { isAuthenticated, isLoading: isAuthLoading, token } = useAuth();

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log("[WebSocket] Connection already in progress, skipping...");
      return;
    }

    // Don't connect if auth is not ready
    if (isAuthLoading || !isAuthenticated || !token) {
      console.log("[WebSocket] Waiting for auth before connecting...", {
        isAuthLoading,
        isAuthenticated,
        hasToken: !!token,
      });
      return;
    }

    isConnectingRef.current = true;

    try {
      setConnectionState("connecting");
      setError(null);

      // Append token to WebSocket URL if available
      let wsUrl = WS_URL;
      if (token) {
        const separator = WS_URL.includes("?") ? "&" : "?";
        wsUrl = `${WS_URL}${separator}token=${encodeURIComponent(token)}`;
        console.log("[WebSocket] Connecting with auth token...");
      }

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("[WebSocket] Connected successfully");
        setIsConnected(true);
        setConnectionState("connected");
        reconnectAttemptsRef.current = 0;
        isConnectingRef.current = false;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          setLastMessage({
            ...message,
            timestamp: message.timestamp || Date.now(),
          });
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err);
        }
      };

      ws.onerror = (event) => {
        console.error("WebSocket error:", event);
        setError(new Error("WebSocket connection error"));
        setConnectionState("error");
      };

      ws.onclose = () => {
        console.log("[WebSocket] Disconnected");
        setIsConnected(false);
        setConnectionState("disconnected");
        wsRef.current = null;
        isConnectingRef.current = false;

        // Attempt to reconnect only if we're still authenticated
        if (isAuthenticated && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttemptsRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(
              `[WebSocket] Attempting to reconnect (${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})...`,
            );
            connect();
          }, RECONNECT_DELAY);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error("[WebSocket] Failed to create connection:", err);
      setError(err as Error);
      setConnectionState("error");
      isConnectingRef.current = false;
    }
  }, [isAuthLoading, isAuthenticated, token]);

  const disconnect = useCallback(() => {
    // Clear any pending connection attempts
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    isConnectingRef.current = false;
    setIsConnected(false);
    setConnectionState("disconnected");
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          ...message,
          timestamp: Date.now(),
        }),
      );
    } else {
      console.warn("[WebSocket] Cannot send message: Not connected");
      setError(new Error("WebSocket is not connected"));
    }
  }, []);

  // Connect when auth is ready, disconnect on unmount
  useEffect(() => {
    // Clear any existing connection timeout
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }

    if (!isAuthLoading && isAuthenticated && token && !wsRef.current) {
      console.log("[WebSocket] Auth ready, scheduling connection...");
      // Add small delay to ensure token propagation
      connectionTimeoutRef.current = setTimeout(() => {
        console.log("[WebSocket] Initiating connection after auth stabilization...");
        connect();
      }, 100); // Reduced delay since we now prevent duplicate connections
    }

    return () => {
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }
    };
  }, [connect, isAuthLoading, isAuthenticated, token]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    sendMessage,
    lastMessage,
    connectionState,
    error,
  };
}
