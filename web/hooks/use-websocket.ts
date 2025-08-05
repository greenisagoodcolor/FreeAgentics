import { useState, useEffect, useCallback, useRef } from "react";
import { useAuth } from "./use-auth";
import { getWebSocketUrl, getAuthenticatedWebSocketUrl, isValidWebSocketUrl } from "../utils/websocket-url";

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

// Use dev endpoint for development (matches backend dev mode)
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
    if (isConnectingRef.current || 
        wsRef.current?.readyState === WebSocket.CONNECTING ||
        wsRef.current?.readyState === WebSocket.OPEN) {
      console.log("[WebSocket] Connection already exists or in progress, skipping...", {
        isConnecting: isConnectingRef.current,
        readyState: wsRef.current?.readyState,
        stateText: wsRef.current?.readyState === WebSocket.OPEN ? 'OPEN' : 
                   wsRef.current?.readyState === WebSocket.CONNECTING ? 'CONNECTING' : 'OTHER'
      });
      return;
    }

    // For dev endpoint, don't require authentication in dev mode
    const WS_URL = getWebSocketUrl('dev');
    const isDevEndpoint = WS_URL.includes("/ws/dev");
    if (!isDevEndpoint && (isAuthLoading || !isAuthenticated || !token)) {
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

      // Validate and construct WebSocket URL
      if (!isValidWebSocketUrl(WS_URL)) {
        throw new Error(`Invalid WebSocket URL: ${WS_URL}`);
      }

      // Use the authenticated URL function if we have a token
      const finalUrl = token && !isDevEndpoint 
        ? getAuthenticatedWebSocketUrl('dev', token)
        : WS_URL;
      
      if (token && !isDevEndpoint) {
        console.log("[WebSocket] Connecting with auth token...");
      } else if (isDevEndpoint) {
        console.log("[WebSocket] Connecting to dev endpoint without auth...");
      }
      console.log('[WebSocket] Final URL:', finalUrl);

      const ws = new WebSocket(finalUrl);

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

  // Unified connection lifecycle management
  useEffect(() => {
    // Clear any existing connection timeout to prevent race conditions
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }

    const isDevEndpoint = getWebSocketUrl('dev').includes("/ws/dev");
    const shouldConnect = isDevEndpoint 
      ? true  // Dev endpoint always ready
      : (!isAuthLoading && isAuthenticated && token);  // Production needs auth

    // Only connect if we should connect and don't have an active connection
    if (shouldConnect && !wsRef.current && !isConnectingRef.current) {
      const endpointType = isDevEndpoint ? "dev" : "authenticated";
      console.log(`[WebSocket] ${endpointType} connection ready, scheduling connection...`);
      
      connectionTimeoutRef.current = setTimeout(() => {
        // Double-check connection state to prevent race conditions
        if (!wsRef.current && !isConnectingRef.current) {
          console.log(`[WebSocket] Initiating ${endpointType} connection...`);
          connect();
        } else {
          console.log(`[WebSocket] Connection already exists or in progress, skipping...`);
        }
      }, 100);
    }

    // Cleanup function
    return () => {
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }
      // Only disconnect on unmount, not on dependency changes
      disconnect();
    };
  }, [connect, disconnect, isAuthLoading, isAuthenticated, token]);

  return {
    isConnected,
    sendMessage,
    lastMessage,
    connectionState,
    error,
  };
}
