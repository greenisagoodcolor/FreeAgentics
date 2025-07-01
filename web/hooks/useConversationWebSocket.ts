"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useDispatch } from "react-redux";
import {
  setWebSocketStatus,
  connectionEstablished,
  connectionLost,
  updateLatency,
  addConnectionError,
} from "@/store/slices/connectionSlice";
import type { Message, Conversation } from "@/lib/types";

interface ConversationEvent {
  type: string;
  timestamp: string;
  conversation_id: string;
  data: any;
  metadata?: any;
}

interface ConversationSubscription {
  conversation_ids?: string[];
  agent_ids?: string[];
  message_types?: string[];
  include_typing?: boolean;
  include_system_messages?: boolean;
  include_metadata?: boolean;
}

interface UseConversationWebSocketOptions {
  autoConnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onEvent?: (event: ConversationEvent) => void;
  onError?: (error: Event) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

interface ConversationWebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastEventTime: Date | null;
  connectionStats: any;
}

export function useConversationWebSocket(
  options: UseConversationWebSocketOptions = {},
) {
  const {
    autoConnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onEvent,
    onError,
    onConnect,
    onDisconnect,
  } = options;

  const dispatch = useDispatch();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const subscriptionRef = useRef<ConversationSubscription>({});
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const connectionIdRef = useRef<string | null>(null);

  const [state, setState] = useState<ConversationWebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastEventTime: null,
    connectionStats: null,
  });

  // Get WebSocket URL - FIXED: Remove /api prefix
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    // Connect directly to backend WebSocket endpoint
    const wsHost = host.replace(":3000", ":8000"); // Use backend port
    return `${protocol}//${wsHost}/ws/conversations`;
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const now = new Date();

        setState((prev) => ({
          ...prev,
          lastEventTime: now,
          error: null,
        }));

        // Handle different message types
        switch (data.type) {
          case "connection_established":
            console.log("WebSocket connection established:", data.client_id);
            connectionIdRef.current = data.client_id;

            setState((prev) => ({
              ...prev,
              isConnected: true,
              isConnecting: false,
            }));

            // Update Redux state
            dispatch(setWebSocketStatus("connected"));
            dispatch(
              connectionEstablished({
                connectionId: data.client_id,
                socketUrl: getWebSocketUrl(),
                apiUrl: getWebSocketUrl()
                  .replace(/:\d+/, ":8000")
                  .replace("ws", "http"),
              }),
            );

            onConnect?.();
            break;

          case "pong":
            // Handle ping/pong for connection health and update latency
            if (data.latency) {
              dispatch(updateLatency(data.latency));
            }
            break;

          case "subscription_updated":
            console.log("Subscription updated:", data.subscription);
            break;

          case "connection_stats":
            setState((prev) => ({ ...prev, connectionStats: data.stats }));
            break;

          case "error":
            console.error("WebSocket error:", data.message);
            setState((prev) => ({ ...prev, error: data.message }));
            dispatch(
              addConnectionError({
                type: "websocket",
                message: data.message,
              }),
            );
            break;

          // Conversation events
          case "message_created":
          case "message_updated":
          case "message_deleted":
          case "conversation_started":
          case "conversation_ended":
          case "agent_typing":
          case "agent_stopped_typing":
          case "agent_joined":
          case "agent_left":
          case "message_queue_updated":
            onEvent?.(data as ConversationEvent);
            break;

          default:
            console.log("Unknown WebSocket message type:", data.type);
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        setState((prev) => ({ ...prev, error: "Failed to parse message" }));
        dispatch(
          addConnectionError({
            type: "websocket",
            message: "Failed to parse WebSocket message",
          }),
        );
      }
    },
    [onEvent, onConnect, dispatch, getWebSocketUrl],
  );

  // Handle connection errors
  const handleError = useCallback(
    (event: Event) => {
      console.error("WebSocket error:", event);
      setState((prev) => ({
        ...prev,
        error: "Connection error",
        isConnected: false,
        isConnecting: false,
      }));

      // Update Redux state
      dispatch(setWebSocketStatus("disconnected"));
      dispatch(
        addConnectionError({
          type: "websocket",
          message: "WebSocket connection error",
        }),
      );

      onError?.(event);
    },
    [onError, dispatch],
  );

  // Handle connection close
  const handleClose = useCallback(() => {
    console.log("WebSocket connection closed");
    setState((prev) => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
    }));

    // Update Redux state
    dispatch(setWebSocketStatus("disconnected"));
    dispatch(
      connectionLost({
        type: "websocket",
        error: "Connection closed",
      }),
    );

    // Clear ping interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    onDisconnect?.();

    // Attempt to reconnect if we haven't exceeded max attempts
    if (reconnectAttemptsRef.current < maxReconnectAttempts) {
      reconnectAttemptsRef.current++;
      console.log(
        `Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`,
      );

      // Update Redux state for reconnecting
      dispatch(setWebSocketStatus("connecting"));

      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, reconnectInterval);
    } else {
      setState((prev) => ({
        ...prev,
        error: "Max reconnection attempts exceeded",
      }));
      dispatch(
        addConnectionError({
          type: "websocket",
          message: "Max reconnection attempts exceeded",
        }),
      );
    }
  }, [onDisconnect, maxReconnectAttempts, reconnectInterval, dispatch]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setState((prev) => ({ ...prev, isConnecting: true, error: null }));
    dispatch(setWebSocketStatus("connecting"));

    try {
      const url = getWebSocketUrl();
      console.log("Connecting to WebSocket:", url);
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log("WebSocket connected");
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts

        // Set up ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          const startTime = Date.now();
          send({ type: "ping", clientTime: startTime });
        }, 30000); // Ping every 30 seconds
      };

      wsRef.current.onmessage = handleMessage;
      wsRef.current.onerror = handleError;
      wsRef.current.onclose = handleClose;
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
      setState((prev) => ({
        ...prev,
        error: "Failed to create connection",
        isConnecting: false,
      }));
      dispatch(setWebSocketStatus("disconnected"));
      dispatch(
        addConnectionError({
          type: "websocket",
          message: "Failed to create WebSocket connection",
        }),
      );
    }
  }, [getWebSocketUrl, handleMessage, handleError, handleClose, dispatch]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setState((prev) => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
    }));

    dispatch(setWebSocketStatus("disconnected"));
  }, [dispatch]);

  // Send message to WebSocket
  const send = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn("WebSocket not connected, cannot send message:", message);
      return false;
    }
  }, []);

  // Subscribe to conversation updates
  const subscribe = useCallback(
    (subscription: ConversationSubscription) => {
      subscriptionRef.current = { ...subscriptionRef.current, ...subscription };
      return send({
        type: "subscribe",
        subscription: subscriptionRef.current,
      });
    },
    [send],
  );

  // Update typing status
  const setTyping = useCallback(
    (conversationId: string, agentId: string, isTyping: boolean) => {
      return send({
        type: "set_typing",
        conversation_id: conversationId,
        agent_id: agentId,
        is_typing: isTyping,
      });
    },
    [send],
  );

  // Get typing status
  const getTypingStatus = useCallback(
    (conversationId: string) => {
      return send({
        type: "get_typing_status",
        conversation_id: conversationId,
      });
    },
    [send],
  );

  // Get connection stats
  const getStats = useCallback(() => {
    return send({ type: "get_stats" });
  }, [send]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Visibility change handler to reconnect when tab becomes visible
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden && !state.isConnected && !state.isConnecting) {
        console.log("Tab became visible, attempting to reconnect...");
        connect();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [state.isConnected, state.isConnecting, connect]);

  // Online/offline handler for better browser compatibility
  useEffect(() => {
    const handleOnline = () => {
      if (!state.isConnected && !state.isConnecting) {
        console.log("Browser came online, attempting to reconnect...");
        connect();
      }
    };

    const handleOffline = () => {
      console.log("Browser went offline");
      setState((prev) => ({ ...prev, error: "Browser offline" }));
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [state.isConnected, state.isConnecting, connect]);

  return {
    // State
    isConnected: state.isConnected,
    isConnecting: state.isConnecting,
    error: state.error,
    lastEventTime: state.lastEventTime,
    connectionStats: state.connectionStats,
    connectionId: connectionIdRef.current,

    // Methods
    connect,
    disconnect,
    send,
    subscribe,
    setTyping,
    getTypingStatus,
    getStats,

    // Connection info
    reconnectAttempts: reconnectAttemptsRef.current,
    maxReconnectAttempts,
  };
}
