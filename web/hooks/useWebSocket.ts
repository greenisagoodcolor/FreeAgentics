"use client";

import { useEffect, useCallback } from "react";
import { useSelector, useDispatch } from "react-redux";
import { RootState } from "@/store";
import { socketService } from "@/services/socketService";

export interface UseWebSocketReturn {
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  connectionId: string | null;
  latency: number | null;
  reconnectAttempts: number;
  error: string | null;

  // Connection methods
  connect: () => void;
  disconnect: () => void;

  // Communication methods
  send: (message: any) => boolean;
  sendMessage: (
    conversationId: string,
    content: string,
    agentId: string,
  ) => void;

  // Subscription methods
  subscribeToAgent: (agentId: string) => void;
  unsubscribeFromAgent: (agentId: string) => void;
  subscribeToConversation: (conversationId: string) => void;
  unsubscribeFromConversation: (conversationId: string) => void;

  // Interaction methods
  setTyping: (
    conversationId: string,
    agentId: string,
    isTyping: boolean,
  ) => void;
  getConnectionStats: () => void;
}

export interface UseWebSocketOptions {
  autoConnect?: boolean;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: string) => void;
}

export function useWebSocket(
  urlOrOptions?: string | UseWebSocketOptions,
): UseWebSocketReturn {
  // Handle legacy URL parameter or new options object
  const options: UseWebSocketOptions =
    typeof urlOrOptions === "string"
      ? { autoConnect: true }
      : urlOrOptions || {};
  const { autoConnect = true, onConnect, onDisconnect, onError } = options;

  // Get connection state from Redux
  const connectionState = useSelector((state: RootState) => state.connection);

  // Connection methods
  const connect = useCallback(() => {
    socketService.connect();
  }, []);

  const disconnect = useCallback(() => {
    socketService.disconnect();
  }, []);

  // Communication methods
  const send = useCallback((message: any) => {
    return socketService.send(message);
  }, []);

  const sendMessage = useCallback(
    (conversationId: string, content: string, agentId: string) => {
      socketService.sendMessage(conversationId, content, agentId);
    },
    [],
  );

  // Subscription methods
  const subscribeToAgent = useCallback((agentId: string) => {
    socketService.subscribeToAgent(agentId);
  }, []);

  const unsubscribeFromAgent = useCallback((agentId: string) => {
    socketService.unsubscribeFromAgent(agentId);
  }, []);

  const subscribeToConversation = useCallback((conversationId: string) => {
    socketService.subscribeToConversation(conversationId);
  }, []);

  const unsubscribeFromConversation = useCallback((conversationId: string) => {
    socketService.unsubscribeFromConversation(conversationId);
  }, []);

  // Interaction methods
  const setTyping = useCallback(
    (conversationId: string, agentId: string, isTyping: boolean) => {
      socketService.setTyping(conversationId, agentId, isTyping);
    },
    [],
  );

  const getConnectionStats = useCallback(() => {
    socketService.getConnectionStats();
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && connectionState.status.websocket !== "connected") {
      connect();
    }
  }, [autoConnect, connect, connectionState.status.websocket]);

  // Connection status callbacks
  useEffect(() => {
    if (connectionState.status.websocket === "connected" && onConnect) {
      onConnect();
    }
  }, [connectionState.status.websocket, onConnect]);

  useEffect(() => {
    if (connectionState.status.websocket === "disconnected" && onDisconnect) {
      onDisconnect();
    }
  }, [connectionState.status.websocket, onDisconnect]);

  useEffect(() => {
    if (connectionState.errors.length > 0 && onError) {
      const latestError =
        connectionState.errors[connectionState.errors.length - 1];
      onError(latestError.message);
    }
  }, [connectionState.errors, onError]);

  return {
    // Connection state from Redux
    isConnected: connectionState.status.websocket === "connected",
    isConnecting: connectionState.status.websocket === "connecting",
    connectionId: connectionState.connectionId,
    latency: connectionState.status.latency,
    reconnectAttempts: connectionState.status.reconnectAttempts,
    error:
      connectionState.errors.length > 0
        ? connectionState.errors[connectionState.errors.length - 1].message
        : null,

    // Methods
    connect,
    disconnect,
    send,
    sendMessage,
    subscribeToAgent,
    unsubscribeFromAgent,
    subscribeToConversation,
    unsubscribeFromConversation,
    setTyping,
    getConnectionStats,
  };
}
