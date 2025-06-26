"use client";

import { useEffect, useRef, useState, useCallback } from 'react';
import type { Message, Conversation } from '@/lib/types';

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
  options: UseConversationWebSocketOptions = {}
) {
  const {
    autoConnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onEvent,
    onError,
    onConnect,
    onDisconnect
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const subscriptionRef = useRef<ConversationSubscription>({});
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [state, setState] = useState<ConversationWebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastEventTime: null,
    connectionStats: null
  });

  // Get WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/api/ws/conversations`;
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      
      setState(prev => ({
        ...prev,
        lastEventTime: new Date(),
        error: null
      }));

      // Handle different message types
      switch (data.type) {
        case 'connection_established':
          console.log('WebSocket connection established:', data.client_id);
          setState(prev => ({ ...prev, isConnected: true, isConnecting: false }));
          onConnect?.();
          break;

        case 'pong':
          // Handle ping/pong for connection health
          break;

        case 'subscription_updated':
          console.log('Subscription updated:', data.subscription);
          break;

        case 'connection_stats':
          setState(prev => ({ ...prev, connectionStats: data.stats }));
          break;

        case 'error':
          console.error('WebSocket error:', data.message);
          setState(prev => ({ ...prev, error: data.message }));
          break;

        // Conversation events
        case 'message_created':
        case 'message_updated':
        case 'message_deleted':
        case 'conversation_started':
        case 'conversation_ended':
        case 'agent_typing':
        case 'agent_stopped_typing':
        case 'agent_joined':
        case 'agent_left':
        case 'message_queue_updated':
          onEvent?.(data as ConversationEvent);
          break;

        default:
          console.log('Unknown WebSocket message type:', data.type);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
      setState(prev => ({ ...prev, error: 'Failed to parse message' }));
    }
  }, [onEvent, onConnect]);

  // Handle connection errors
  const handleError = useCallback((event: Event) => {
    console.error('WebSocket error:', event);
    setState(prev => ({ 
      ...prev, 
      error: 'Connection error',
      isConnected: false,
      isConnecting: false 
    }));
    onError?.(event);
  }, [onError]);

  // Handle connection close
  const handleClose = useCallback(() => {
    console.log('WebSocket connection closed');
    setState(prev => ({ 
      ...prev, 
      isConnected: false, 
      isConnecting: false 
    }));
    
    // Clear ping interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    
    onDisconnect?.();
    
    // Attempt to reconnect if we haven't exceeded max attempts
    if (reconnectAttemptsRef.current < maxReconnectAttempts) {
      reconnectAttemptsRef.current++;
      console.log(`Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
      
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, reconnectInterval);
    } else {
      setState(prev => ({ 
        ...prev, 
        error: 'Max reconnection attempts exceeded' 
      }));
    }
  }, [onDisconnect, maxReconnectAttempts, reconnectInterval]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setState(prev => ({ ...prev, isConnecting: true, error: null }));

    try {
      const url = getWebSocketUrl();
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts
        
        // Set up ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          send({ type: 'ping' });
        }, 30000); // Ping every 30 seconds
      };

      wsRef.current.onmessage = handleMessage;
      wsRef.current.onerror = handleError;
      wsRef.current.onclose = handleClose;

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setState(prev => ({ 
        ...prev, 
        error: 'Failed to create connection',
        isConnecting: false 
      }));
    }
  }, [getWebSocketUrl, handleMessage, handleError, handleClose]);

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

    setState(prev => ({ 
      ...prev, 
      isConnected: false, 
      isConnecting: false 
    }));
  }, []);

  // Send message to WebSocket
  const send = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
      return false;
    }
  }, []);

  // Subscribe to conversation updates
  const subscribe = useCallback((subscription: ConversationSubscription) => {
    subscriptionRef.current = { ...subscriptionRef.current, ...subscription };
    return send({
      type: 'subscribe',
      subscription: subscriptionRef.current
    });
  }, [send]);

  // Update typing status
  const setTyping = useCallback((conversationId: string, agentId: string, isTyping: boolean) => {
    return send({
      type: 'set_typing',
      conversation_id: conversationId,
      agent_id: agentId,
      is_typing: isTyping
    });
  }, [send]);

  // Get typing status
  const getTypingStatus = useCallback((conversationId: string) => {
    return send({
      type: 'get_typing_status',
      conversation_id: conversationId
    });
  }, [send]);

  // Get connection stats
  const getStats = useCallback(() => {
    return send({ type: 'get_stats' });
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
        console.log('Tab became visible, attempting to reconnect...');
        connect();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [state.isConnected, state.isConnecting, connect]);

  return {
    // State
    isConnected: state.isConnected,
    isConnecting: state.isConnecting,
    error: state.error,
    lastEventTime: state.lastEventTime,
    connectionStats: state.connectionStats,

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
    maxReconnectAttempts
  };
} 