import { useState, useEffect, useCallback, useRef } from "react";

export interface MarkovBlanketEvent {
  type: string;
  timestamp: string;
  agent_id: string;
  data: any;
  severity: "info" | "warning" | "error" | "critical";
  metadata?: any;
}

export interface MarkovBlanketSubscription {
  agent_ids?: string[];
  event_types?: string[];
  severity_levels?: string[];
  include_mathematical_proofs?: boolean;
  include_detailed_metrics?: boolean;
  violation_alerts_only?: boolean;
  real_time_updates?: boolean;
}

export interface BoundaryViolation {
  agent_id: string;
  violation_type: string;
  independence_measure: number;
  threshold: number;
  mathematical_justification: string;
  evidence: any;
  severity: string;
  timestamp: string;
}

export interface MonitoringStatus {
  monitoring_active: boolean;
  monitored_agents: string[];
  total_violations: number;
  system_uptime: number;
  last_check: string;
}

export interface ConnectionStats {
  total_connections: number;
  total_events_sent: number;
  active_violations: number;
  monitored_agents: number;
  system_uptime: number;
  connections: Array<{
    client_id: string;
    connected_at: string;
    events_sent: number;
    subscribed_agents: number;
  }>;
}

export interface UseMarkovBlanketWebSocketOptions {
  autoConnect?: boolean;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  subscription?: MarkovBlanketSubscription;
  onEvent?: (event: MarkovBlanketEvent) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: string) => void;
  onViolation?: (violation: BoundaryViolation) => void;
}

export interface UseMarkovBlanketWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastEventTime: Date | null;
  connectionStats: ConnectionStats | null;
  monitoringStatus: MonitoringStatus | null;
  violations: BoundaryViolation[];

  // Connection management
  connect: () => void;
  disconnect: () => void;

  // Subscription management
  updateSubscription: (subscription: MarkovBlanketSubscription) => void;

  // Agent management
  registerAgent: (agentId: string) => void;
  unregisterAgent: (agentId: string) => void;

  // Monitoring control
  startMonitoring: () => void;
  stopMonitoring: () => void;

  // Data fetching
  getMonitoringStatus: () => void;
  getAgentViolations: (agentId: string) => void;
  getConnectionStats: () => void;
  getComplianceReport: (agentId?: string) => void;

  // Utility
  sendMessage: (message: any) => void;
  ping: () => void;
}

export function useMarkovBlanketWebSocket(
  options: UseMarkovBlanketWebSocketOptions = {},
): UseMarkovBlanketWebSocketReturn {
  const {
    autoConnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 5,
    subscription,
    onEvent,
    onConnect,
    onDisconnect,
    onError,
    onViolation,
  } = options;

  const [state, setState] = useState({
    isConnected: false,
    isConnecting: false,
    error: null as string | null,
    lastEventTime: null as Date | null,
    connectionStats: null as ConnectionStats | null,
    monitoringStatus: null as MonitoringStatus | null,
    violations: [] as BoundaryViolation[],
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Handle incoming messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);

        setState((prev) => ({
          ...prev,
          lastEventTime: new Date(),
          error: null,
        }));

        // Handle different message types
        switch (data.type) {
          case "connection_established":
            console.log(
              "Markov Blanket WebSocket connection established:",
              data.client_id,
            );
            setState((prev) => ({
              ...prev,
              isConnected: true,
              isConnecting: false,
            }));
            onConnect?.();
            break;

          case "pong":
            // Handle ping/pong for connection health
            break;

          case "subscription_updated":
            console.log(
              "Markov Blanket subscription updated:",
              data.subscription,
            );
            break;

          case "monitoring_status":
            setState((prev) => ({ ...prev, monitoringStatus: data.data }));
            break;

          case "connection_stats":
            setState((prev) => ({ ...prev, connectionStats: data.stats }));
            break;

          case "agent_violations":
            setState((prev) => ({
              ...prev,
              violations: [...prev.violations, ...data.violations],
            }));
            break;

          case "compliance_report":
            console.log(
              "Compliance report received for agent:",
              data.agent_id,
              data.report,
            );
            break;

          case "error":
            console.error("Markov Blanket WebSocket error:", data.message);
            setState((prev) => ({ ...prev, error: data.message }));
            onError?.(data.message);
            break;

          // Monitoring events
          case "boundary_violation":
            const violation: BoundaryViolation = {
              agent_id: data.agent_id,
              violation_type: data.data.violation_type,
              independence_measure: data.data.independence_measure,
              threshold: data.data.threshold,
              mathematical_justification: data.data.mathematical_justification,
              evidence: data.data.evidence,
              severity: data.severity,
              timestamp: data.timestamp,
            };
            setState((prev) => ({
              ...prev,
              violations: [...prev.violations, violation],
            }));
            onViolation?.(violation);
            onEvent?.(data as MarkovBlanketEvent);
            break;

          case "state_update":
          case "agent_registered":
          case "agent_unregistered":
          case "monitoring_started":
          case "monitoring_stopped":
          case "threshold_breach":
          case "integrity_update":
          case "monitoring_error":
            onEvent?.(data as MarkovBlanketEvent);
            break;

          default:
            console.log(
              "Unknown Markov Blanket WebSocket message type:",
              data.type,
            );
        }
      } catch (error) {
        console.error("Error parsing Markov Blanket WebSocket message:", error);
        setState((prev) => ({ ...prev, error: "Failed to parse message" }));
        onError?.("Failed to parse message");
      }
    },
    [onEvent, onConnect, onError, onViolation],
  );

  // Handle connection open
  const handleOpen = useCallback(() => {
    console.log("Markov Blanket WebSocket connection opened");
    reconnectAttemptsRef.current = 0;

    setState((prev) => ({
      ...prev,
      isConnected: true,
      isConnecting: false,
      error: null,
    }));

    // Send initial subscription if provided
    if (subscription) {
      setTimeout(() => {
        updateSubscription(subscription);
      }, 100);
    }
  }, [subscription]);

  // Handle connection close
  const handleClose = useCallback(
    (event: CloseEvent) => {
      console.log(
        "Markov Blanket WebSocket connection closed:",
        event.code,
        event.reason,
      );

      setState((prev) => ({
        ...prev,
        isConnected: false,
        isConnecting: false,
      }));

      onDisconnect?.();

      // Attempt to reconnect if not a manual disconnect
      if (
        event.code !== 1000 &&
        reconnectAttemptsRef.current < maxReconnectAttempts
      ) {
        reconnectAttemptsRef.current++;
        console.log(
          `Attempting to reconnect Markov Blanket WebSocket (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`,
        );

        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectDelay);
      }
    },
    [maxReconnectAttempts, reconnectDelay, onDisconnect],
  );

  // Handle connection error
  const handleError = useCallback(
    (event: Event) => {
      console.error("Markov Blanket WebSocket error:", event);
      setState((prev) => ({
        ...prev,
        error: "Connection error",
        isConnecting: false,
      }));
      onError?.("Connection error");
    },
    [onError],
  );

  // Connect function
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setState((prev) => ({ ...prev, isConnecting: true, error: null }));

    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/api/ws/markov-blanket`;

      wsRef.current = new WebSocket(wsUrl);
      wsRef.current.onopen = handleOpen;
      wsRef.current.onmessage = handleMessage;
      wsRef.current.onclose = handleClose;
      wsRef.current.onerror = handleError;
    } catch (error) {
      console.error(
        "Error creating Markov Blanket WebSocket connection:",
        error,
      );
      setState((prev) => ({
        ...prev,
        error: "Failed to create connection",
        isConnecting: false,
      }));
      onError?.("Failed to create connection");
    }
  }, [handleOpen, handleMessage, handleClose, handleError, onError]);

  // Disconnect function
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, "Manual disconnect");
      wsRef.current = null;
    }

    setState((prev) => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
    }));
  }, []);

  // Send message function
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn(
        "Markov Blanket WebSocket not connected, cannot send message:",
        message,
      );
    }
  }, []);

  // Subscription management
  const updateSubscription = useCallback(
    (newSubscription: MarkovBlanketSubscription) => {
      sendMessage({
        type: "subscribe",
        subscription: newSubscription,
      });
    },
    [sendMessage],
  );

  // Agent management
  const registerAgent = useCallback(
    (agentId: string) => {
      sendMessage({
        type: "register_agent",
        agent_id: agentId,
      });
    },
    [sendMessage],
  );

  const unregisterAgent = useCallback(
    (agentId: string) => {
      sendMessage({
        type: "unregister_agent",
        agent_id: agentId,
      });
    },
    [sendMessage],
  );

  // Monitoring control
  const startMonitoring = useCallback(() => {
    sendMessage({ type: "start_monitoring" });
  }, [sendMessage]);

  const stopMonitoring = useCallback(() => {
    sendMessage({ type: "stop_monitoring" });
  }, [sendMessage]);

  // Data fetching
  const getMonitoringStatus = useCallback(() => {
    sendMessage({ type: "get_monitoring_status" });
  }, [sendMessage]);

  const getAgentViolations = useCallback(
    (agentId: string) => {
      sendMessage({
        type: "get_agent_violations",
        agent_id: agentId,
      });
    },
    [sendMessage],
  );

  const getConnectionStats = useCallback(() => {
    sendMessage({ type: "get_stats" });
  }, [sendMessage]);

  const getComplianceReport = useCallback(
    (agentId?: string) => {
      sendMessage({
        type: "get_compliance_report",
        agent_id: agentId,
      });
    },
    [sendMessage],
  );

  // Ping function
  const ping = useCallback(() => {
    sendMessage({ type: "ping" });
  }, [sendMessage]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    updateSubscription,
    registerAgent,
    unregisterAgent,
    startMonitoring,
    stopMonitoring,
    getMonitoringStatus,
    getAgentViolations,
    getConnectionStats,
    getComplianceReport,
    sendMessage,
    ping,
  };
}
