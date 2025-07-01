import { store } from "@/store";
import {
  connectionEstablished,
  connectionLost,
  setWebSocketStatus,
  updateLatency,
  incrementReconnectAttempt,
  addConnectionError,
} from "@/store/slices/connectionSlice";
import {
  addMessage,
  setTypingIndicators,
} from "@/store/slices/conversationSlice";
import { updateAgentStatus, setTypingAgents } from "@/store/slices/agentSlice";
import {
  addKnowledgeNode,
  addKnowledgeEdge,
  updateAgentKnowledge,
} from "@/store/slices/knowledgeSlice";

// WebSocket event types for type safety
export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

class UnifiedWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;
  private connectionUrl: string = "";
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectInterval: number = 3000;
  private isConnecting: boolean = false;
  private subscriptions: Set<string> = new Set();

  constructor() {
    // Initialize with proper WebSocket URL (backend port)
    this.connectionUrl = this.getWebSocketUrl();
  }

  private getWebSocketUrl(): string {
    if (typeof window === "undefined") return "";

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    // Connect directly to backend WebSocket endpoint
    const wsHost = host.replace(":3000", ":8000"); // Use backend port
    return `${protocol}//${wsHost}/ws/conversations`;
  }

  connect(url?: string): void {
    if (url) {
      this.connectionUrl = url;
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log("WebSocket already connected");
      return;
    }

    if (this.isConnecting) {
      console.log("WebSocket connection already in progress");
      return;
    }

    this.isConnecting = true;
    store.dispatch(setWebSocketStatus("connecting"));

    try {
      console.log("Connecting to WebSocket:", this.connectionUrl);
      this.ws = new WebSocket(this.connectionUrl);

      this.ws.onopen = () => {
        console.log("WebSocket connected successfully");
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        store.dispatch(setWebSocketStatus("connected"));

        // Request connection info
        this.send({
          type: "ping",
          clientTime: Date.now(),
        });

        this.startPingInterval();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event);
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.isConnecting = false;

        store.dispatch(
          addConnectionError({
            type: "websocket",
            message: "WebSocket connection error",
          }),
        );
      };

      this.ws.onclose = () => {
        console.log("WebSocket connection closed");
        this.isConnecting = false;

        store.dispatch(setWebSocketStatus("disconnected"));
        store.dispatch(
          connectionLost({
            type: "websocket",
            error: "Connection closed",
          }),
        );

        this.stopPingInterval();
        this.handleReconnect();
      };
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
      this.isConnecting = false;

      store.dispatch(setWebSocketStatus("disconnected"));
      store.dispatch(
        addConnectionError({
          type: "websocket",
          message: "Failed to create WebSocket connection",
        }),
      );
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);

      // Handle different message types
      switch (data.type) {
        case "connection_established":
          console.log("WebSocket connection established:", data.client_id);
          store.dispatch(
            connectionEstablished({
              connectionId: data.client_id,
              socketUrl: this.connectionUrl,
              apiUrl: this.connectionUrl
                .replace(/:\d+/, ":8000")
                .replace("ws", "http"),
            }),
          );
          break;

        case "pong":
          // Handle ping/pong for connection health
          if (data.clientTime) {
            const latency = Date.now() - data.clientTime;
            store.dispatch(updateLatency(latency));
          }
          break;

        case "error":
          console.error("WebSocket error:", data.message);
          store.dispatch(
            addConnectionError({
              type: "websocket",
              message: data.message,
            }),
          );
          break;

        // Agent events
        case "agent_status":
        case "agent:status":
          store.dispatch(
            updateAgentStatus({
              agentId: data.agentId || data.agent_id,
              status: data.status,
            }),
          );
          break;

        case "agent_typing":
        case "agent:typing":
          if (data.agentIds) {
            store.dispatch(setTypingAgents(data.agentIds));
          }
          if (data.conversationId) {
            store.dispatch(setTypingIndicators(data));
          }
          break;

        // Message events
        case "message_new":
        case "message:new":
        case "message_created":
          if (data.message) {
            store.dispatch(addMessage(data.message));
          }
          break;

        // Knowledge events
        case "knowledge_node_added":
        case "knowledge:node:added":
          if (data.node) {
            store.dispatch(addKnowledgeNode(data.node));
          }
          break;

        case "knowledge_edge_added":
        case "knowledge:edge:added":
          if (data.edge) {
            store.dispatch(addKnowledgeEdge(data.edge));
          }
          break;

        case "knowledge_agent_update":
        case "knowledge:agent:update":
          store.dispatch(updateAgentKnowledge(data));
          break;

        default:
          console.log("Unknown WebSocket message type:", data.type, data);
      }
    } catch (error) {
      console.error("Error parsing WebSocket message:", error);
      store.dispatch(
        addConnectionError({
          type: "websocket",
          message: "Failed to parse WebSocket message",
        }),
      );
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("Max reconnection attempts exceeded");
      store.dispatch(
        addConnectionError({
          type: "websocket",
          message: "Max reconnection attempts exceeded",
        }),
      );
      return;
    }

    this.reconnectAttempts++;
    console.log(
      `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`,
    );

    store.dispatch(incrementReconnectAttempt());

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, this.reconnectInterval);
  }

  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({
          type: "ping",
          clientTime: Date.now(),
        });
      }
    }, 30000); // Every 30 seconds
  }

  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  disconnect(): void {
    this.stopPingInterval();

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnecting = false;
    this.reconnectAttempts = 0;
    store.dispatch(setWebSocketStatus("disconnected"));
  }

  // Send methods
  send(message: WebSocketMessage): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      return true;
    } else {
      console.warn("Cannot send message: WebSocket not connected", message);
      return false;
    }
  }

  // Specific methods for different use cases
  sendMessage(conversationId: string, content: string, agentId: string): void {
    this.send({
      type: "message_send",
      conversation_id: conversationId,
      content,
      agent_id: agentId,
      timestamp: Date.now(),
    });
  }

  createAgent(templateId: string, name?: string): void {
    this.send({
      type: "agent_create",
      template_id: templateId,
      name,
      timestamp: Date.now(),
    });
  }

  updateAgentParameters(agentId: string, parameters: any): void {
    this.send({
      type: "agent_update_parameters",
      agent_id: agentId,
      parameters,
      timestamp: Date.now(),
    });
  }

  startConversation(type: string, participants: string[]): void {
    this.send({
      type: "conversation_start",
      conversation_type: type,
      participants,
      timestamp: Date.now(),
    });
  }

  subscribeToAgent(agentId: string): void {
    this.subscriptions.add(`agent:${agentId}`);
    this.send({
      type: "subscribe",
      subscription: {
        agent_ids: [agentId],
      },
    });
  }

  unsubscribeFromAgent(agentId: string): void {
    this.subscriptions.delete(`agent:${agentId}`);
    this.send({
      type: "unsubscribe",
      subscription: {
        agent_ids: [agentId],
      },
    });
  }

  subscribeToConversation(conversationId: string): void {
    this.subscriptions.add(`conversation:${conversationId}`);
    this.send({
      type: "subscribe",
      subscription: {
        conversation_ids: [conversationId],
      },
    });
  }

  unsubscribeFromConversation(conversationId: string): void {
    this.subscriptions.delete(`conversation:${conversationId}`);
    this.send({
      type: "unsubscribe",
      subscription: {
        conversation_ids: [conversationId],
      },
    });
  }

  setTyping(conversationId: string, agentId: string, isTyping: boolean): void {
    this.send({
      type: "set_typing",
      conversation_id: conversationId,
      agent_id: agentId,
      is_typing: isTyping,
    });
  }

  getConnectionStats(): void {
    this.send({
      type: "get_stats",
    });
  }

  // Connection status methods
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getIsConnecting(): boolean {
    return this.isConnecting;
  }

  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  // Singleton instance
  private static instance: UnifiedWebSocketService;

  static getInstance(): UnifiedWebSocketService {
    if (!UnifiedWebSocketService.instance) {
      UnifiedWebSocketService.instance = new UnifiedWebSocketService();
    }
    return UnifiedWebSocketService.instance;
  }
}

export const socketService = UnifiedWebSocketService.getInstance();
