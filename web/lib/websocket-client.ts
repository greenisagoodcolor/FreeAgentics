export type WebSocketState = "connecting" | "connected" | "disconnected" | "error";

export interface WebSocketMessage {
  type: string;
  data: unknown;
}

export interface WebSocketOptions {
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private options: Required<WebSocketOptions>;
  private reconnectAttempts = 0;
  private messageHandlers = new Map<string, Set<(data: unknown) => void>>();
  private stateChangeHandlers = new Set<(state: WebSocketState) => void>();
  private currentState: WebSocketState = "disconnected";

  constructor(url: string, options: WebSocketOptions = {}) {
    this.url = url;
    this.options = {
      reconnect: true,
      reconnectInterval: 3000,
      maxReconnectAttempts: 5,
      ...options,
    };
  }

  connect(): void {
    // Skip WebSocket operations during SSR
    if (typeof window === "undefined") {
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.setState("connecting");

    try {
      // Append token to WebSocket URL if available
      let wsUrl = this.url;
      const token = localStorage.getItem("freeagentics_auth_token");
      if (token) {
        const separator = this.url.includes('?') ? '&' : '?';
        wsUrl = `${this.url}${separator}token=${encodeURIComponent(token)}`;
      }
      
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.setState("connected");
      };

      this.ws.onclose = () => {
        this.setState("disconnected");
        this.handleReconnect();
      };

      this.ws.onerror = () => {
        this.setState("error");
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          this.handleMessage(message);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };
    } catch (error) {
      this.setState("error");
      this.handleReconnect();
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.options.reconnect = false;
      this.ws.close();
      this.ws = null;
    }
  }

  send(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket is not connected");
    }
  }

  on(type: string, handler: (data: unknown) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    this.messageHandlers.get(type)!.add(handler);

    return () => {
      const handlers = this.messageHandlers.get(type);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.messageHandlers.delete(type);
        }
      }
    };
  }

  onStateChange(handler: (state: WebSocketState) => void): () => void {
    this.stateChangeHandlers.add(handler);
    return () => {
      this.stateChangeHandlers.delete(handler);
    };
  }

  getState(): WebSocketState {
    return this.currentState;
  }

  private setState(state: WebSocketState): void {
    this.currentState = state;
    this.stateChangeHandlers.forEach((handler) => handler(state));
  }

  private handleMessage(message: WebSocketMessage): void {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message.data));
    }
  }

  private handleReconnect(): void {
    if (this.options.reconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.connect();
      }, this.options.reconnectInterval);
    }
  }
}

// Singleton instance
let wsClient: WebSocketClient | null = null;

export function getWebSocketClient(url?: string): WebSocketClient {
  if (!wsClient && url) {
    wsClient = new WebSocketClient(url);
  }
  if (!wsClient) {
    throw new Error("WebSocket client not initialized");
  }
  return wsClient;
}

export function initializeWebSocket(url: string, options?: WebSocketOptions): WebSocketClient {
  wsClient = new WebSocketClient(url, options);
  return wsClient;
}

export function resetWebSocketClient(): void {
  if (wsClient) {
    wsClient.disconnect();
  }
  wsClient = null;
}
