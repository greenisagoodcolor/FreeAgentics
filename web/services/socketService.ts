import { io, Socket } from 'socket.io-client';
import { store } from '@/store';
import {
  connectionEstablished,
  connectionLost,
  setWebSocketStatus,
  updateLatency,
  incrementReconnectAttempt,
  addConnectionError,
} from '@/store/slices/connectionSlice';
import {
  addMessage,
  setTypingIndicators,
} from '@/store/slices/conversationSlice';
import {
  updateAgentStatus,
  setTypingAgents,
} from '@/store/slices/agentSlice';
import {
  addKnowledgeNode,
  addKnowledgeEdge,
  updateAgentKnowledge,
} from '@/store/slices/knowledgeSlice';

// Socket event types
export interface SocketEvents {
  // Connection events
  connect: () => void;
  disconnect: (reason: string) => void;
  connect_error: (error: Error) => void;
  reconnect: (attemptNumber: number) => void;
  reconnect_attempt: (attemptNumber: number) => void;
  reconnect_error: (error: Error) => void;
  reconnect_failed: () => void;
  
  // Custom events
  'connection:established': (data: { connectionId: string; serverTime: number }) => void;
  'ping:response': (data: { latency: number; serverTime: number }) => void;
  
  // Agent events
  'agent:status': (data: { agentId: string; status: string }) => void;
  'agent:typing': (data: { conversationId: string; agentIds: string[] }) => void;
  'agent:created': (data: { agent: any }) => void;
  'agent:updated': (data: { agentId: string; updates: any }) => void;
  
  // Message events
  'message:new': (data: { message: any }) => void;
  'message:queued': (data: { messageId: string; conversationId: string }) => void;
  'message:delivered': (data: { messageId: string }) => void;
  'message:failed': (data: { messageId: string; error: string }) => void;
  
  // Knowledge events
  'knowledge:node:added': (data: { node: any }) => void;
  'knowledge:edge:added': (data: { edge: any }) => void;
  'knowledge:agent:update': (data: { agentId: string; nodeIds: string[]; operation: 'add' | 'remove' }) => void;
  
  // Analytics events
  'analytics:update': (data: { metrics: any }) => void;
  'analytics:snapshot': (data: { snapshot: any }) => void;
}

class SocketService {
  private socket: Socket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;
  private connectionUrl: string = '';

  constructor() {
    // Initialize with environment variable or default
    this.connectionUrl = process.env.NEXT_PUBLIC_SOCKET_URL || 'http://localhost:8000';
  }

  connect(url?: string): void {
    if (url) {
      this.connectionUrl = url;
    }

    if (this.socket?.connected) {
      console.log('Socket already connected');
      return;
    }

    // Update connection status
    store.dispatch(setWebSocketStatus('connecting'));

    // Create socket connection
    this.socket = io(this.connectionUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
      autoConnect: true,
    });

    this.setupEventListeners();
    this.startPingInterval();
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('Socket connected');
      store.dispatch(setWebSocketStatus('connected'));
      
      // Request connection info
      this.socket?.emit('connection:request', {
        clientTime: Date.now(),
      });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      store.dispatch(connectionLost({ type: 'websocket', error: reason }));
      this.stopPingInterval();
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      store.dispatch(addConnectionError({
        type: 'websocket',
        message: error.message,
      }));
    });

    this.socket.on('reconnect_attempt', (attemptNumber) => {
      console.log('Reconnection attempt:', attemptNumber);
      store.dispatch(incrementReconnectAttempt());
    });

    // Custom events
    this.socket.on('connection:established', (data) => {
      store.dispatch(connectionEstablished({
        connectionId: data.connectionId,
        socketUrl: this.connectionUrl,
        apiUrl: this.connectionUrl.replace(/:\d+$/, ':8000'), // Assume API on same host
      }));
    });

    this.socket.on('ping:response', (data) => {
      store.dispatch(updateLatency(data.latency));
    });

    // Agent events
    this.socket.on('agent:status', (data) => {
      store.dispatch(updateAgentStatus({
        agentId: data.agentId,
        status: data.status as any,
      }));
    });

    this.socket.on('agent:typing', (data) => {
      store.dispatch(setTypingAgents(data.agentIds));
      store.dispatch(setTypingIndicators(data));
    });

    // Message events
    this.socket.on('message:new', (data) => {
      store.dispatch(addMessage(data.message));
    });

    // Knowledge events
    this.socket.on('knowledge:node:added', (data) => {
      store.dispatch(addKnowledgeNode(data.node));
    });

    this.socket.on('knowledge:edge:added', (data) => {
      store.dispatch(addKnowledgeEdge(data.edge));
    });

    this.socket.on('knowledge:agent:update', (data) => {
      store.dispatch(updateAgentKnowledge(data));
    });
  }

  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (this.socket?.connected) {
        const startTime = Date.now();
        this.socket.emit('ping', { clientTime: startTime });
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
    
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // Emit methods
  emit(event: string, data?: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn(`Cannot emit ${event}: Socket not connected`);
    }
  }

  // Specific emit methods for type safety
  sendMessage(conversationId: string, content: string, agentId: string): void {
    this.emit('message:send', {
      conversationId,
      content,
      agentId,
      timestamp: Date.now(),
    });
  }

  createAgent(templateId: string, name?: string): void {
    this.emit('agent:create', {
      templateId,
      name,
      timestamp: Date.now(),
    });
  }

  updateAgentParameters(agentId: string, parameters: any): void {
    this.emit('agent:update:parameters', {
      agentId,
      parameters,
      timestamp: Date.now(),
    });
  }

  startConversation(type: string, participants: string[]): void {
    this.emit('conversation:start', {
      type,
      participants,
      timestamp: Date.now(),
    });
  }

  subscribeToAgent(agentId: string): void {
    this.emit('agent:subscribe', { agentId });
  }

  unsubscribeFromAgent(agentId: string): void {
    this.emit('agent:unsubscribe', { agentId });
  }

  subscribeToConversation(conversationId: string): void {
    this.emit('conversation:subscribe', { conversationId });
  }

  unsubscribeFromConversation(conversationId: string): void {
    this.emit('conversation:unsubscribe', { conversationId });
  }

  // Singleton instance
  private static instance: SocketService;

  static getInstance(): SocketService {
    if (!SocketService.instance) {
      SocketService.instance = new SocketService();
    }
    return SocketService.instance;
  }
}

export const socketService = SocketService.getInstance(); 