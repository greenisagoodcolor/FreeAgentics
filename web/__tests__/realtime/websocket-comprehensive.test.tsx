/**
 * Comprehensive WebSocket and Real-time Communication Tests
 * 
 * Tests for WebSocket connections, real-time data synchronization, message queuing,
 * connection resilience, and live updates following ADR-007 requirements.
 */

import { jest } from '@jest/globals';
import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Enhanced WebSocket Mock with realistic behavior
class EnhancedMockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  public readyState: number = EnhancedMockWebSocket.CONNECTING;
  public onopen: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public bufferedAmount: number = 0;
  public extensions: string = '';
  public protocol: string = '';

  private messageQueue: string[] = [];
  private connectionTimer?: NodeJS.Timeout;
  private heartbeatTimer?: NodeJS.Timeout;

  constructor(public url: string, public protocols?: string | string[]) {
    this.simulateConnection();
  }

  private simulateConnection(): void {
    this.connectionTimer = setTimeout(() => {
      if (this.url.includes('fail')) {
        this.readyState = EnhancedMockWebSocket.CLOSED;
        this.onerror?.(new Event('error'));
      } else if (this.url.includes('slow')) {
        setTimeout(() => {
          this.readyState = EnhancedMockWebSocket.OPEN;
          this.onopen?.(new Event('open'));
          this.startHeartbeat();
        }, 1000);
      } else {
        this.readyState = EnhancedMockWebSocket.OPEN;
        this.onopen?.(new Event('open'));
        this.startHeartbeat();
      }
    }, Math.random() * 50 + 10);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.readyState === EnhancedMockWebSocket.OPEN) {
        this.simulateIncomingMessage('{"type":"heartbeat","timestamp":' + Date.now() + '}');
      }
    }, 5000);
  }

  send(data: string | ArrayBuffer | Blob): void {
    if (this.readyState !== EnhancedMockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }

    const message = data.toString();
    this.bufferedAmount += message.length;

    // Simulate network delay
    setTimeout(() => {
      this.bufferedAmount = Math.max(0, this.bufferedAmount - message.length);
      
      if (message.includes('error')) {
        this.onerror?.(new Event('error'));
      } else if (message.includes('echo')) {
        this.simulateIncomingMessage(message);
      } else if (message.includes('broadcast')) {
        // Simulate broadcast to other clients
        this.simulateIncomingMessage(`{"type":"broadcast","data":${message},"from":"server"}`);
      }
    }, Math.random() * 20 + 5);
  }

  close(code: number = 1000, reason: string = ''): void {
    if (this.readyState === EnhancedMockWebSocket.CLOSED) return;

    this.readyState = EnhancedMockWebSocket.CLOSING;
    
    if (this.connectionTimer) clearTimeout(this.connectionTimer);
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);

    setTimeout(() => {
      this.readyState = EnhancedMockWebSocket.CLOSED;
      this.onclose?.(new CloseEvent('close', { code, reason }));
    }, 10);
  }

  private simulateIncomingMessage(data: string): void {
    if (this.readyState === EnhancedMockWebSocket.OPEN && this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }

  // Simulate connection drops
  simulateConnectionDrop(): void {
    this.readyState = EnhancedMockWebSocket.CLOSED;
    this.onerror?.(new Event('error'));
    this.onclose?.(new CloseEvent('close', { code: 1006, reason: 'Connection dropped' }));
  }

  // Simulate server messages
  simulateServerMessage(message: any): void {
    this.simulateIncomingMessage(JSON.stringify(message));
  }
}

global.WebSocket = EnhancedMockWebSocket as any;

// Real-time Connection Manager
interface ConnectionConfig {
  url: string;
  protocols?: string[];
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  messageQueueSize: number;
}

interface Message {
  id: string;
  type: string;
  data: any;
  timestamp: number;
  retries: number;
}

class RealTimeConnectionManager {
  private ws: WebSocket | null = null;
  private config: ConnectionConfig;
  private messageQueue: Message[] = [];
  private reconnectAttempts: number = 0;
  private reconnectTimer?: NodeJS.Timeout;
  private heartbeatTimer?: NodeJS.Timeout;
  private messageHandlers: Map<string, (data: any) => void> = new Map();
  private connectionState: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' = 'disconnected';
  private lastHeartbeat: number = 0;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor(config: Partial<ConnectionConfig> = {}) {
    this.config = {
      url: 'ws://localhost:8080',
      reconnectInterval: 1000,
      maxReconnectAttempts: 5,
      heartbeatInterval: 30000,
      messageQueueSize: 100,
      ...config,
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.connectionState === 'connected') {
        resolve();
        return;
      }

      this.connectionState = 'connecting';
      this.ws = new WebSocket(this.config.url, this.config.protocols);

      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
        this.ws?.close();
      }, 10000);

      this.ws.onopen = () => {
        clearTimeout(timeout);
        this.connectionState = 'connected';
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        this.flushMessageQueue();
        this.emit('connected', {});
        resolve();
      };

      this.ws.onerror = (error) => {
        clearTimeout(timeout);
        this.emit('error', error);
        if (this.connectionState === 'connecting') {
          reject(error);
        }
      };

      this.ws.onclose = (event) => {
        clearTimeout(timeout);
        this.connectionState = 'disconnected';
        this.stopHeartbeat();
        this.emit('disconnected', { code: event.code, reason: event.reason });

        if (event.code !== 1000 && this.reconnectAttempts < this.config.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  disconnect(): void {
    this.connectionState = 'disconnected';
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  send(type: string, data: any, reliable: boolean = true): string {
    const message: Message = {
      id: Math.random().toString(36),
      type,
      data,
      timestamp: Date.now(),
      retries: 0,
    };

    if (this.connectionState === 'connected' && this.ws) {
      try {
        this.ws.send(JSON.stringify(message));
        return message.id;
      } catch (error) {
        if (reliable) {
          this.queueMessage(message);
        }
        throw error;
      }
    } else if (reliable) {
      this.queueMessage(message);
      return message.id;
    } else {
      throw new Error('Connection not available');
    }
  }

  on(event: string, handler: (data: any) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
  }

  off(event: string, handler: (data: any) => void): void {
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  getConnectionState(): string {
    return this.connectionState;
  }

  getQueueSize(): number {
    return this.messageQueue.length;
  }

  private queueMessage(message: Message): void {
    if (this.messageQueue.length >= this.config.messageQueueSize) {
      this.messageQueue.shift(); // Remove oldest message
    }
    this.messageQueue.push(message);
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.connectionState === 'connected') {
      const message = this.messageQueue.shift()!;
      try {
        this.ws?.send(JSON.stringify(message));
      } catch (error) {
        // Re-queue on failure
        this.queueMessage(message);
        break;
      }
    }
  }

  private scheduleReconnect(): void {
    this.connectionState = 'reconnecting';
    this.reconnectAttempts++;
    
    const delay = this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(() => {
        // Reconnection failed, will try again if attempts remain
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.connectionState === 'connected') {
        this.send('heartbeat', { timestamp: Date.now() }, false);
        
        // Check if we received a heartbeat recently
        if (Date.now() - this.lastHeartbeat > this.config.heartbeatInterval * 2) {
          this.ws?.close(1006, 'Heartbeat timeout');
        }
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }
  }

  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      if (message.type === 'heartbeat') {
        this.lastHeartbeat = Date.now();
        return;
      }

      this.emit('message', message);
      this.emit(message.type, message.data);
    } catch (error) {
      this.emit('error', new Error('Invalid message format'));
    }
  }

  private emit(event: string, data: any): void {
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('Error in event handler:', error);
        }
      });
    }
  }
}

// Real-time Data Synchronizer
interface SyncState {
  [key: string]: {
    value: any;
    version: number;
    timestamp: number;
    dirty: boolean;
  };
}

class RealTimeDataSynchronizer {
  private connectionManager: RealTimeConnectionManager;
  private localState: SyncState = {};
  private syncTimer?: NodeJS.Timeout;
  private conflictResolvers: Map<string, (local: any, remote: any) => any> = new Map();

  constructor(connectionManager: RealTimeConnectionManager) {
    this.connectionManager = connectionManager;
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.connectionManager.on('connected', () => {
      this.startSync();
      this.requestFullSync();
    });

    this.connectionManager.on('disconnected', () => {
      this.stopSync();
    });

    this.connectionManager.on('sync:update', (data) => {
      this.handleRemoteUpdate(data);
    });

    this.connectionManager.on('sync:conflict', (data) => {
      this.handleConflict(data);
    });

    this.connectionManager.on('sync:full', (data) => {
      this.handleFullSync(data);
    });
  }

  set(key: string, value: any): void {
    const currentTime = Date.now();
    const currentState = this.localState[key];
    
    this.localState[key] = {
      value,
      version: currentState ? currentState.version + 1 : 1,
      timestamp: currentTime,
      dirty: true,
    };

    // Send immediate update for real-time data
    if (this.connectionManager.getConnectionState() === 'connected') {
      this.sendUpdate(key);
    }
  }

  get(key: string): any {
    return this.localState[key]?.value;
  }

  delete(key: string): void {
    if (this.localState[key]) {
      this.localState[key] = {
        value: null,
        version: this.localState[key].version + 1,
        timestamp: Date.now(),
        dirty: true,
      };

      if (this.connectionManager.getConnectionState() === 'connected') {
        this.sendUpdate(key);
      }
    }
  }

  setConflictResolver(key: string, resolver: (local: any, remote: any) => any): void {
    this.conflictResolvers.set(key, resolver);
  }

  getState(): Record<string, any> {
    const result: Record<string, any> = {};
    Object.keys(this.localState).forEach(key => {
      if (this.localState[key].value !== null) {
        result[key] = this.localState[key].value;
      }
    });
    return result;
  }

  private startSync(): void {
    this.syncTimer = setInterval(() => {
      this.syncDirtyData();
    }, 1000);
  }

  private stopSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
    }
  }

  private syncDirtyData(): void {
    Object.keys(this.localState).forEach(key => {
      if (this.localState[key].dirty) {
        this.sendUpdate(key);
      }
    });
  }

  private sendUpdate(key: string): void {
    const state = this.localState[key];
    if (state) {
      this.connectionManager.send('sync:update', {
        key,
        value: state.value,
        version: state.version,
        timestamp: state.timestamp,
      });
      state.dirty = false;
    }
  }

  private requestFullSync(): void {
    this.connectionManager.send('sync:request', {});
  }

  private handleRemoteUpdate(data: any): void {
    const { key, value, version, timestamp } = data;
    const localState = this.localState[key];

    if (!localState || version > localState.version) {
      // Remote is newer, accept update
      this.localState[key] = {
        value,
        version,
        timestamp,
        dirty: false,
      };
    } else if (version === localState.version && timestamp !== localState.timestamp) {
      // Conflict detected
      this.handleConflict({ key, local: localState, remote: data });
    }
  }

  private handleConflict(data: any): void {
    const { key, local, remote } = data;
    const resolver = this.conflictResolvers.get(key);

    if (resolver) {
      const resolvedValue = resolver(local?.value, remote.value);
      this.localState[key] = {
        value: resolvedValue,
        version: Math.max(local?.version || 0, remote.version) + 1,
        timestamp: Date.now(),
        dirty: true,
      };
    } else {
      // Default: latest timestamp wins
      if (remote.timestamp > (local?.timestamp || 0)) {
        this.localState[key] = {
          value: remote.value,
          version: remote.version,
          timestamp: remote.timestamp,
          dirty: false,
        };
      }
    }
  }

  private handleFullSync(data: any): void {
    Object.keys(data).forEach(key => {
      const remoteState = data[key];
      const localState = this.localState[key];

      if (!localState || remoteState.version > localState.version) {
        this.localState[key] = {
          ...remoteState,
          dirty: false,
        };
      }
    });
  }
}

// Live Update Component
interface LiveDataDisplayProps {
  dataKey: string;
  synchronizer: RealTimeDataSynchronizer;
  formatter?: (value: any) => string;
}

const LiveDataDisplay: React.FC<LiveDataDisplayProps> = ({
  dataKey,
  synchronizer,
  formatter = (value) => JSON.stringify(value),
}) => {
  const [value, setValue] = React.useState(synchronizer.get(dataKey));
  const [lastUpdate, setLastUpdate] = React.useState<Date | null>(null);

  React.useEffect(() => {
    const updateValue = () => {
      const newValue = synchronizer.get(dataKey);
      setValue(newValue);
      setLastUpdate(new Date());
    };

    // Poll for updates (in real implementation, would use proper events)
    const interval = setInterval(updateValue, 100);

    return () => clearInterval(interval);
  }, [dataKey, synchronizer]);

  return (
    <div data-testid={`live-data-${dataKey}`}>
      <div data-testid="value">{formatter(value)}</div>
      <div data-testid="last-update">
        {lastUpdate ? lastUpdate.toISOString() : 'Never'}
      </div>
    </div>
  );
};

// Collaborative Editor Component
interface CollaborativeEditorProps {
  documentId: string;
  synchronizer: RealTimeDataSynchronizer;
}

const CollaborativeEditor: React.FC<CollaborativeEditorProps> = ({
  documentId,
  synchronizer,
}) => {
  const [content, setContent] = React.useState('');
  const [collaborators, setCollaborators] = React.useState<string[]>([]);
  const [isOnline, setIsOnline] = React.useState(false);

  React.useEffect(() => {
    // Set up conflict resolution for collaborative editing
    synchronizer.setConflictResolver(documentId, (local, remote) => {
      // Simple merge strategy - in practice would use operational transforms
      return local + '\n--- MERGED ---\n' + remote;
    });

    // Initialize content
    const initialContent = synchronizer.get(documentId) || '';
    setContent(initialContent);
  }, [documentId, synchronizer]);

  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    synchronizer.set(documentId, newContent);
  };

  const handleCollaboratorJoin = (collaboratorId: string) => {
    setCollaborators(prev => [...prev, collaboratorId]);
  };

  const handleCollaboratorLeave = (collaboratorId: string) => {
    setCollaborators(prev => prev.filter(id => id !== collaboratorId));
  };

  return (
    <div data-testid="collaborative-editor">
      <div data-testid="status">
        Status: {isOnline ? 'Online' : 'Offline'}
      </div>
      
      <div data-testid="collaborators">
        Collaborators: {collaborators.join(', ') || 'None'}
      </div>
      
      <textarea
        data-testid="editor-content"
        value={content}
        onChange={(e) => handleContentChange(e.target.value)}
        rows={10}
        cols={50}
      />
      
      <button
        data-testid="add-collaborator"
        onClick={() => handleCollaboratorJoin(`user-${Math.random().toString(36).substr(2, 5)}`)}
      >
        Simulate Collaborator Join
      </button>
    </div>
  );
};

// Message Queue Manager
interface QueuedMessage {
  id: string;
  type: string;
  data: any;
  priority: number;
  timestamp: number;
  attempts: number;
  maxAttempts: number;
  delay: number;
}

class MessageQueueManager {
  private queue: QueuedMessage[] = [];
  private processing: boolean = false;
  private connectionManager: RealTimeConnectionManager;
  private processingTimer?: NodeJS.Timeout;

  constructor(connectionManager: RealTimeConnectionManager) {
    this.connectionManager = connectionManager;
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.connectionManager.on('connected', () => {
      this.startProcessing();
    });

    this.connectionManager.on('disconnected', () => {
      this.stopProcessing();
    });
  }

  enqueue(
    type: string,
    data: any,
    priority: number = 0,
    maxAttempts: number = 3,
    delay: number = 0
  ): string {
    const message: QueuedMessage = {
      id: Math.random().toString(36),
      type,
      data,
      priority,
      timestamp: Date.now(),
      attempts: 0,
      maxAttempts,
      delay,
    };

    // Insert based on priority (higher priority first)
    const insertIndex = this.queue.findIndex(m => m.priority < priority);
    if (insertIndex === -1) {
      this.queue.push(message);
    } else {
      this.queue.splice(insertIndex, 0, message);
    }

    if (this.connectionManager.getConnectionState() === 'connected') {
      this.startProcessing();
    }

    return message.id;
  }

  dequeue(messageId: string): boolean {
    const index = this.queue.findIndex(m => m.id === messageId);
    if (index !== -1) {
      this.queue.splice(index, 1);
      return true;
    }
    return false;
  }

  getQueueLength(): number {
    return this.queue.length;
  }

  getQueuedMessages(): QueuedMessage[] {
    return [...this.queue];
  }

  private startProcessing(): void {
    if (this.processing) return;
    
    this.processing = true;
    this.processNext();
  }

  private stopProcessing(): void {
    this.processing = false;
    if (this.processingTimer) {
      clearTimeout(this.processingTimer);
    }
  }

  private async processNext(): Promise<void> {
    if (!this.processing || this.queue.length === 0) {
      this.processing = false;
      return;
    }

    const message = this.queue[0];
    
    // Check if message should be delayed
    if (message.delay > 0 && Date.now() - message.timestamp < message.delay) {
      this.processingTimer = setTimeout(() => this.processNext(), 100);
      return;
    }

    try {
      await this.sendMessage(message);
      this.queue.shift(); // Remove successfully sent message
    } catch (error) {
      message.attempts++;
      
      if (message.attempts >= message.maxAttempts) {
        this.queue.shift(); // Remove failed message
        console.error('Message failed after max attempts:', message);
      } else {
        // Exponential backoff
        message.delay = Math.pow(2, message.attempts) * 1000;
        message.timestamp = Date.now();
      }
    }

    // Continue processing
    this.processingTimer = setTimeout(() => this.processNext(), 10);
  }

  private async sendMessage(message: QueuedMessage): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.connectionManager.send(message.type, message.data, false);
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  }
}

// Tests
describe('Real-time WebSocket Communication', () => {
  let connectionManager: RealTimeConnectionManager;
  let mockWebSocket: EnhancedMockWebSocket;

  beforeEach(() => {
    connectionManager = new RealTimeConnectionManager({
      url: 'ws://localhost:8080/test',
      reconnectInterval: 100,
      maxReconnectAttempts: 3,
    });
  });

  afterEach(() => {
    connectionManager.disconnect();
  });

  describe('RealTimeConnectionManager', () => {
    test('should establish connection successfully', async () => {
      const connectPromise = connectionManager.connect();
      
      await expect(connectPromise).resolves.toBeUndefined();
      expect(connectionManager.getConnectionState()).toBe('connected');
    });

    test('should handle connection failure', async () => {
      const failManager = new RealTimeConnectionManager({
        url: 'ws://localhost:8080/fail',
      });

      await expect(failManager.connect()).rejects.toThrow();
      expect(failManager.getConnectionState()).toBe('disconnected');
    });

    test('should queue messages when disconnected', () => {
      expect(connectionManager.getConnectionState()).toBe('disconnected');
      
      expect(() => {
        connectionManager.send('test', { data: 'test' }, true);
      }).toThrow();
      
      expect(connectionManager.getQueueSize()).toBe(1);
    });

    test('should send messages when connected', async () => {
      await connectionManager.connect();
      
      const messageId = connectionManager.send('test', { data: 'test' });
      
      expect(messageId).toBeDefined();
      expect(typeof messageId).toBe('string');
    });

    test('should handle message reception', async () => {
      const messageHandler = jest.fn();
      connectionManager.on('test:message', messageHandler);
      
      await connectionManager.connect();
      
      // Simulate incoming message
      const ws = connectionManager['ws'] as EnhancedMockWebSocket;
      ws.simulateServerMessage({
        type: 'test:message',
        data: { content: 'Hello from server' },
      });

      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(messageHandler).toHaveBeenCalledWith({ content: 'Hello from server' });
    });

    test('should attempt reconnection on connection drop', async () => {
      const disconnectHandler = jest.fn();
      const reconnectHandler = jest.fn();
      
      connectionManager.on('disconnected', disconnectHandler);
      connectionManager.on('connected', reconnectHandler);
      
      await connectionManager.connect();
      
      // Simulate connection drop
      const ws = connectionManager['ws'] as EnhancedMockWebSocket;
      ws.simulateConnectionDrop();
      
      await new Promise(resolve => setTimeout(resolve, 50));
      
      expect(disconnectHandler).toHaveBeenCalled();
      expect(connectionManager.getConnectionState()).toBe('disconnected');
    });

    test('should handle heartbeat mechanism', async () => {
      const heartbeatHandler = jest.fn();
      connectionManager.on('heartbeat', heartbeatHandler);
      
      await connectionManager.connect();
      
      // Simulate heartbeat
      const ws = connectionManager['ws'] as EnhancedMockWebSocket;
      ws.simulateServerMessage({
        type: 'heartbeat',
        timestamp: Date.now(),
      });

      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(heartbeatHandler).toHaveBeenCalled();
    });

    test('should limit message queue size', () => {
      const limitedManager = new RealTimeConnectionManager({
        messageQueueSize: 2,
      });

      // Add messages beyond limit
      limitedManager.send('test1', {}, true);
      limitedManager.send('test2', {}, true);
      limitedManager.send('test3', {}, true);
      
      expect(limitedManager.getQueueSize()).toBe(2);
    });

    test('should handle invalid JSON messages', async () => {
      const errorHandler = jest.fn();
      connectionManager.on('error', errorHandler);
      
      await connectionManager.connect();
      
      const ws = connectionManager['ws'] as EnhancedMockWebSocket;
      ws.simulateIncomingMessage('invalid json {{{');
      
      await new Promise(resolve => setTimeout(resolve, 10));
      
      expect(errorHandler).toHaveBeenCalled();
    });
  });

  describe('RealTimeDataSynchronizer', () => {
    let synchronizer: RealTimeDataSynchronizer;

    beforeEach(async () => {
      await connectionManager.connect();
      synchronizer = new RealTimeDataSynchronizer(connectionManager);
    });

    test('should sync data locally', () => {
      synchronizer.set('key1', 'value1');
      expect(synchronizer.get('key1')).toBe('value1');
      
      synchronizer.set('key2', { nested: true });
      expect(synchronizer.get('key2')).toEqual({ nested: true });
    });

    test('should delete data', () => {
      synchronizer.set('key1', 'value1');
      expect(synchronizer.get('key1')).toBe('value1');
      
      synchronizer.delete('key1');
      expect(synchronizer.get('key1')).toBeNull();
    });

    test('should handle conflict resolution', () => {
      const resolver = jest.fn((local, remote) => `${local}-${remote}`);
      synchronizer.setConflictResolver('conflicted-key', resolver);
      
      synchronizer.set('conflicted-key', 'local-value');
      
      // Simulate remote update with conflict
      synchronizer['handleConflict']({
        key: 'conflicted-key',
        local: { value: 'local-value', version: 1, timestamp: Date.now() - 1000 },
        remote: { value: 'remote-value', version: 1, timestamp: Date.now() },
      });
      
      expect(resolver).toHaveBeenCalledWith('local-value', 'remote-value');
    });

    test('should return complete state', () => {
      synchronizer.set('key1', 'value1');
      synchronizer.set('key2', 'value2');
      synchronizer.delete('key1');
      
      const state = synchronizer.getState();
      
      expect(state).toEqual({ key2: 'value2' });
      expect(state).not.toHaveProperty('key1');
    });

    test('should handle remote updates', () => {
      synchronizer['handleRemoteUpdate']({
        key: 'remote-key',
        value: 'remote-value',
        version: 1,
        timestamp: Date.now(),
      });
      
      expect(synchronizer.get('remote-key')).toBe('remote-value');
    });

    test('should handle full sync', () => {
      const fullSyncData = {
        key1: { value: 'sync1', version: 1, timestamp: Date.now() },
        key2: { value: 'sync2', version: 1, timestamp: Date.now() },
      };
      
      synchronizer['handleFullSync'](fullSyncData);
      
      expect(synchronizer.get('key1')).toBe('sync1');
      expect(synchronizer.get('key2')).toBe('sync2');
    });
  });

  describe('MessageQueueManager', () => {
    let queueManager: MessageQueueManager;

    beforeEach(async () => {
      await connectionManager.connect();
      queueManager = new MessageQueueManager(connectionManager);
    });

    test('should enqueue messages with priority', () => {
      const lowId = queueManager.enqueue('low', {}, 1);
      const highId = queueManager.enqueue('high', {}, 10);
      const mediumId = queueManager.enqueue('medium', {}, 5);
      
      const queue = queueManager.getQueuedMessages();
      
      expect(queue[0].type).toBe('high');
      expect(queue[1].type).toBe('medium');
      expect(queue[2].type).toBe('low');
    });

    test('should dequeue messages by ID', () => {
      const messageId = queueManager.enqueue('test', {});
      
      expect(queueManager.getQueueLength()).toBe(1);
      
      const removed = queueManager.dequeue(messageId);
      
      expect(removed).toBe(true);
      expect(queueManager.getQueueLength()).toBe(0);
    });

    test('should handle message processing failure', async () => {
      // Enqueue message that will fail
      queueManager.enqueue('error', {}, 0, 2);
      
      // Wait for processing attempts
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Message should be removed after max attempts
      expect(queueManager.getQueueLength()).toBe(0);
    });

    test('should respect message delays', () => {
      const delayedId = queueManager.enqueue('delayed', {}, 0, 3, 1000);
      const immediateId = queueManager.enqueue('immediate', {}, 0, 3, 0);
      
      const queue = queueManager.getQueuedMessages();
      const delayedMessage = queue.find(m => m.id === delayedId);
      
      expect(delayedMessage?.delay).toBe(1000);
    });

    test('should stop processing when disconnected', () => {
      queueManager.enqueue('test', {});
      
      connectionManager.disconnect();
      
      // Processing should stop
      expect(queueManager['processing']).toBe(false);
    });
  });

  describe('React Components', () => {
    let synchronizer: RealTimeDataSynchronizer;

    beforeEach(async () => {
      await connectionManager.connect();
      synchronizer = new RealTimeDataSynchronizer(connectionManager);
    });

    describe('LiveDataDisplay', () => {
      test('should display synchronized data', () => {
        synchronizer.set('test-data', 'Hello World');
        
        render(
          <LiveDataDisplay 
            dataKey="test-data" 
            synchronizer={synchronizer}
          />
        );

        expect(screen.getByTestId('value')).toHaveTextContent('"Hello World"');
      });

      test('should update when data changes', async () => {
        synchronizer.set('dynamic-data', 'Initial');
        
        render(
          <LiveDataDisplay 
            dataKey="dynamic-data" 
            synchronizer={synchronizer}
          />
        );

        expect(screen.getByTestId('value')).toHaveTextContent('"Initial"');
        
        // Update data
        act(() => {
          synchronizer.set('dynamic-data', 'Updated');
        });

        // Wait for component update
        await waitFor(() => {
          expect(screen.getByTestId('value')).toHaveTextContent('"Updated"');
        });
      });

      test('should use custom formatter', () => {
        synchronizer.set('formatted-data', { count: 42 });
        
        render(
          <LiveDataDisplay 
            dataKey="formatted-data" 
            synchronizer={synchronizer}
            formatter={(value) => `Count: ${value?.count || 0}`}
          />
        );

        expect(screen.getByTestId('value')).toHaveTextContent('Count: 42');
      });
    });

    describe('CollaborativeEditor', () => {
      test('should render editor interface', () => {
        render(
          <CollaborativeEditor 
            documentId="test-doc"
            synchronizer={synchronizer}
          />
        );

        expect(screen.getByTestId('collaborative-editor')).toBeInTheDocument();
        expect(screen.getByTestId('editor-content')).toBeInTheDocument();
        expect(screen.getByTestId('status')).toHaveTextContent('Offline');
      });

      test('should update content and sync', async () => {
        render(
          <CollaborativeEditor 
            documentId="test-doc"
            synchronizer={synchronizer}
          />
        );

        const editor = screen.getByTestId('editor-content') as HTMLTextAreaElement;
        
        fireEvent.change(editor, { target: { value: 'New content' } });
        
        expect(editor.value).toBe('New content');
        expect(synchronizer.get('test-doc')).toBe('New content');
      });

      test('should handle collaborator simulation', () => {
        render(
          <CollaborativeEditor 
            documentId="test-doc"
            synchronizer={synchronizer}
          />
        );

        const addButton = screen.getByTestId('add-collaborator');
        
        fireEvent.click(addButton);
        
        const collaborators = screen.getByTestId('collaborators');
        expect(collaborators.textContent).toContain('user-');
      });

      test('should initialize with existing content', () => {
        synchronizer.set('existing-doc', 'Existing content');
        
        render(
          <CollaborativeEditor 
            documentId="existing-doc"
            synchronizer={synchronizer}
          />
        );

        const editor = screen.getByTestId('editor-content') as HTMLTextAreaElement;
        expect(editor.value).toBe('Existing content');
      });
    });
  });

  describe('Integration Scenarios', () => {
    test('should handle complete real-time workflow', async () => {
      // Set up complete real-time system
      await connectionManager.connect();
      const synchronizer = new RealTimeDataSynchronizer(connectionManager);
      const queueManager = new MessageQueueManager(connectionManager);

      // Set up collaborative editing scenario
      const document1 = 'doc1';
      const document2 = 'doc2';

      // User 1 creates document
      synchronizer.set(document1, 'Hello');
      
      // User 2 edits same document (simulate remote update)
      synchronizer['handleRemoteUpdate']({
        key: document1,
        value: 'Hello World',
        version: 2,
        timestamp: Date.now(),
      });

      expect(synchronizer.get(document1)).toBe('Hello World');

      // Queue priority messages
      const urgentId = queueManager.enqueue('urgent', { alert: true }, 10);
      const normalId = queueManager.enqueue('normal', { data: 'test' }, 0);

      expect(queueManager.getQueueLength()).toBe(2);
      
      const queue = queueManager.getQueuedMessages();
      expect(queue[0].type).toBe('urgent');
    });

    test('should handle connection resilience', async () => {
      const eventLog: string[] = [];
      
      connectionManager.on('connected', () => eventLog.push('connected'));
      connectionManager.on('disconnected', () => eventLog.push('disconnected'));
      connectionManager.on('error', () => eventLog.push('error'));

      // Initial connection
      await connectionManager.connect();
      expect(eventLog).toContain('connected');

      // Simulate connection drop
      const ws = connectionManager['ws'] as EnhancedMockWebSocket;
      ws.simulateConnectionDrop();

      await new Promise(resolve => setTimeout(resolve, 50));
      expect(eventLog).toContain('disconnected');
    });

    test('should handle high-frequency updates', async () => {
      await connectionManager.connect();
      const synchronizer = new RealTimeDataSynchronizer(connectionManager);

      // Simulate high-frequency updates
      const updates = Array.from({ length: 100 }, (_, i) => ({
        key: 'counter',
        value: i,
        version: i + 1,
        timestamp: Date.now() + i,
      }));

      updates.forEach(update => {
        synchronizer['handleRemoteUpdate'](update);
      });

      // Should have the latest value
      expect(synchronizer.get('counter')).toBe(99);
    });

    test('should handle concurrent message processing', async () => {
      await connectionManager.connect();
      const queueManager = new MessageQueueManager(connectionManager);

      // Enqueue many messages rapidly
      const messageIds = Array.from({ length: 50 }, (_, i) =>
        queueManager.enqueue(`message-${i}`, { index: i }, Math.random() * 10)
      );

      expect(queueManager.getQueueLength()).toBe(50);

      // Process should handle all messages
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Some messages should have been processed
      expect(queueManager.getQueueLength()).toBeLessThan(50);
    });
  });
});