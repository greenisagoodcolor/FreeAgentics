import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface ConnectionStatus {
  websocket: 'connecting' | 'connected' | 'disconnected' | 'error';
  api: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastPing: number | null;
  latency: number | null;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
}

export interface ConnectionError {
  type: 'websocket' | 'api';
  message: string;
  timestamp: number;
  code?: string;
}

interface ConnectionState {
  status: ConnectionStatus;
  errors: ConnectionError[];
  isReconnecting: boolean;
  autoReconnect: boolean;
  reconnectDelay: number; // milliseconds
  socketUrl: string | null;
  apiUrl: string | null;
  connectionId: string | null;
  heartbeatInterval: number; // milliseconds
}

const initialState: ConnectionState = {
  status: {
    websocket: 'disconnected',
    api: 'disconnected',
    lastPing: null,
    latency: null,
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
  },
  errors: [],
  isReconnecting: false,
  autoReconnect: true,
  reconnectDelay: 1000,
  socketUrl: null,
  apiUrl: null,
  connectionId: null,
  heartbeatInterval: 30000, // 30 seconds
};

const connectionSlice = createSlice({
  name: 'connection',
  initialState,
  reducers: {
    // WebSocket connection management
    setWebSocketStatus: (state, action: PayloadAction<ConnectionStatus['websocket']>) => {
      state.status.websocket = action.payload;
      
      if (action.payload === 'connected') {
        state.status.reconnectAttempts = 0;
        state.isReconnecting = false;
      } else if (action.payload === 'disconnected' && state.autoReconnect) {
        state.isReconnecting = true;
      }
    },

    // API connection management
    setApiStatus: (state, action: PayloadAction<ConnectionStatus['api']>) => {
      state.status.api = action.payload;
    },

    // Connection established
    connectionEstablished: (state, action: PayloadAction<{
      connectionId: string;
      socketUrl: string;
      apiUrl: string;
    }>) => {
      const { connectionId, socketUrl, apiUrl } = action.payload;
      state.connectionId = connectionId;
      state.socketUrl = socketUrl;
      state.apiUrl = apiUrl;
      state.status.websocket = 'connected';
      state.status.api = 'connected';
      state.isReconnecting = false;
      state.status.reconnectAttempts = 0;
    },

    // Connection lost
    connectionLost: (state, action: PayloadAction<{
      type: 'websocket' | 'api';
      error?: string;
    }>) => {
      const { type, error } = action.payload;
      
      if (type === 'websocket') {
        state.status.websocket = 'disconnected';
      } else {
        state.status.api = 'disconnected';
      }

      if (error) {
        state.errors.push({
          type,
          message: error,
          timestamp: Date.now(),
        });
      }

      if (state.autoReconnect && state.status.reconnectAttempts < state.status.maxReconnectAttempts) {
        state.isReconnecting = true;
      }
    },

    // Update latency
    updateLatency: (state, action: PayloadAction<number>) => {
      state.status.latency = action.payload;
      state.status.lastPing = Date.now();
    },

    // Reconnection attempt
    incrementReconnectAttempt: (state) => {
      state.status.reconnectAttempts += 1;
      
      if (state.status.reconnectAttempts >= state.status.maxReconnectAttempts) {
        state.isReconnecting = false;
        state.status.websocket = 'error';
      } else {
        // Exponential backoff
        state.reconnectDelay = Math.min(
          state.reconnectDelay * 2,
          30000 // Max 30 seconds
        );
      }
    },

    // Reset reconnection
    resetReconnection: (state) => {
      state.status.reconnectAttempts = 0;
      state.reconnectDelay = 1000;
      state.isReconnecting = false;
    },

    // Toggle auto-reconnect
    toggleAutoReconnect: (state) => {
      state.autoReconnect = !state.autoReconnect;
    },

    // Add error
    addConnectionError: (state, action: PayloadAction<Omit<ConnectionError, 'timestamp'>>) => {
      state.errors.push({
        ...action.payload,
        timestamp: Date.now(),
      });

      // Keep only last 50 errors
      if (state.errors.length > 50) {
        state.errors = state.errors.slice(-50);
      }
    },

    // Clear errors
    clearConnectionErrors: (state) => {
      state.errors = [];
    },

    // Update connection URLs
    updateConnectionUrls: (state, action: PayloadAction<{
      socketUrl?: string;
      apiUrl?: string;
    }>) => {
      if (action.payload.socketUrl) {
        state.socketUrl = action.payload.socketUrl;
      }
      if (action.payload.apiUrl) {
        state.apiUrl = action.payload.apiUrl;
      }
    },

    // Set heartbeat interval
    setHeartbeatInterval: (state, action: PayloadAction<number>) => {
      state.heartbeatInterval = action.payload;
    },

    // Force reconnect
    forceReconnect: (state) => {
      state.status.websocket = 'disconnected';
      state.status.api = 'disconnected';
      state.isReconnecting = true;
      state.status.reconnectAttempts = 0;
      state.reconnectDelay = 1000;
    },

    // Complete disconnect
    disconnect: (state) => {
      state.status.websocket = 'disconnected';
      state.status.api = 'disconnected';
      state.connectionId = null;
      state.isReconnecting = false;
      state.autoReconnect = false;
    },
  },
});

export const {
  setWebSocketStatus,
  setApiStatus,
  connectionEstablished,
  connectionLost,
  updateLatency,
  incrementReconnectAttempt,
  resetReconnection,
  toggleAutoReconnect,
  addConnectionError,
  clearConnectionErrors,
  updateConnectionUrls,
  setHeartbeatInterval,
  forceReconnect,
  disconnect,
} = connectionSlice.actions;

export default connectionSlice.reducer; 