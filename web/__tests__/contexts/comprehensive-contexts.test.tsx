/**
 * Comprehensive Context Tests
 * 
 * Tests for React contexts including LLM context, theme provider,
 * and state management contexts following ADR-007 requirements.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { renderHook, act } from '@testing-library/react';
import { jest } from '@jest/globals';

// Mock comprehensive context implementations
interface LLMConfig {
  provider: string;
  model: string;
  apiKey: string;
  temperature: number;
  maxTokens: number;
}

interface LLMContextType {
  config: LLMConfig;
  setConfig: (config: Partial<LLMConfig>) => void;
  isConnected: boolean;
  sendMessage: (message: string) => Promise<string>;
  resetConnection: () => void;
  usage: {
    tokensUsed: number;
    requestCount: number;
    errorCount: number;
  };
}

const defaultLLMConfig: LLMConfig = {
  provider: 'openai',
  model: 'gpt-3.5-turbo',
  apiKey: '',
  temperature: 0.7,
  maxTokens: 1000,
};

const LLMContext = React.createContext<LLMContextType | undefined>(undefined);

const LLMProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfigState] = React.useState<LLMConfig>(defaultLLMConfig);
  const [isConnected, setIsConnected] = React.useState(false);
  const [usage, setUsage] = React.useState({
    tokensUsed: 0,
    requestCount: 0,
    errorCount: 0,
  });

  const setConfig = React.useCallback((newConfig: Partial<LLMConfig>) => {
    setConfigState(prev => ({ ...prev, ...newConfig }));
  }, []);

  const sendMessage = React.useCallback(async (message: string): Promise<string> => {
    if (!config.apiKey) {
      setUsage(prev => ({ ...prev, errorCount: prev.errorCount + 1 }));
      throw new Error('No API key configured');
    }

    setUsage(prev => ({ ...prev, requestCount: prev.requestCount + 1 }));
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Simulate token usage
    const estimatedTokens = message.length / 4 + 50; // Rough estimate
    setUsage(prev => ({ ...prev, tokensUsed: prev.tokensUsed + estimatedTokens }));
    
    return `Response to: ${message}`;
  }, [config.apiKey]);

  const resetConnection = React.useCallback(() => {
    setIsConnected(false);
    setUsage({ tokensUsed: 0, requestCount: 0, errorCount: 0 });
    setTimeout(() => setIsConnected(true), 100);
  }, []);

  React.useEffect(() => {
    if (config.apiKey) {
      setIsConnected(true);
    } else {
      setIsConnected(false);
    }
  }, [config.apiKey]);

  const value: LLMContextType = {
    config,
    setConfig,
    isConnected,
    sendMessage,
    resetConnection,
    usage,
  };

  return <LLMContext.Provider value={value}>{children}</LLMContext.Provider>;
};

const useLLM = () => {
  const context = React.useContext(LLMContext);
  if (!context) {
    throw new Error('useLLM must be used within LLMProvider');
  }
  return context;
};

// Theme Context
type Theme = 'light' | 'dark' | 'auto';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  systemTheme: 'light' | 'dark';
  effectiveTheme: 'light' | 'dark';
  colors: {
    primary: string;
    secondary: string;
    background: string;
    text: string;
  };
}

const ThemeContext = React.createContext<ThemeContextType | undefined>(undefined);

const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setTheme] = React.useState<Theme>('auto');
  const [systemTheme, setSystemTheme] = React.useState<'light' | 'dark'>('light');

  // Mock system theme detection
  React.useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemTheme(mediaQuery.matches ? 'dark' : 'light');
    
    const handler = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };
    
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const effectiveTheme = theme === 'auto' ? systemTheme : theme;

  const colors = React.useMemo(() => {
    if (effectiveTheme === 'dark') {
      return {
        primary: '#3b82f6',
        secondary: '#6b7280',
        background: '#111827',
        text: '#f9fafb',
      };
    }
    return {
      primary: '#2563eb',
      secondary: '#4b5563',
      background: '#ffffff',
      text: '#111827',
    };
  }, [effectiveTheme]);

  const value: ThemeContextType = {
    theme,
    setTheme,
    systemTheme,
    effectiveTheme,
    colors,
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

const useTheme = () => {
  const context = React.useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

// Application State Context
interface AppState {
  user: {
    id: string;
    name: string;
    preferences: Record<string, any>;
  } | null;
  notifications: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: number;
  }>;
  isLoading: boolean;
  error: string | null;
}

interface AppContextType {
  state: AppState;
  setUser: (user: AppState['user']) => void;
  addNotification: (notification: Omit<AppState['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateUserPreferences: (preferences: Record<string, any>) => void;
}

const AppContext = React.createContext<AppContextType | undefined>(undefined);

const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = React.useState<AppState>({
    user: null,
    notifications: [],
    isLoading: false,
    error: null,
  });

  const setUser = React.useCallback((user: AppState['user']) => {
    setState(prev => ({ ...prev, user }));
  }, []);

  const addNotification = React.useCallback((notification: Omit<AppState['notifications'][0], 'id' | 'timestamp'>) => {
    const newNotification = {
      ...notification,
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
    };
    
    setState(prev => ({
      ...prev,
      notifications: [...prev.notifications, newNotification],
    }));
  }, []);

  const removeNotification = React.useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      notifications: prev.notifications.filter(n => n.id !== id),
    }));
  }, []);

  const clearNotifications = React.useCallback(() => {
    setState(prev => ({ ...prev, notifications: [] }));
  }, []);

  const setLoading = React.useCallback((loading: boolean) => {
    setState(prev => ({ ...prev, isLoading: loading }));
  }, []);

  const setError = React.useCallback((error: string | null) => {
    setState(prev => ({ ...prev, error }));
  }, []);

  const updateUserPreferences = React.useCallback((preferences: Record<string, any>) => {
    setState(prev => ({
      ...prev,
      user: prev.user ? {
        ...prev.user,
        preferences: { ...prev.user.preferences, ...preferences },
      } : null,
    }));
  }, []);

  const value: AppContextType = {
    state,
    setUser,
    addNotification,
    removeNotification,
    clearNotifications,
    setLoading,
    setError,
    updateUserPreferences,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

const useApp = () => {
  const context = React.useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};

// WebSocket Context
interface WebSocketContextType {
  socket: WebSocket | null;
  isConnected: boolean;
  lastMessage: any;
  sendMessage: (message: any) => void;
  connect: (url: string) => void;
  disconnect: () => void;
  reconnect: () => void;
  connectionAttempts: number;
  maxReconnectAttempts: number;
}

const WebSocketContext = React.createContext<WebSocketContextType | undefined>(undefined);

const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [socket, setSocket] = React.useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = React.useState(false);
  const [lastMessage, setLastMessage] = React.useState<any>(null);
  const [connectionAttempts, setConnectionAttempts] = React.useState(0);
  const [currentUrl, setCurrentUrl] = React.useState<string>('');
  const maxReconnectAttempts = 5;

  const connect = React.useCallback((url: string) => {
    if (socket) {
      socket.close();
    }
    
    setCurrentUrl(url);
    
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        setIsConnected(true);
        setConnectionAttempts(0);
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        setSocket(null);
        
        // Auto-reconnect logic
        if (connectionAttempts < maxReconnectAttempts) {
          setTimeout(() => {
            setConnectionAttempts(prev => prev + 1);
            connect(url);
          }, Math.pow(2, connectionAttempts) * 1000); // Exponential backoff
        }
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch {
          setLastMessage(event.data);
        }
      };
      
      ws.onerror = () => {
        setIsConnected(false);
      };
      
      setSocket(ws);
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  }, [socket, connectionAttempts, maxReconnectAttempts]);

  const disconnect = React.useCallback(() => {
    if (socket) {
      socket.close();
      setSocket(null);
      setIsConnected(false);
      setConnectionAttempts(0);
    }
  }, [socket]);

  const reconnect = React.useCallback(() => {
    if (currentUrl) {
      setConnectionAttempts(0);
      connect(currentUrl);
    }
  }, [currentUrl, connect]);

  const sendMessage = React.useCallback((message: any) => {
    if (socket && isConnected) {
      const messageStr = typeof message === 'string' ? message : JSON.stringify(message);
      socket.send(messageStr);
    }
  }, [socket, isConnected]);

  React.useEffect(() => {
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [socket]);

  const value: WebSocketContextType = {
    socket,
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnect,
    connectionAttempts,
    maxReconnectAttempts,
  };

  return <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>;
};

const useWebSocket = () => {
  const context = React.useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
};

// Test Components
const LLMTestComponent: React.FC = () => {
  const { config, setConfig, isConnected, sendMessage, usage } = useLLM();
  const [response, setResponse] = React.useState('');
  const [message, setMessage] = React.useState('');

  const handleSendMessage = async () => {
    try {
      const result = await sendMessage(message);
      setResponse(result);
    } catch (error) {
      setResponse(`Error: ${error}`);
    }
  };

  return (
    <div>
      <div data-testid="connection-status">
        {isConnected ? 'Connected' : 'Disconnected'}
      </div>
      <div data-testid="current-model">{config.model}</div>
      <div data-testid="token-usage">{usage.tokensUsed}</div>
      <input
        data-testid="api-key-input"
        value={config.apiKey}
        onChange={e => setConfig({ apiKey: e.target.value })}
        placeholder="API Key"
      />
      <input
        data-testid="message-input"
        value={message}
        onChange={e => setMessage(e.target.value)}
        placeholder="Message"
      />
      <button data-testid="send-button" onClick={handleSendMessage}>
        Send
      </button>
      <div data-testid="response">{response}</div>
    </div>
  );
};

const ThemeTestComponent: React.FC = () => {
  const { theme, setTheme, effectiveTheme, colors } = useTheme();

  return (
    <div style={{ backgroundColor: colors.background, color: colors.text }}>
      <div data-testid="current-theme">{theme}</div>
      <div data-testid="effective-theme">{effectiveTheme}</div>
      <div data-testid="background-color">{colors.background}</div>
      <button data-testid="light-theme" onClick={() => setTheme('light')}>
        Light
      </button>
      <button data-testid="dark-theme" onClick={() => setTheme('dark')}>
        Dark
      </button>
      <button data-testid="auto-theme" onClick={() => setTheme('auto')}>
        Auto
      </button>
    </div>
  );
};

const AppTestComponent: React.FC = () => {
  const {
    state,
    setUser,
    addNotification,
    removeNotification,
    clearNotifications,
    setLoading,
    setError,
  } = useApp();

  return (
    <div>
      <div data-testid="user-name">{state.user?.name || 'Not logged in'}</div>
      <div data-testid="notification-count">{state.notifications.length}</div>
      <div data-testid="loading-status">{state.isLoading ? 'Loading' : 'Idle'}</div>
      <div data-testid="error-message">{state.error || 'No error'}</div>
      
      <button
        data-testid="login-button"
        onClick={() => setUser({ id: '1', name: 'Test User', preferences: {} })}
      >
        Login
      </button>
      
      <button
        data-testid="logout-button"
        onClick={() => setUser(null)}
      >
        Logout
      </button>
      
      <button
        data-testid="add-notification"
        onClick={() => addNotification({ type: 'info', message: 'Test notification' })}
      >
        Add Notification
      </button>
      
      <button
        data-testid="clear-notifications"
        onClick={clearNotifications}
      >
        Clear Notifications
      </button>
      
      <button
        data-testid="set-loading"
        onClick={() => setLoading(true)}
      >
        Set Loading
      </button>
      
      <button
        data-testid="set-error"
        onClick={() => setError('Test error')}
      >
        Set Error
      </button>
      
      {state.notifications.map(notification => (
        <div key={notification.id} data-testid={`notification-${notification.id}`}>
          {notification.message}
          <button onClick={() => removeNotification(notification.id)}>
            Remove
          </button>
        </div>
      ))}
    </div>
  );
};

describe('Comprehensive Context Tests', () => {
  describe('LLMContext', () => {
    const renderWithProvider = (component: React.ReactElement) => {
      return render(<LLMProvider>{component}</LLMProvider>);
    };

    it('provides default configuration', () => {
      renderWithProvider(<LLMTestComponent />);
      
      expect(screen.getByTestId('current-model')).toHaveTextContent('gpt-3.5-turbo');
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');
      expect(screen.getByTestId('token-usage')).toHaveTextContent('0');
    });

    it('updates configuration', () => {
      renderWithProvider(<LLMTestComponent />);
      
      const apiKeyInput = screen.getByTestId('api-key-input');
      
      fireEvent.change(apiKeyInput, { target: { value: 'test-api-key' } });
      
      expect(apiKeyInput).toHaveValue('test-api-key');
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected');
    });

    it('sends messages and tracks usage', async () => {
      renderWithProvider(<LLMTestComponent />);
      
      // Set API key first
      fireEvent.change(screen.getByTestId('api-key-input'), {
        target: { value: 'test-api-key' },
      });
      
      // Send a message
      fireEvent.change(screen.getByTestId('message-input'), {
        target: { value: 'Hello, AI!' },
      });
      
      fireEvent.click(screen.getByTestId('send-button'));
      
      await waitFor(() => {
        expect(screen.getByTestId('response')).toHaveTextContent('Response to: Hello, AI!');
      });
      
      // Check token usage increased
      await waitFor(() => {
        expect(parseInt(screen.getByTestId('token-usage').textContent || '0')).toBeGreaterThan(0);
      });
    });

    it('handles errors when no API key is set', async () => {
      renderWithProvider(<LLMTestComponent />);
      
      fireEvent.change(screen.getByTestId('message-input'), {
        target: { value: 'Hello!' },
      });
      
      fireEvent.click(screen.getByTestId('send-button'));
      
      await waitFor(() => {
        expect(screen.getByTestId('response')).toHaveTextContent('Error: No API key configured');
      });
    });

    it('throws error when used outside provider', () => {
      const TestComponent = () => {
        useLLM();
        return <div>Test</div>;
      };
      
      expect(() => render(<TestComponent />)).toThrow(
        'useLLM must be used within LLMProvider'
      );
    });
  });

  describe('ThemeContext', () => {
    const renderWithProvider = (component: React.ReactElement) => {
      return render(<ThemeProvider>{component}</ThemeProvider>);
    };

    it('provides default theme', () => {
      renderWithProvider(<ThemeTestComponent />);
      
      expect(screen.getByTestId('current-theme')).toHaveTextContent('auto');
      expect(screen.getByTestId('effective-theme')).toHaveTextContent('light');
    });

    it('switches to light theme', () => {
      renderWithProvider(<ThemeTestComponent />);
      
      fireEvent.click(screen.getByTestId('light-theme'));
      
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
      expect(screen.getByTestId('effective-theme')).toHaveTextContent('light');
    });

    it('switches to dark theme', () => {
      renderWithProvider(<ThemeTestComponent />);
      
      fireEvent.click(screen.getByTestId('dark-theme'));
      
      expect(screen.getByTestId('current-theme')).toHaveTextContent('dark');
      expect(screen.getByTestId('effective-theme')).toHaveTextContent('dark');
    });

    it('provides correct colors for themes', () => {
      renderWithProvider(<ThemeTestComponent />);
      
      // Light theme colors
      fireEvent.click(screen.getByTestId('light-theme'));
      expect(screen.getByTestId('background-color')).toHaveTextContent('#ffffff');
      
      // Dark theme colors
      fireEvent.click(screen.getByTestId('dark-theme'));
      expect(screen.getByTestId('background-color')).toHaveTextContent('#111827');
    });

    it('respects system theme preference', () => {
      // Mock system prefers dark
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: jest.fn().mockImplementation(query => ({
          matches: query === '(prefers-color-scheme: dark)',
          media: query,
          onchange: null,
          addListener: jest.fn(),
          removeListener: jest.fn(),
          addEventListener: jest.fn(),
          removeEventListener: jest.fn(),
          dispatchEvent: jest.fn(),
        })),
      });
      
      renderWithProvider(<ThemeTestComponent />);
      
      fireEvent.click(screen.getByTestId('auto-theme'));
      expect(screen.getByTestId('effective-theme')).toHaveTextContent('dark');
    });

    it('throws error when used outside provider', () => {
      const TestComponent = () => {
        useTheme();
        return <div>Test</div>;
      };
      
      expect(() => render(<TestComponent />)).toThrow(
        'useTheme must be used within ThemeProvider'
      );
    });
  });

  describe('AppContext', () => {
    const renderWithProvider = (component: React.ReactElement) => {
      return render(<AppProvider>{component}</AppProvider>);
    };

    it('provides default state', () => {
      renderWithProvider(<AppTestComponent />);
      
      expect(screen.getByTestId('user-name')).toHaveTextContent('Not logged in');
      expect(screen.getByTestId('notification-count')).toHaveTextContent('0');
      expect(screen.getByTestId('loading-status')).toHaveTextContent('Idle');
      expect(screen.getByTestId('error-message')).toHaveTextContent('No error');
    });

    it('manages user state', () => {
      renderWithProvider(<AppTestComponent />);
      
      // Login
      fireEvent.click(screen.getByTestId('login-button'));
      expect(screen.getByTestId('user-name')).toHaveTextContent('Test User');
      
      // Logout
      fireEvent.click(screen.getByTestId('logout-button'));
      expect(screen.getByTestId('user-name')).toHaveTextContent('Not logged in');
    });

    it('manages notifications', () => {
      renderWithProvider(<AppTestComponent />);
      
      // Add notification
      fireEvent.click(screen.getByTestId('add-notification'));
      expect(screen.getByTestId('notification-count')).toHaveTextContent('1');
      
      // Add another notification
      fireEvent.click(screen.getByTestId('add-notification'));
      expect(screen.getByTestId('notification-count')).toHaveTextContent('2');
      
      // Clear all notifications
      fireEvent.click(screen.getByTestId('clear-notifications'));
      expect(screen.getByTestId('notification-count')).toHaveTextContent('0');
    });

    it('removes individual notifications', () => {
      renderWithProvider(<AppTestComponent />);
      
      // Add notifications
      fireEvent.click(screen.getByTestId('add-notification'));
      fireEvent.click(screen.getByTestId('add-notification'));
      
      expect(screen.getByTestId('notification-count')).toHaveTextContent('2');
      
      // Remove first notification
      const removeButtons = screen.getAllByText('Remove');
      fireEvent.click(removeButtons[0]);
      
      expect(screen.getByTestId('notification-count')).toHaveTextContent('1');
    });

    it('manages loading state', () => {
      renderWithProvider(<AppTestComponent />);
      
      fireEvent.click(screen.getByTestId('set-loading'));
      expect(screen.getByTestId('loading-status')).toHaveTextContent('Loading');
    });

    it('manages error state', () => {
      renderWithProvider(<AppTestComponent />);
      
      fireEvent.click(screen.getByTestId('set-error'));
      expect(screen.getByTestId('error-message')).toHaveTextContent('Test error');
    });

    it('updates user preferences', () => {
      const { result } = renderHook(() => useApp(), {
        wrapper: AppProvider,
      });
      
      // Set user first
      act(() => {
        result.current.setUser({ id: '1', name: 'Test User', preferences: { theme: 'light' } });
      });
      
      expect(result.current.state.user?.preferences.theme).toBe('light');
      
      // Update preferences
      act(() => {
        result.current.updateUserPreferences({ theme: 'dark', language: 'en' });
      });
      
      expect(result.current.state.user?.preferences.theme).toBe('dark');
      expect(result.current.state.user?.preferences.language).toBe('en');
    });

    it('handles preference updates with no user', () => {
      const { result } = renderHook(() => useApp(), {
        wrapper: AppProvider,
      });
      
      // Try to update preferences with no user
      act(() => {
        result.current.updateUserPreferences({ theme: 'dark' });
      });
      
      expect(result.current.state.user).toBeNull();
    });

    it('throws error when used outside provider', () => {
      const TestComponent = () => {
        useApp();
        return <div>Test</div>;
      };
      
      expect(() => render(<TestComponent />)).toThrow(
        'useApp must be used within AppProvider'
      );
    });
  });

  describe('WebSocketContext', () => {
    // Mock WebSocket for testing
    const mockWebSocket = {
      send: jest.fn(),
      close: jest.fn(),
      onopen: null as any,
      onclose: null as any,
      onmessage: null as any,
      onerror: null as any,
    };

    beforeEach(() => {
      (global as any).WebSocket = jest.fn(() => mockWebSocket);
      jest.clearAllMocks();
    });

    const renderWithProvider = (component: React.ReactElement) => {
      return render(<WebSocketProvider>{component}</WebSocketProvider>);
    };

    it('provides initial WebSocket state', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      expect(result.current.socket).toBeNull();
      expect(result.current.isConnected).toBe(false);
      expect(result.current.connectionAttempts).toBe(0);
      expect(result.current.maxReconnectAttempts).toBe(5);
    });

    it('connects to WebSocket', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
      });
      
      expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8080');
      expect(result.current.socket).toBe(mockWebSocket);
    });

    it('handles WebSocket connection open', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
      });
      
      act(() => {
        mockWebSocket.onopen();
      });
      
      expect(result.current.isConnected).toBe(true);
    });

    it('handles WebSocket connection close', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
      });
      
      act(() => {
        mockWebSocket.onopen();
      });
      
      expect(result.current.isConnected).toBe(true);
      
      act(() => {
        mockWebSocket.onclose();
      });
      
      expect(result.current.isConnected).toBe(false);
    });

    it('sends messages through WebSocket', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
        mockWebSocket.onopen();
      });
      
      act(() => {
        result.current.sendMessage({ type: 'test', data: 'hello' });
      });
      
      expect(mockWebSocket.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'test', data: 'hello' })
      );
    });

    it('handles string messages', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
        mockWebSocket.onopen();
      });
      
      act(() => {
        result.current.sendMessage('hello');
      });
      
      expect(mockWebSocket.send).toHaveBeenCalledWith('hello');
    });

    it('receives and parses JSON messages', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
      });
      
      const messageData = { type: 'response', data: 'test' };
      
      act(() => {
        mockWebSocket.onmessage({
          data: JSON.stringify(messageData),
        });
      });
      
      expect(result.current.lastMessage).toEqual(messageData);
    });

    it('handles non-JSON messages', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
      });
      
      act(() => {
        mockWebSocket.onmessage({ data: 'plain text message' });
      });
      
      expect(result.current.lastMessage).toBe('plain text message');
    });

    it('disconnects WebSocket', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
        mockWebSocket.onopen();
      });
      
      expect(result.current.isConnected).toBe(true);
      
      act(() => {
        result.current.disconnect();
      });
      
      expect(mockWebSocket.close).toHaveBeenCalled();
      expect(result.current.socket).toBeNull();
      expect(result.current.isConnected).toBe(false);
    });

    it('attempts reconnection', () => {
      const { result } = renderHook(() => useWebSocket(), {
        wrapper: WebSocketProvider,
      });
      
      act(() => {
        result.current.connect('ws://localhost:8080');
      });
      
      act(() => {
        result.current.reconnect();
      });
      
      expect(global.WebSocket).toHaveBeenCalledTimes(2);
    });

    it('throws error when used outside provider', () => {
      const TestComponent = () => {
        useWebSocket();
        return <div>Test</div>;
      };
      
      expect(() => render(<TestComponent />)).toThrow(
        'useWebSocket must be used within WebSocketProvider'
      );
    });
  });

  describe('Context Integration', () => {
    const CombinedProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
      return (
        <AppProvider>
          <ThemeProvider>
            <LLMProvider>
              <WebSocketProvider>
                {children}
              </WebSocketProvider>
            </LLMProvider>
          </ThemeProvider>
        </AppProvider>
      );
    };

    it('integrates multiple contexts', () => {
      const TestComponent = () => {
        const { state } = useApp();
        const { theme } = useTheme();
        const { config } = useLLM();
        const { isConnected } = useWebSocket();
        
        return (
          <div>
            <div data-testid="app-loading">{state.isLoading ? 'loading' : 'idle'}</div>
            <div data-testid="theme">{theme}</div>
            <div data-testid="llm-model">{config.model}</div>
            <div data-testid="ws-connected">{isConnected ? 'connected' : 'disconnected'}</div>
          </div>
        );
      };
      
      render(
        <CombinedProvider>
          <TestComponent />
        </CombinedProvider>
      );
      
      expect(screen.getByTestId('app-loading')).toHaveTextContent('idle');
      expect(screen.getByTestId('theme')).toHaveTextContent('auto');
      expect(screen.getByTestId('llm-model')).toHaveTextContent('gpt-3.5-turbo');
      expect(screen.getByTestId('ws-connected')).toHaveTextContent('disconnected');
    });

    it('maintains context isolation', () => {
      const { result: appResult } = renderHook(() => useApp(), {
        wrapper: ({ children }) => <AppProvider>{children}</AppProvider>,
      });
      
      const { result: themeResult } = renderHook(() => useTheme(), {
        wrapper: ({ children }) => <ThemeProvider>{children}</ThemeProvider>,
      });
      
      act(() => {
        appResult.current.setLoading(true);
      });
      
      act(() => {
        themeResult.current.setTheme('dark');
      });
      
      expect(appResult.current.state.isLoading).toBe(true);
      expect(themeResult.current.theme).toBe('dark');
    });
  });
});
