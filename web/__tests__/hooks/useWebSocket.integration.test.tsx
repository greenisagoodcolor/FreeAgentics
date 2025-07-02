import React from "react";
import { renderHook, act } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { UseWebSocketOptions } from "@/hooks/useWebSocket";

// Mock the socketService
jest.mock("@/services/socketService", () => ({
  socketService: {
    connect: jest.fn(),
    disconnect: jest.fn(),
    send: jest.fn(() => true),
    sendMessage: jest.fn(),
    subscribeToAgent: jest.fn(),
    unsubscribeFromAgent: jest.fn(),
    subscribeToConversation: jest.fn(),
    unsubscribeFromConversation: jest.fn(),
    setTyping: jest.fn(),
    getConnectionStats: jest.fn(),
  },
}));

// Get references to the mocked functions
const mockSocketService = require("@/services/socketService").socketService;

// Create a test Redux store
const createTestStore = (initialState = {}) => {
  const defaultState = {
    connection: {
      status: {
        websocket: "disconnected",
        latency: null,
        reconnectAttempts: 0,
      },
      connectionId: null,
      errors: [],
      ...initialState,
    },
  };

  return configureStore({
    reducer: {
      connection: () => defaultState.connection,
    },
  });
};

// Helper to render hook with Redux provider
const renderUseWebSocket = (
  options?: UseWebSocketOptions,
  storeState?: any,
) => {
  const store = createTestStore(storeState);

  return renderHook(() => useWebSocket(options), {
    wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
  });
};

describe("useWebSocket Hook Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset mock implementations
    mockSocketService.send.mockReturnValue(true);
  });

  describe("Initialization", () => {
    it("returns correct initial state when disconnected", () => {
      const { result } = renderUseWebSocket();

      expect(result.current.isConnected).toBe(false);
      expect(result.current.isConnecting).toBe(false);
      expect(result.current.connectionId).toBe(null);
      expect(result.current.latency).toBe(null);
      expect(result.current.reconnectAttempts).toBe(0);
      expect(result.current.error).toBe(null);
    });

    it("auto-connects by default", () => {
      renderUseWebSocket();

      expect(mockSocketService.connect).toHaveBeenCalledTimes(1);
    });

    it("does not auto-connect when disabled", () => {
      renderUseWebSocket({ autoConnect: false });

      expect(mockSocketService.connect).not.toHaveBeenCalled();
    });

    it("handles legacy URL parameter", () => {
      const { result } = renderUseWebSocket("ws://test.com");

      expect(result.current.isConnected).toBe(false);
      expect(mockSocketService.connect).toHaveBeenCalled();
    });

    it("handles empty options object", () => {
      const { result } = renderUseWebSocket({});

      expect(result.current.isConnected).toBe(false);
      expect(mockSocketService.connect).toHaveBeenCalled();
    });

    it("handles undefined options", () => {
      const { result } = renderUseWebSocket(undefined);

      expect(result.current.isConnected).toBe(false);
      expect(mockSocketService.connect).toHaveBeenCalled();
    });
  });

  describe("Connection State Management", () => {
    it("reflects connected state correctly", () => {
      const { result } = renderUseWebSocket(undefined, {
        status: {
          websocket: "connected",
          latency: 50,
          reconnectAttempts: 0,
        },
        connectionId: "conn-123",
        errors: [],
      });

      expect(result.current.isConnected).toBe(true);
      expect(result.current.isConnecting).toBe(false);
      expect(result.current.connectionId).toBe("conn-123");
      expect(result.current.latency).toBe(50);
    });

    it("reflects connecting state correctly", () => {
      const { result } = renderUseWebSocket(undefined, {
        status: {
          websocket: "connecting",
          latency: null,
          reconnectAttempts: 0,
        },
        connectionId: null,
        errors: [],
      });

      expect(result.current.isConnected).toBe(false);
      expect(result.current.isConnecting).toBe(true);
    });

    it("reflects reconnect attempts correctly", () => {
      const { result } = renderUseWebSocket(undefined, {
        status: {
          websocket: "disconnected",
          latency: null,
          reconnectAttempts: 3,
        },
        connectionId: null,
        errors: [],
      });

      expect(result.current.reconnectAttempts).toBe(3);
    });

    it("handles error state correctly", () => {
      const { result } = renderUseWebSocket(undefined, {
        status: {
          websocket: "disconnected",
          latency: null,
          reconnectAttempts: 0,
        },
        connectionId: null,
        errors: [
          { message: "Old error", timestamp: Date.now() - 1000 },
          { message: "Latest error", timestamp: Date.now() },
        ],
      });

      expect(result.current.error).toBe("Latest error");
    });

    it("returns null error when no errors exist", () => {
      const { result } = renderUseWebSocket(undefined, {
        status: {
          websocket: "disconnected",
          latency: null,
          reconnectAttempts: 0,
        },
        connectionId: null,
        errors: [],
      });

      expect(result.current.error).toBe(null);
    });
  });

  describe("Connection Methods", () => {
    it("provides connect method that calls socketService", () => {
      const { result } = renderUseWebSocket({ autoConnect: false });

      act(() => {
        result.current.connect();
      });

      expect(mockSocketService.connect).toHaveBeenCalledTimes(1);
    });

    it("provides disconnect method that calls socketService", () => {
      const { result } = renderUseWebSocket();

      act(() => {
        result.current.disconnect();
      });

      expect(mockSocketService.disconnect).toHaveBeenCalledTimes(1);
    });

    it("connect method is memoized", () => {
      const { result, rerender } = renderUseWebSocket();

      const firstConnect = result.current.connect;
      rerender();
      const secondConnect = result.current.connect;

      expect(firstConnect).toBe(secondConnect);
    });

    it("disconnect method is memoized", () => {
      const { result, rerender } = renderUseWebSocket();

      const firstDisconnect = result.current.disconnect;
      rerender();
      const secondDisconnect = result.current.disconnect;

      expect(firstDisconnect).toBe(secondDisconnect);
    });
  });

  describe("Communication Methods", () => {
    it("provides send method that calls socketService", () => {
      const { result } = renderUseWebSocket();
      const testMessage = { type: "test", data: "hello" };

      const success = result.current.send(testMessage);

      expect(mockSocketService.send).toHaveBeenCalledWith(testMessage);
      expect(success).toBe(true);
    });

    it("provides sendMessage method that calls socketService", () => {
      const { result } = renderUseWebSocket();

      act(() => {
        result.current.sendMessage("conv-123", "Hello world", "agent-456");
      });

      expect(mockSocketService.sendMessage).toHaveBeenCalledWith(
        "conv-123",
        "Hello world",
        "agent-456",
      );
    });

    it("send method returns boolean result", () => {
      mockSocketService.send.mockReturnValueOnce(false);
      const { result } = renderUseWebSocket();

      const success = result.current.send({ test: "data" });

      expect(success).toBe(false);
    });

    it("communication methods are memoized", () => {
      const { result, rerender } = renderUseWebSocket();

      const firstSend = result.current.send;
      const firstSendMessage = result.current.sendMessage;

      rerender();

      const secondSend = result.current.send;
      const secondSendMessage = result.current.sendMessage;

      expect(firstSend).toBe(secondSend);
      expect(firstSendMessage).toBe(secondSendMessage);
    });
  });

  describe("Subscription Methods", () => {
    it("provides agent subscription methods", () => {
      const { result } = renderUseWebSocket();

      act(() => {
        result.current.subscribeToAgent("agent-123");
      });

      expect(mockSocketService.subscribeToAgent).toHaveBeenCalledWith(
        "agent-123",
      );

      act(() => {
        result.current.unsubscribeFromAgent("agent-123");
      });

      expect(mockSocketService.unsubscribeFromAgent).toHaveBeenCalledWith(
        "agent-123",
      );
    });

    it("provides conversation subscription methods", () => {
      const { result } = renderUseWebSocket();

      act(() => {
        result.current.subscribeToConversation("conv-456");
      });

      expect(mockSocketService.subscribeToConversation).toHaveBeenCalledWith(
        "conv-456",
      );

      act(() => {
        result.current.unsubscribeFromConversation("conv-456");
      });

      expect(
        mockSocketService.unsubscribeFromConversation,
      ).toHaveBeenCalledWith("conv-456");
    });

    it("subscription methods are memoized", () => {
      const { result, rerender } = renderUseWebSocket();

      const firstSubscribeAgent = result.current.subscribeToAgent;
      const firstUnsubscribeAgent = result.current.unsubscribeFromAgent;
      const firstSubscribeConv = result.current.subscribeToConversation;
      const firstUnsubscribeConv = result.current.unsubscribeFromConversation;

      rerender();

      expect(result.current.subscribeToAgent).toBe(firstSubscribeAgent);
      expect(result.current.unsubscribeFromAgent).toBe(firstUnsubscribeAgent);
      expect(result.current.subscribeToConversation).toBe(firstSubscribeConv);
      expect(result.current.unsubscribeFromConversation).toBe(
        firstUnsubscribeConv,
      );
    });
  });

  describe("Interaction Methods", () => {
    it("provides setTyping method", () => {
      const { result } = renderUseWebSocket();

      act(() => {
        result.current.setTyping("conv-123", "agent-456", true);
      });

      expect(mockSocketService.setTyping).toHaveBeenCalledWith(
        "conv-123",
        "agent-456",
        true,
      );
    });

    it("provides getConnectionStats method", () => {
      const { result } = renderUseWebSocket();

      act(() => {
        result.current.getConnectionStats();
      });

      expect(mockSocketService.getConnectionStats).toHaveBeenCalledTimes(1);
    });

    it("interaction methods are memoized", () => {
      const { result, rerender } = renderUseWebSocket();

      const firstSetTyping = result.current.setTyping;
      const firstGetStats = result.current.getConnectionStats;

      rerender();

      expect(result.current.setTyping).toBe(firstSetTyping);
      expect(result.current.getConnectionStats).toBe(firstGetStats);
    });
  });

  describe("Callback Handling", () => {
    it("calls onConnect callback when connection state changes to connected", () => {
      const onConnect = jest.fn();

      // Start with connected state to trigger callback
      renderUseWebSocket(
        { onConnect },
        {
          status: {
            websocket: "connected",
            latency: null,
            reconnectAttempts: 0,
          },
          connectionId: "test-123",
          errors: [],
        },
      );

      expect(onConnect).toHaveBeenCalledTimes(1);
    });

    it("calls onDisconnect callback when connection state changes to disconnected", () => {
      const onDisconnect = jest.fn();

      // Start with disconnected state to trigger callback
      renderUseWebSocket(
        { onDisconnect },
        {
          status: {
            websocket: "disconnected",
            latency: null,
            reconnectAttempts: 0,
          },
          connectionId: null,
          errors: [],
        },
      );

      expect(onDisconnect).toHaveBeenCalledTimes(1);
    });

    it("calls onError callback when errors are added", () => {
      const onError = jest.fn();

      // Start with an error to trigger callback
      renderUseWebSocket(
        { onError },
        {
          status: {
            websocket: "disconnected",
            latency: null,
            reconnectAttempts: 0,
          },
          connectionId: null,
          errors: [{ message: "Connection failed", timestamp: Date.now() }],
        },
      );

      expect(onError).toHaveBeenCalledWith("Connection failed");
    });

    it("does not call callbacks when they are not provided", () => {
      const store = createTestStore();

      // Should not throw when callbacks are undefined
      expect(() => {
        renderHook(() => useWebSocket({}), {
          wrapper: ({ children }) => (
            <Provider store={store}>{children}</Provider>
          ),
        });
      }).not.toThrow();
    });
  });

  describe("Auto-Connection Logic", () => {
    it("connects when autoConnect is true and not already connected", () => {
      renderUseWebSocket({ autoConnect: true });

      expect(mockSocketService.connect).toHaveBeenCalledTimes(1);
    });

    it("does not connect when already connected", () => {
      renderUseWebSocket(
        { autoConnect: true },
        {
          status: {
            websocket: "connected",
            latency: null,
            reconnectAttempts: 0,
          },
          connectionId: "test-123",
          errors: [],
        },
      );

      expect(mockSocketService.connect).not.toHaveBeenCalled();
    });

    it("connects even when status is connecting (current behavior)", () => {
      renderUseWebSocket(
        { autoConnect: true },
        {
          status: {
            websocket: "connecting",
            latency: null,
            reconnectAttempts: 0,
          },
          connectionId: null,
          errors: [],
        },
      );

      // Current implementation calls connect() for any status !== 'connected'
      expect(mockSocketService.connect).toHaveBeenCalled();
    });

    it("respects autoConnect changes", () => {
      // Test with autoConnect disabled
      renderUseWebSocket({ autoConnect: false });
      expect(mockSocketService.connect).not.toHaveBeenCalled();

      // Clear mocks and test with autoConnect enabled
      jest.clearAllMocks();
      renderUseWebSocket({ autoConnect: true });
      expect(mockSocketService.connect).toHaveBeenCalledTimes(1);
    });
  });

  describe("Edge Cases", () => {
    it("handles rapid state changes", () => {
      const { result } = renderUseWebSocket();

      // Rapidly call methods
      act(() => {
        result.current.connect();
        result.current.disconnect();
        result.current.send({ test: "data" });
        result.current.subscribeToAgent("agent-1");
        result.current.unsubscribeFromAgent("agent-1");
      });

      expect(mockSocketService.connect).toHaveBeenCalledTimes(2); // Initial + manual
      expect(mockSocketService.disconnect).toHaveBeenCalledTimes(1);
      expect(mockSocketService.send).toHaveBeenCalledTimes(1);
      expect(mockSocketService.subscribeToAgent).toHaveBeenCalledTimes(1);
      expect(mockSocketService.unsubscribeFromAgent).toHaveBeenCalledTimes(1);
    });

    it("handles multiple errors in sequence", () => {
      const onError = jest.fn();

      renderUseWebSocket(
        { onError },
        {
          status: {
            websocket: "disconnected",
            latency: null,
            reconnectAttempts: 0,
          },
          connectionId: null,
          errors: [
            { message: "First error", timestamp: Date.now() - 2000 },
            { message: "Second error", timestamp: Date.now() - 1000 },
            { message: "Latest error", timestamp: Date.now() },
          ],
        },
      );

      expect(onError).toHaveBeenCalledWith("Latest error");
    });

    it("handles empty string parameters", () => {
      const { result } = renderUseWebSocket();

      expect(() => {
        act(() => {
          result.current.sendMessage("", "", "");
          result.current.subscribeToAgent("");
          result.current.setTyping("", "", false);
        });
      }).not.toThrow();
    });

    it("handles null/undefined parameters gracefully", () => {
      const { result } = renderUseWebSocket();

      expect(() => {
        act(() => {
          result.current.send(null);
          result.current.send(undefined);
        });
      }).not.toThrow();
    });

    it("handles very large message objects", () => {
      const { result } = renderUseWebSocket();
      const largeMessage = {
        data: "x".repeat(10000),
        nested: {
          array: new Array(1000).fill("test"),
        },
      };

      expect(() => {
        result.current.send(largeMessage);
      }).not.toThrow();

      expect(mockSocketService.send).toHaveBeenCalledWith(largeMessage);
    });
  });

  describe("Performance", () => {
    it("maintains stable references for methods", () => {
      const { result, rerender } = renderUseWebSocket();

      const methods = {
        connect: result.current.connect,
        disconnect: result.current.disconnect,
        send: result.current.send,
        sendMessage: result.current.sendMessage,
        subscribeToAgent: result.current.subscribeToAgent,
        unsubscribeFromAgent: result.current.unsubscribeFromAgent,
        subscribeToConversation: result.current.subscribeToConversation,
        unsubscribeFromConversation: result.current.unsubscribeFromConversation,
        setTyping: result.current.setTyping,
        getConnectionStats: result.current.getConnectionStats,
      };

      // Re-render multiple times
      for (let i = 0; i < 5; i++) {
        rerender();
      }

      // All methods should maintain their references
      expect(result.current.connect).toBe(methods.connect);
      expect(result.current.disconnect).toBe(methods.disconnect);
      expect(result.current.send).toBe(methods.send);
      expect(result.current.sendMessage).toBe(methods.sendMessage);
      expect(result.current.subscribeToAgent).toBe(methods.subscribeToAgent);
      expect(result.current.unsubscribeFromAgent).toBe(
        methods.unsubscribeFromAgent,
      );
      expect(result.current.subscribeToConversation).toBe(
        methods.subscribeToConversation,
      );
      expect(result.current.unsubscribeFromConversation).toBe(
        methods.unsubscribeFromConversation,
      );
      expect(result.current.setTyping).toBe(methods.setTyping);
      expect(result.current.getConnectionStats).toBe(
        methods.getConnectionStats,
      );
    });

    it("handles frequent state updates efficiently", () => {
      const { result } = renderUseWebSocket();

      const startTime = performance.now();

      // Perform many operations
      act(() => {
        for (let i = 0; i < 100; i++) {
          result.current.send({ iteration: i });
          result.current.subscribeToAgent(`agent-${i}`);
          result.current.setTyping("conv", `agent-${i}`, i % 2 === 0);
        }
      });

      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100); // Should be fast
      expect(mockSocketService.send).toHaveBeenCalledTimes(100);
    });
  });

  describe("Hook Cleanup", () => {
    it("unmounts without errors", () => {
      const { unmount } = renderUseWebSocket();

      expect(() => unmount()).not.toThrow();
    });

    it("handles multiple mount/unmount cycles", () => {
      for (let i = 0; i < 5; i++) {
        const { unmount } = renderUseWebSocket();
        unmount();
      }

      // Should not accumulate errors or memory leaks
      expect(mockSocketService.connect).toHaveBeenCalledTimes(5);
    });

    it("maintains functionality after re-mounting", () => {
      const { unmount } = renderUseWebSocket();
      unmount();

      const { result } = renderUseWebSocket();

      act(() => {
        result.current.send({ test: "after remount" });
      });

      expect(mockSocketService.send).toHaveBeenCalledWith({
        test: "after remount",
      });
    });
  });
});
