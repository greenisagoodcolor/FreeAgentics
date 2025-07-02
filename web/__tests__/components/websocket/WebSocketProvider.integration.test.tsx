import React from "react";
import { render, screen, act } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import { WebSocketProvider } from "@/components/WebSocketProvider";

// Mock the useWebSocket hook
const mockConnect = jest.fn();
const mockDisconnect = jest.fn();
const mockSend = jest.fn();
const mockSendMessage = jest.fn();
const mockSubscribeToAgent = jest.fn();
const mockUnsubscribeFromAgent = jest.fn();
const mockSubscribeToConversation = jest.fn();
const mockUnsubscribeFromConversation = jest.fn();
const mockSetTyping = jest.fn();
const mockGetConnectionStats = jest.fn();

const mockWebSocketReturnValue = {
  isConnected: false,
  isConnecting: false,
  connectionId: null,
  latency: null,
  reconnectAttempts: 0,
  error: null,
  connect: mockConnect,
  disconnect: mockDisconnect,
  send: mockSend,
  sendMessage: mockSendMessage,
  subscribeToAgent: mockSubscribeToAgent,
  unsubscribeFromAgent: mockUnsubscribeFromAgent,
  subscribeToConversation: mockSubscribeToConversation,
  unsubscribeFromConversation: mockUnsubscribeFromConversation,
  setTyping: mockSetTyping,
  getConnectionStats: mockGetConnectionStats,
};

jest.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: jest.fn(() => mockWebSocketReturnValue),
}));

// Mock console methods to track calls
const mockConsoleLog = jest.fn();
const mockConsoleError = jest.fn();

// Override console methods
const originalConsoleLog = console.log;
const originalConsoleError = console.error;

beforeAll(() => {
  console.log = mockConsoleLog;
  console.error = mockConsoleError;
});

afterAll(() => {
  console.log = originalConsoleLog;
  console.error = originalConsoleError;
});

// Create a minimal Redux store for testing
const createTestStore = () => {
  return configureStore({
    reducer: {
      connection: () => ({
        status: {
          websocket: "disconnected",
          latency: null,
          reconnectAttempts: 0,
        },
        connectionId: null,
        errors: [],
      }),
    },
  });
};

describe("WebSocketProvider Integration Tests", () => {
  let store: ReturnType<typeof createTestStore>;
  const useWebSocketMock = require("@/hooks/useWebSocket").useWebSocket;

  beforeEach(() => {
    jest.clearAllMocks();
    store = createTestStore();

    // Reset the mock to default values
    useWebSocketMock.mockReturnValue(mockWebSocketReturnValue);
  });

  afterEach(() => {
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
  });

  const renderWebSocketProvider = (props = {}) => {
    const defaultProps = {
      children: <div data-testid="child-content">Test Content</div>,
      ...props,
    };

    return render(
      <Provider store={store}>
        <WebSocketProvider {...defaultProps} />
      </Provider>,
    );
  };

  describe("Component Rendering", () => {
    it("renders children correctly", () => {
      renderWebSocketProvider();

      expect(screen.getByTestId("child-content")).toBeInTheDocument();
      expect(screen.getByText("Test Content")).toBeInTheDocument();
    });

    it("does not show connection status by default", () => {
      renderWebSocketProvider();

      expect(
        screen.queryByText(/Connected|Connecting|Disconnected/),
      ).not.toBeInTheDocument();
    });

    it("shows connection status when enabled", () => {
      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("Disconnected")).toBeInTheDocument();
    });

    it("handles server-side rendering gracefully", () => {
      // Mock the mounted state to simulate server-side rendering
      const useState = jest.spyOn(React, "useState");
      useState.mockReturnValueOnce([false, jest.fn()]); // mounted = false

      renderWebSocketProvider();

      expect(screen.getByTestId("child-content")).toBeInTheDocument();

      useState.mockRestore();
    });
  });

  describe("WebSocket Integration", () => {
    it("initializes useWebSocket with default options", () => {
      renderWebSocketProvider();

      expect(useWebSocketMock).toHaveBeenCalledWith({
        autoConnect: true,
        onConnect: expect.any(Function),
        onDisconnect: expect.any(Function),
        onError: expect.any(Function),
      });
    });

    it("respects autoConnect prop", () => {
      renderWebSocketProvider({ autoConnect: false });

      expect(useWebSocketMock).toHaveBeenCalledWith({
        autoConnect: false,
        onConnect: expect.any(Function),
        onDisconnect: expect.any(Function),
        onError: expect.any(Function),
      });
    });

    it("logs connection success", () => {
      renderWebSocketProvider();

      // Get the onConnect callback and call it
      const onConnectCallback = useWebSocketMock.mock.calls[0][0].onConnect;
      act(() => {
        onConnectCallback();
      });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        "🟢 WebSocket connected successfully",
      );
    });

    it("logs disconnection", () => {
      renderWebSocketProvider();

      // Get the onDisconnect callback and call it
      const onDisconnectCallback =
        useWebSocketMock.mock.calls[0][0].onDisconnect;
      act(() => {
        onDisconnectCallback();
      });

      expect(mockConsoleLog).toHaveBeenCalledWith("🔴 WebSocket disconnected");
    });

    it("logs connection errors", () => {
      renderWebSocketProvider();

      // Get the onError callback and call it
      const onErrorCallback = useWebSocketMock.mock.calls[0][0].onError;
      const testError = "Connection failed";

      act(() => {
        onErrorCallback(testError);
      });

      expect(mockConsoleError).toHaveBeenCalledWith(
        "🟠 WebSocket error:",
        testError,
      );
    });
  });

  describe("Connection Status Indicator", () => {
    beforeEach(() => {
      // Enable connection status display for these tests
      renderWebSocketProvider({ showConnectionStatus: true });
    });

    it("displays disconnected status by default", () => {
      expect(screen.getByText("Disconnected")).toBeInTheDocument();

      const statusDot = document.querySelector(".bg-gray-500");
      expect(statusDot).toBeInTheDocument();
    });

    it("displays connecting status", () => {
      // Update mock to return connecting state
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        isConnecting: true,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("Connecting...")).toBeInTheDocument();

      const statusDot = document.querySelector(".bg-yellow-500");
      expect(statusDot).toBeInTheDocument();
    });

    it("displays connected status with latency", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        isConnected: true,
        latency: 50,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("Connected (50ms)")).toBeInTheDocument();

      const statusDot = document.querySelector(".bg-green-500");
      expect(statusDot).toBeInTheDocument();
    });

    it("displays connected status without latency", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        isConnected: true,
        latency: null,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("Connected")).toBeInTheDocument();
    });

    it("displays error status", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        error: "Connection timeout",
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("Error: Connection timeout")).toBeInTheDocument();

      const statusDot = document.querySelector(".bg-red-500");
      expect(statusDot).toBeInTheDocument();
    });

    it("displays reconnect attempts", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        reconnectAttempts: 3,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("(Retry 3)")).toBeInTheDocument();
    });

    it("does not display reconnect attempts when zero", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        reconnectAttempts: 0,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.queryByText(/Retry/)).not.toBeInTheDocument();
    });

    it("applies correct CSS classes for status indicator", () => {
      const statusContainer = document.querySelector(
        ".px-3.py-2.bg-gray-900.rounded-lg.border.border-gray-700.text-sm",
      );
      expect(statusContainer).toBeInTheDocument();

      const statusText = document.querySelector(".text-gray-200");
      expect(statusText).toBeInTheDocument();
    });

    it("positions status indicator correctly", () => {
      const statusContainer = document.querySelector(
        ".fixed.top-4.right-4.z-50",
      );
      expect(statusContainer).toBeInTheDocument();
    });
  });

  describe("Prop Handling", () => {
    it("handles multiple children", () => {
      render(
        <Provider store={store}>
          <WebSocketProvider>
            <div data-testid="child-1">Child 1</div>
            <div data-testid="child-2">Child 2</div>
          </WebSocketProvider>
        </Provider>,
      );

      expect(screen.getByTestId("child-1")).toBeInTheDocument();
      expect(screen.getByTestId("child-2")).toBeInTheDocument();
    });

    it("handles complex children structures", () => {
      render(
        <Provider store={store}>
          <WebSocketProvider>
            <div>
              <header data-testid="header">Header</header>
              <main data-testid="main">
                <section data-testid="section">Section Content</section>
              </main>
            </div>
          </WebSocketProvider>
        </Provider>,
      );

      expect(screen.getByTestId("header")).toBeInTheDocument();
      expect(screen.getByTestId("main")).toBeInTheDocument();
      expect(screen.getByTestId("section")).toBeInTheDocument();
    });

    it("passes props correctly to useWebSocket", () => {
      const customProps = {
        autoConnect: false,
        showConnectionStatus: true,
      };

      renderWebSocketProvider(customProps);

      expect(useWebSocketMock).toHaveBeenCalledWith({
        autoConnect: false,
        onConnect: expect.any(Function),
        onDisconnect: expect.any(Function),
        onError: expect.any(Function),
      });
    });
  });

  describe("State Changes", () => {
    it("updates status when connection state changes", () => {
      const { rerender } = renderWebSocketProvider({
        showConnectionStatus: true,
      });

      // Initially disconnected
      expect(screen.getByText("Disconnected")).toBeInTheDocument();

      // Update to connecting
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        isConnecting: true,
      });

      rerender(
        <Provider store={store}>
          <WebSocketProvider showConnectionStatus={true}>
            <div data-testid="child-content">Test Content</div>
          </WebSocketProvider>
        </Provider>,
      );

      expect(screen.getByText("Connecting...")).toBeInTheDocument();
    });

    it("handles dynamic showConnectionStatus prop", () => {
      const { rerender } = renderWebSocketProvider({
        showConnectionStatus: false,
      });

      // Initially hidden
      expect(screen.queryByText("Disconnected")).not.toBeInTheDocument();

      // Show status
      rerender(
        <Provider store={store}>
          <WebSocketProvider showConnectionStatus={true}>
            <div data-testid="child-content">Test Content</div>
          </WebSocketProvider>
        </Provider>,
      );

      expect(screen.getByText("Disconnected")).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("handles null children gracefully", () => {
      render(
        <Provider store={store}>
          <WebSocketProvider>{null}</WebSocketProvider>
        </Provider>,
      );

      // Should not throw error
      expect(document.body).toBeInTheDocument();
    });

    it("handles undefined children gracefully", () => {
      render(
        <Provider store={store}>
          <WebSocketProvider>{undefined}</WebSocketProvider>
        </Provider>,
      );

      // Should not throw error
      expect(document.body).toBeInTheDocument();
    });

    it("handles empty children gracefully", () => {
      render(
        <Provider store={store}>
          <WebSocketProvider>{""}</WebSocketProvider>
        </Provider>,
      );

      // Should not throw error
      expect(document.body).toBeInTheDocument();
    });

    it("handles boolean children gracefully", () => {
      render(
        <Provider store={store}>
          <WebSocketProvider>
            {true && <div data-testid="conditional">Conditional Content</div>}
            {false && <div data-testid="hidden">Hidden Content</div>}
          </WebSocketProvider>
        </Provider>,
      );

      expect(screen.getByTestId("conditional")).toBeInTheDocument();
      expect(screen.queryByTestId("hidden")).not.toBeInTheDocument();
    });

    it("handles very long error messages", () => {
      const longError = "A".repeat(200);

      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        error: longError,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText(`Error: ${longError}`)).toBeInTheDocument();
    });

    it("handles high reconnect attempt counts", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        reconnectAttempts: 999,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("(Retry 999)")).toBeInTheDocument();
    });

    it("handles very high latency values", () => {
      useWebSocketMock.mockReturnValue({
        ...mockWebSocketReturnValue,
        isConnected: true,
        latency: 5000,
      });

      renderWebSocketProvider({ showConnectionStatus: true });

      expect(screen.getByText("Connected (5000ms)")).toBeInTheDocument();
    });
  });

  describe("Performance", () => {
    it("renders efficiently with multiple re-renders", () => {
      const { rerender } = renderWebSocketProvider();

      const startTime = performance.now();

      // Perform multiple re-renders
      for (let i = 0; i < 10; i++) {
        rerender(
          <Provider store={store}>
            <WebSocketProvider>
              <div data-testid="child-content">Test Content {i}</div>
            </WebSocketProvider>
          </Provider>,
        );
      }

      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100); // Should be fast
      expect(screen.getByText("Test Content 9")).toBeInTheDocument();
    });

    it("handles rapid state changes efficiently", () => {
      renderWebSocketProvider({ showConnectionStatus: true });

      const states = [
        { isConnecting: true },
        { isConnected: true },
        { error: "Test error" },
        { isConnected: false, error: null },
      ];

      states.forEach((state, index) => {
        useWebSocketMock.mockReturnValue({
          ...mockWebSocketReturnValue,
          ...state,
        });

        // Force re-render by changing children
        render(
          <Provider store={store}>
            <WebSocketProvider showConnectionStatus={true}>
              <div data-testid={`child-${index}`}>Content {index}</div>
            </WebSocketProvider>
          </Provider>,
        );
      });

      // Should handle all state changes without errors
      expect(screen.getByTestId("child-3")).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("provides meaningful status information", () => {
      renderWebSocketProvider({ showConnectionStatus: true });

      const statusText = screen.getByText("Disconnected");
      expect(statusText).toBeInTheDocument();
      expect(statusText).toHaveClass("text-gray-200");
    });

    it("uses appropriate color coding for status", () => {
      const statusStates = [
        { isConnected: true, expectedColor: "bg-green-500" },
        { isConnecting: true, expectedColor: "bg-yellow-500" },
        { error: "Test error", expectedColor: "bg-red-500" },
        { isConnected: false, expectedColor: "bg-gray-500" },
      ];

      statusStates.forEach((state, index) => {
        useWebSocketMock.mockReturnValue({
          ...mockWebSocketReturnValue,
          ...state,
        });

        const { container } = render(
          <Provider store={store}>
            <WebSocketProvider showConnectionStatus={true}>
              <div>Test {index}</div>
            </WebSocketProvider>
          </Provider>,
        );

        const statusDot = container.querySelector(`.${state.expectedColor}`);
        expect(statusDot).toBeInTheDocument();
      });
    });

    it("maintains readable text contrast", () => {
      renderWebSocketProvider({ showConnectionStatus: true });

      // Status container should have dark background
      const statusContainer = document.querySelector(".bg-gray-900");
      expect(statusContainer).toBeInTheDocument();

      // Text should be light colored
      const statusText = document.querySelector(".text-gray-200");
      expect(statusText).toBeInTheDocument();
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts and unmounts without errors", () => {
      const { unmount } = renderWebSocketProvider();

      expect(screen.getByTestId("child-content")).toBeInTheDocument();

      expect(() => unmount()).not.toThrow();
    });

    it("handles prop changes gracefully", () => {
      const { rerender } = renderWebSocketProvider({
        autoConnect: true,
        showConnectionStatus: false,
      });

      expect(screen.queryByText("Disconnected")).not.toBeInTheDocument();

      rerender(
        <Provider store={store}>
          <WebSocketProvider autoConnect={false} showConnectionStatus={true}>
            <div data-testid="child-content">Updated Content</div>
          </WebSocketProvider>
        </Provider>,
      );

      expect(screen.getByText("Disconnected")).toBeInTheDocument();
      expect(screen.getByText("Updated Content")).toBeInTheDocument();
    });

    it("maintains consistent behavior across mount/unmount cycles", () => {
      for (let i = 0; i < 5; i++) {
        const { unmount } = renderWebSocketProvider();

        expect(screen.getByTestId("child-content")).toBeInTheDocument();
        expect(useWebSocketMock).toHaveBeenCalled();

        unmount();
      }
    });
  });
});
