import React from "react";
import { renderHook, act } from "@testing-library/react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

// Mock Redux store for testing
const mockStore = configureStore({
  reducer: {
    connection: (
      state = {
        status: {
          websocket: "disconnected",
          latency: null,
          reconnectAttempts: 0,
        },
        connectionId: null,
        errors: [],
      },
    ) => state,
  },
});

// Mock socket service
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

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <Provider store={mockStore}>{children}</Provider>
);

describe("useWebSocket Hook", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Connection Management", () => {
    it("returns connection state from Redux", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });

      expect(result.current.isConnected).toBe(false);
      expect(result.current.isConnecting).toBe(false);
      expect(result.current.connectionId).toBe(null);
      expect(result.current.latency).toBe(null);
      expect(result.current.reconnectAttempts).toBe(0);
      expect(result.current.error).toBe(null);
    });

    it("provides connection methods", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });

      expect(typeof result.current.connect).toBe("function");
      expect(typeof result.current.disconnect).toBe("function");
    });

    it("handles options parameter", () => {
      const options = { autoConnect: false };
      const { result } = renderHook(() => useWebSocket(options), { wrapper });

      expect(result.current.isConnected).toBe(false);
    });
  });

  describe("Message Handling", () => {
    it("provides send methods", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });

      expect(typeof result.current.send).toBe("function");
      expect(typeof result.current.sendMessage).toBe("function");
    });

    it("sendMessage requires 3 parameters", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });

      act(() => {
        result.current.sendMessage("conv1", "hello", "agent1");
      });

      expect(
        require("@/services/socketService").socketService.sendMessage,
      ).toHaveBeenCalledWith("conv1", "hello", "agent1");
    });
  });

  describe("Subscription Methods", () => {
    it("provides subscription methods", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });

      expect(typeof result.current.subscribeToAgent).toBe("function");
      expect(typeof result.current.unsubscribeFromAgent).toBe("function");
      expect(typeof result.current.subscribeToConversation).toBe("function");
      expect(typeof result.current.unsubscribeFromConversation).toBe(
        "function",
      );
    });

    it("calls socket service methods correctly", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });
      const { socketService } = require("@/services/socketService");

      act(() => {
        result.current.subscribeToAgent("agent1");
        result.current.unsubscribeFromAgent("agent1");
        result.current.subscribeToConversation("conv1");
        result.current.unsubscribeFromConversation("conv1");
      });

      expect(socketService.subscribeToAgent).toHaveBeenCalledWith("agent1");
      expect(socketService.unsubscribeFromAgent).toHaveBeenCalledWith("agent1");
      expect(socketService.subscribeToConversation).toHaveBeenCalledWith(
        "conv1",
      );
      expect(socketService.unsubscribeFromConversation).toHaveBeenCalledWith(
        "conv1",
      );
    });
  });

  describe("Interaction Methods", () => {
    it("provides interaction methods", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });

      expect(typeof result.current.setTyping).toBe("function");
      expect(typeof result.current.getConnectionStats).toBe("function");
    });

    it("setTyping calls socket service correctly", () => {
      const { result } = renderHook(() => useWebSocket(), { wrapper });
      const { socketService } = require("@/services/socketService");

      act(() => {
        result.current.setTyping("conv1", "agent1", true);
      });

      expect(socketService.setTyping).toHaveBeenCalledWith(
        "conv1",
        "agent1",
        true,
      );
    });
  });

  describe("Auto Connect", () => {
    it("connects automatically by default", () => {
      const { socketService } = require("@/services/socketService");
      renderHook(() => useWebSocket(), { wrapper });

      expect(socketService.connect).toHaveBeenCalled();
    });

    it("does not connect when autoConnect is false", () => {
      const { socketService } = require("@/services/socketService");
      jest.clearAllMocks();

      renderHook(() => useWebSocket({ autoConnect: false }), { wrapper });

      expect(socketService.connect).not.toHaveBeenCalled();
    });
  });
});
