import { renderHook, act, waitFor } from "@testing-library/react";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";
import { apiClient } from "@/lib/api-client";

// Mock dependencies
jest.mock("@/lib/api-client", () => ({
  apiClient: {
    processPrompt: jest.fn(),
    getSuggestions: jest.fn(),
  },
}));

const mockApiClient = apiClient as jest.Mocked<typeof apiClient>;

// Mock WebSocket
class MockWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  private listeners: Map<string, ((event: any) => void)[]> = new Map();

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event("open"));
      }
      const openListeners = this.listeners.get("open") || [];
      openListeners.forEach((listener) => listener(new Event("open")));
    }, 0);
  }

  addEventListener(type: string, listener: (event: any) => void) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, []);
    }
    this.listeners.get(type)!.push(listener);
  }

  removeEventListener(type: string, listener: (event: any) => void) {
    const listeners = this.listeners.get(type) || [];
    const index = listeners.indexOf(listener);
    if (index > -1) {
      listeners.splice(index, 1);
    }
  }

  send(_data: string) {}

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent("close"));
    }
    const closeListeners = this.listeners.get("close") || [];
    closeListeners.forEach((listener) => listener(new CloseEvent("close")));
  }
}

// Add WebSocket constants to MockWebSocket
(MockWebSocket as any).CONNECTING = 0;
(MockWebSocket as any).OPEN = 1;
(MockWebSocket as any).CLOSING = 2;
(MockWebSocket as any).CLOSED = 3;

global.WebSocket = MockWebSocket as unknown as {
  new (url: string | URL, protocols?: string | string[]): WebSocket;
  readonly CONNECTING: 0;
  readonly OPEN: 1;
  readonly CLOSING: 2;
  readonly CLOSED: 3;
};

describe("usePromptProcessor", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should initialize with default state", () => {
    const { result } = renderHook(() => usePromptProcessor());

    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.agents).toEqual([]);
    expect(result.current.knowledgeGraph).toEqual({ nodes: [], edges: [] });
    expect(result.current.suggestions).toEqual([]);
    expect(result.current.conversationId).toBeNull();
    expect(result.current.iterationContext).toBeNull();
  });

  it("should submit prompt successfully", async () => {
    mockApiClient.processPrompt.mockResolvedValueOnce({
      success: true,
      data: {
        agents: [
          { id: "agent-1", name: "Test Agent", description: "Test description", status: "active" },
        ],
        knowledgeGraph: {
          nodes: [{ id: "node-1", label: "Test Node", type: "concept" }],
          edges: [],
        },
        suggestions: [],
        conversationId: "conv-123",
      },
    });

    const { result } = renderHook(() => usePromptProcessor());

    await act(async () => {
      await result.current.submitPrompt("Test prompt");
    });

    expect(mockApiClient.processPrompt).toHaveBeenCalledWith({
      prompt: "Test prompt",
      conversationId: undefined,
    });
    expect(result.current.agents).toHaveLength(1);
    expect(result.current.knowledgeGraph?.nodes).toHaveLength(1);
    expect(result.current.isLoading).toBe(false);
  });

  it("should handle prompt submission error", async () => {
    mockApiClient.processPrompt.mockResolvedValueOnce({
      success: false,
      error: "API Error",
    });

    const { result } = renderHook(() => usePromptProcessor());

    await act(async () => {
      await result.current.submitPrompt("Test prompt");
    });

    expect(result.current.error).toBe("API Error");
    expect(result.current.isLoading).toBe(false);
  });

  it("should fetch suggestions with debouncing", async () => {
    jest.useFakeTimers();

    mockApiClient.getSuggestions.mockResolvedValueOnce({
      success: true,
      data: ["How to test?", "What is testing?"],
    });

    const { result } = renderHook(() => usePromptProcessor());

    act(() => {
      result.current.fetchSuggestions("test");
    });

    // Should not call immediately
    expect(mockApiClient.getSuggestions).not.toHaveBeenCalled();

    // Fast forward debounce timer and wait for the async operation
    await act(async () => {
      jest.advanceTimersByTime(300);
      // Wait for promises to resolve
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(mockApiClient.getSuggestions).toHaveBeenCalledWith("test");
    expect(result.current.suggestions).toEqual(["How to test?", "What is testing?"]);

    jest.useRealTimers();
  });

  it("should not fetch suggestions for empty query", () => {
    const { result } = renderHook(() => usePromptProcessor());

    act(() => {
      result.current.fetchSuggestions("");
    });

    expect(mockApiClient.getSuggestions).not.toHaveBeenCalled();
  });

  it("should handle suggestion fetch error", async () => {
    jest.useFakeTimers();

    mockApiClient.getSuggestions.mockRejectedValueOnce(new Error("API Error"));

    const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation();
    const { result } = renderHook(() => usePromptProcessor());

    act(() => {
      result.current.fetchSuggestions("test");
    });

    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        "Error fetching suggestions:",
        expect.any(Error),
      );
    });

    consoleErrorSpy.mockRestore();
    jest.useRealTimers();
  });

  it("should clear error on retry", async () => {
    mockApiClient.processPrompt.mockResolvedValueOnce({
      success: false,
      error: "API Error",
    });

    const { result } = renderHook(() => usePromptProcessor());

    // First attempt fails
    await act(async () => {
      await result.current.submitPrompt("Test prompt");
    });

    expect(result.current.error).toBe("API Error");

    // Retry just clears the error (doesn't actually retry the request)
    act(() => {
      result.current.retry();
    });

    expect(result.current.error).toBeNull();
    expect(mockApiClient.processPrompt).toHaveBeenCalledTimes(1); // Only called once
  });

  it("should reset conversation", () => {
    const { result } = renderHook(() => usePromptProcessor());

    // Set some state first
    act(() => {
      result.current.agents.push({
        id: "agent-1",
        name: "Test",
        type: "explorer",
        status: "active",
      });
      result.current.knowledgeGraph?.nodes.push({ id: "node-1", label: "Test", type: "concept" });
    });

    // Reset
    act(() => {
      result.current.resetConversation();
    });

    expect(result.current.agents).toEqual([]);
    expect(result.current.knowledgeGraph).toEqual({ nodes: [], edges: [] });
    expect(result.current.conversationId).toBeNull();
    expect(result.current.iterationContext).toBeNull();
  });

  it("should handle WebSocket messages", async () => {
    const { result } = renderHook(() => usePromptProcessor());

    // Wait for WebSocket connection
    await waitFor(() => {
      const ws = (result.current as unknown as { wsRef?: { current?: MockWebSocket } }).wsRef
        ?.current;
      expect(ws).toBeTruthy();
    });

    const ws = (result.current as unknown as { wsRef: { current: MockWebSocket } }).wsRef.current;

    // Simulate agent update message
    act(() => {
      if (ws.onmessage) {
        ws.onmessage(
          new MessageEvent("message", {
            data: JSON.stringify({
              type: "agent_update",
              agent: { id: "agent-1", name: "Updated Agent", status: "active" },
            }),
          }),
        );
      }
    });

    expect(result.current.agents).toHaveLength(1);
    expect(result.current.agents[0].name).toBe("Updated Agent");
  });

  it("should handle knowledge graph update via WebSocket", async () => {
    const { result } = renderHook(() => usePromptProcessor());

    // Wait for WebSocket connection
    await waitFor(() => {
      const ws = (result.current as unknown as { wsRef?: { current?: MockWebSocket } }).wsRef
        ?.current;
      expect(ws).toBeTruthy();
    });

    const ws = (result.current as unknown as { wsRef: { current: MockWebSocket } }).wsRef.current;

    // Simulate knowledge graph update
    act(() => {
      if (ws.onmessage) {
        ws.onmessage(
          new MessageEvent("message", {
            data: JSON.stringify({
              type: "knowledge_graph_update",
              graph: {
                nodes: [{ id: "node-1", label: "New Node", type: "concept" }],
                edges: [{ source: "node-1", target: "node-2", relationship: "relates_to" }],
              },
            }),
          }),
        );
      }
    });

    expect(result.current.knowledgeGraph?.nodes).toHaveLength(1);
    expect(result.current.knowledgeGraph?.edges).toHaveLength(1);
  });

  it("should cleanup WebSocket on unmount", () => {
    const { result, unmount } = renderHook(() => usePromptProcessor());

    const ws = (result.current as unknown as { wsRef: { current: MockWebSocket } }).wsRef.current;
    const closeSpy = jest.spyOn(ws, "close");

    unmount();

    expect(closeSpy).toHaveBeenCalled();
  });
});
