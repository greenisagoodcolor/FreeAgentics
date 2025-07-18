import { renderHook, act, waitFor } from "@testing-library/react";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";
import { getApiClient } from "@/lib/api-client";

// Mock dependencies
jest.mock("@/lib/api-client");

const mockApiClient = {
  submitPrompt: jest.fn(),
  getPromptSuggestions: jest.fn(),
};

(getApiClient as jest.Mock).mockReturnValue(mockApiClient);

// Mock WebSocket
class MockWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event("open"));
      }
    }, 0);
  }

  send(data: string) {}
  close() {}
}

global.WebSocket = MockWebSocket as any;

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
    mockApiClient.submitPrompt.mockResolvedValueOnce({
      id: "prompt-123",
      prompt: "Test prompt",
      status: "processing",
      agents: [{ id: "agent-1", name: "Test Agent" }],
      knowledge_graph: {
        nodes: [{ id: "node-1", label: "Test Node", type: "concept" }],
        edges: [],
      },
    });

    const { result } = renderHook(() => usePromptProcessor());

    await act(async () => {
      await result.current.submitPrompt("Test prompt");
    });

    expect(mockApiClient.submitPrompt).toHaveBeenCalledWith("Test prompt");
    expect(result.current.agents).toHaveLength(1);
    expect(result.current.knowledgeGraph.nodes).toHaveLength(1);
    expect(result.current.isLoading).toBe(false);
  });

  it("should handle prompt submission error", async () => {
    const error = new Error("API Error");
    mockApiClient.submitPrompt.mockRejectedValueOnce(error);

    const { result } = renderHook(() => usePromptProcessor());

    await act(async () => {
      await result.current.submitPrompt("Test prompt");
    });

    expect(result.current.error).toBe("Failed to process prompt: API Error");
    expect(result.current.isLoading).toBe(false);
  });

  it("should fetch suggestions with debouncing", async () => {
    jest.useFakeTimers();

    mockApiClient.getPromptSuggestions.mockResolvedValueOnce({
      suggestions: ["How to test?", "What is testing?"],
    });

    const { result } = renderHook(() => usePromptProcessor());

    act(() => {
      result.current.fetchSuggestions("test");
    });

    // Should not call immediately
    expect(mockApiClient.getPromptSuggestions).not.toHaveBeenCalled();

    // Fast forward debounce timer
    act(() => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(mockApiClient.getPromptSuggestions).toHaveBeenCalledWith("test");
    });

    expect(result.current.suggestions).toEqual(["How to test?", "What is testing?"]);

    jest.useRealTimers();
  });

  it("should not fetch suggestions for empty query", () => {
    const { result } = renderHook(() => usePromptProcessor());

    act(() => {
      result.current.fetchSuggestions("");
    });

    expect(mockApiClient.getPromptSuggestions).not.toHaveBeenCalled();
  });

  it("should handle suggestion fetch error", async () => {
    jest.useFakeTimers();

    mockApiClient.getPromptSuggestions.mockRejectedValueOnce(new Error("API Error"));

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

  it("should retry failed prompt", async () => {
    const error = new Error("API Error");
    mockApiClient.submitPrompt.mockRejectedValueOnce(error).mockResolvedValueOnce({
      id: "prompt-123",
      prompt: "Test prompt",
      status: "processing",
      agents: [],
    });

    const { result } = renderHook(() => usePromptProcessor());

    // First attempt fails
    await act(async () => {
      await result.current.submitPrompt("Test prompt");
    });

    expect(result.current.error).toBeTruthy();

    // Retry succeeds
    await act(async () => {
      await result.current.retry();
    });

    expect(result.current.error).toBeNull();
    expect(mockApiClient.submitPrompt).toHaveBeenCalledTimes(2);
  });

  it("should reset conversation", () => {
    const { result } = renderHook(() => usePromptProcessor());

    // Set some state first
    act(() => {
      result.current.agents.push({ id: "agent-1", name: "Test" } as any);
      result.current.knowledgeGraph.nodes.push({ id: "node-1", label: "Test", type: "concept" });
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
      const ws = (result.current as any).wsRef?.current;
      expect(ws).toBeTruthy();
    });

    const ws = (result.current as any).wsRef.current as MockWebSocket;

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
      const ws = (result.current as any).wsRef?.current;
      expect(ws).toBeTruthy();
    });

    const ws = (result.current as any).wsRef.current as MockWebSocket;

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

    expect(result.current.knowledgeGraph.nodes).toHaveLength(1);
    expect(result.current.knowledgeGraph.edges).toHaveLength(1);
  });

  it("should cleanup WebSocket on unmount", () => {
    const { result, unmount } = renderHook(() => usePromptProcessor());

    const ws = (result.current as any).wsRef.current as MockWebSocket;
    const closeSpy = jest.spyOn(ws, "close");

    unmount();

    expect(closeSpy).toHaveBeenCalled();
  });
});
