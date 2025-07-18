import { renderHook, act } from "@testing-library/react";
import { useAgentConversation } from "@/hooks/use-agent-conversation";

describe("useAgentConversation", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should initialize with default state", () => {
    const { result } = renderHook(() => useAgentConversation());

    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.getConversationHistory()).toEqual([]);
  });

  it("should send message successfully", async () => {
    const { result } = renderHook(() => useAgentConversation());

    await act(async () => {
      await result.current.sendMessage("Hello, agent!");
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();

    const history = result.current.getConversationHistory();
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      message: "Hello, agent!",
      type: "user",
    });
  });

  it("should send message with agent ID", async () => {
    const { result } = renderHook(() => useAgentConversation());

    await act(async () => {
      await result.current.sendMessage("Hello, specific agent!", "agent-123");
    });

    const history = result.current.getConversationHistory();
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      message: "Hello, specific agent!",
      type: "user",
      agentId: "agent-123",
    });
  });

  it("should handle empty messages", async () => {
    const { result } = renderHook(() => useAgentConversation());

    await act(async () => {
      await result.current.sendMessage("");
    });

    const history = result.current.getConversationHistory();
    expect(history).toHaveLength(0);
  });

  it("should maintain conversation history", async () => {
    const { result } = renderHook(() => useAgentConversation());

    await act(async () => {
      await result.current.sendMessage("First message");
    });

    await act(async () => {
      await result.current.sendMessage("Second message");
    });

    await act(async () => {
      await result.current.sendMessage("Third message");
    });

    const history = result.current.getConversationHistory();
    expect(history).toHaveLength(3);
    expect(history[0].message).toBe("First message");
    expect(history[1].message).toBe("Second message");
    expect(history[2].message).toBe("Third message");
  });

  it("should filter history by agent ID", async () => {
    const { result } = renderHook(() => useAgentConversation());

    await act(async () => {
      await result.current.sendMessage("Message to agent 1", "agent-1");
      await result.current.sendMessage("Message to agent 2", "agent-2");
      await result.current.sendMessage("Another message to agent 1", "agent-1");
    });

    const agent1History = result.current.getConversationHistory("agent-1");
    expect(agent1History).toHaveLength(2);
    expect(agent1History[0].agentId).toBe("agent-1");
    expect(agent1History[1].agentId).toBe("agent-1");

    const agent2History = result.current.getConversationHistory("agent-2");
    expect(agent2History).toHaveLength(1);
    expect(agent2History[0].agentId).toBe("agent-2");
  });

  it("should create session", async () => {
    const { result } = renderHook(() => useAgentConversation());

    await act(async () => {
      await result.current.createSession("agent-123");
    });

    // Session creation is a mock for now
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("should handle errors gracefully", async () => {
    const { result } = renderHook(() => useAgentConversation());

    // Simulate an error by overriding the internal implementation
    const originalSendMessage = result.current.sendMessage;
    result.current.sendMessage = jest.fn().mockRejectedValueOnce(new Error("Network error"));

    await act(async () => {
      try {
        await result.current.sendMessage("This will fail");
      } catch (error) {
        // Expected error
      }
    });

    // Restore original implementation
    result.current.sendMessage = originalSendMessage;

    // The hook should handle errors internally
    expect(result.current.error).toBe("Failed to send message");
  });

  it("should track loading state", async () => {
    const { result } = renderHook(() => useAgentConversation());

    let loadingStateDuringRequest = false;

    const promise = act(async () => {
      const sendPromise = result.current.sendMessage("Test message");
      loadingStateDuringRequest = result.current.isLoading;
      await sendPromise;
    });

    await promise;

    expect(loadingStateDuringRequest).toBe(true);
    expect(result.current.isLoading).toBe(false);
  });

  it("should reset error on successful operation", async () => {
    const { result } = renderHook(() => useAgentConversation());

    // First, create an error state
    result.current.error = "Previous error";

    await act(async () => {
      await result.current.sendMessage("Success message");
    });

    expect(result.current.error).toBeNull();
  });
});
