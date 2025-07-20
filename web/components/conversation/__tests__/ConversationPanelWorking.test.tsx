/**
 * Working test implementation for ConversationPanel - following TDD
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ConversationPanel from "../ConversationPanel";
import type { Agent } from "@/lib/types";

// Mock WebSocket client
const mockWsClient = {
  connect: jest.fn().mockResolvedValue(undefined),
  send: jest.fn(),
  subscribe: jest.fn(() => jest.fn()),
  getConnectionState: jest.fn(() => "connected"),
};

jest.mock("@/lib/websocket-client", () => ({
  getWebSocketClient: jest.fn(() => mockWsClient),
}));

// Mock use-agent-conversation hook
const mockUseAgentConversation = {
  sendMessage: jest.fn(),
  getConversationHistory: jest.fn(),
  createSession: jest.fn(),
  isLoading: false,
  error: null,
};

jest.mock("@/hooks/use-agent-conversation", () => ({
  useAgentConversation: jest.fn(() => mockUseAgentConversation),
}));

describe("ConversationPanel - Working Implementation", () => {
  const mockAgents: Agent[] = [
    {
      id: "agent-1",
      name: "Test Agent 1",
      template: "assistant",
      status: "active",
      pymdp_config: {},
      beliefs: {},
      preferences: {},
      metrics: {},
      parameters: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      inference_count: 0,
      total_steps: 0,
    },
    {
      id: "agent-2",
      name: "Test Agent 2",
      template: "specialist",
      status: "active",
      pymdp_config: {},
      beliefs: {},
      preferences: {},
      metrics: {},
      parameters: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      inference_count: 0,
      total_steps: 0,
    },
  ];

  const mockCurrentUser = {
    id: "user-1",
    name: "Test User",
    avatar: "/test-avatar.png",
  };

  const defaultProps = {
    conversationId: "conv-123",
    currentUser: mockCurrentUser,
    agents: mockAgents,
    messages: [],
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders without crashing", () => {
    render(<ConversationPanel {...defaultProps} />);
    expect(screen.getByText("Conversation")).toBeInTheDocument();
  });

  it("displays conversation title", () => {
    render(<ConversationPanel {...defaultProps} />);
    expect(screen.getByText("Conversation")).toBeInTheDocument();
  });

  it("shows connection status", () => {
    render(<ConversationPanel {...defaultProps} />);
    expect(screen.getByText("disconnected")).toBeInTheDocument();
  });

  it("shows participant count", () => {
    render(<ConversationPanel {...defaultProps} />);
    expect(screen.getByText("1 participants")).toBeInTheDocument();
  });

  it("has agent selector", () => {
    render(<ConversationPanel {...defaultProps} />);

    const select = screen.getByDisplayValue("Select Agent");
    expect(select).toBeInTheDocument();

    // Check that agents are in the options
    expect(screen.getByText("Test Agent 1")).toBeInTheDocument();
    expect(screen.getByText("Test Agent 2")).toBeInTheDocument();
  });

  it("has message input area", () => {
    render(<ConversationPanel {...defaultProps} />);

    expect(screen.getByPlaceholderText("Type your message...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  it("establishes WebSocket connection on mount", async () => {
    render(<ConversationPanel {...defaultProps} />);

    await waitFor(() => {
      expect(mockWsClient.connect).toHaveBeenCalled();
    });

    expect(mockWsClient.send).toHaveBeenCalledWith({
      type: "subscribe_conversation",
      conversation_id: "conv-123",
    });
  });

  it("subscribes to WebSocket events", () => {
    render(<ConversationPanel {...defaultProps} />);

    expect(mockWsClient.subscribe).toHaveBeenCalledWith(
      "conversation_message",
      expect.any(Function),
    );
    expect(mockWsClient.subscribe).toHaveBeenCalledWith("llm_response_chunk", expect.any(Function));
    expect(mockWsClient.subscribe).toHaveBeenCalledWith("user_typing", expect.any(Function));
  });

  it("sends typing indicator when typing", async () => {
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    await user.type(textarea, "Hello");

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith({
        type: "user_typing",
        conversation_id: "conv-123",
        is_typing: true,
      });
    });
  });

  it("sends message when Enter is pressed", async () => {
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    await user.type(textarea, "Test message");

    fireEvent.keyDown(textarea, { key: "Enter", code: "Enter" });

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith({
        type: "send_message",
        data: expect.objectContaining({
          conversation_id: "conv-123",
          content: "Test message",
          message_type: "user",
        }),
      });
    });
  });

  it("sends message when send button is clicked", async () => {
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    const sendButton = screen.getByRole("button", { name: /send/i });

    await user.type(textarea, "Test message");
    await user.click(sendButton);

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith({
        type: "send_message",
        data: expect.objectContaining({
          conversation_id: "conv-123",
          content: "Test message",
        }),
      });
    });
  });

  it("does not send empty messages", async () => {
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    const sendButton = screen.getByRole("button", { name: /send/i });
    await user.click(sendButton);

    expect(mockWsClient.send).not.toHaveBeenCalledWith({
      type: "send_message",
      data: expect.any(Object),
    });
  });

  it("allows shift+enter for new lines without sending", async () => {
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    await user.type(textarea, "Line 1");

    fireEvent.keyDown(textarea, {
      key: "Enter",
      code: "Enter",
      shiftKey: true,
    });

    expect(mockWsClient.send).not.toHaveBeenCalledWith({
      type: "send_message",
      data: expect.any(Object),
    });
  });

  it("uses API when agent is selected", async () => {
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    // Select an agent
    const select = screen.getByDisplayValue("Select Agent");
    await user.selectOptions(select, "agent-1");

    // Send a message
    const textarea = screen.getByPlaceholderText("Type your message...");
    await user.type(textarea, "Test message");

    fireEvent.keyDown(textarea, { key: "Enter", code: "Enter" });

    await waitFor(() => {
      expect(mockUseAgentConversation.sendMessage).toHaveBeenCalledWith("Test message");
    });
  });

  it("loads conversation history when agent is selected", async () => {
    const user = userEvent.setup();
    const mockHistory = [
      {
        conversation_id: "conv-1",
        prompt: "Previous message",
        response: "Previous response",
        created_at: new Date().toISOString(),
        provider: "test",
        token_count: 10,
        processing_time_ms: 100,
      },
    ];

    mockUseAgentConversation.getConversationHistory.mockResolvedValue(mockHistory);

    render(<ConversationPanel {...defaultProps} />);

    // Select an agent
    const select = screen.getByDisplayValue("Select Agent");
    await user.selectOptions(select, "agent-1");

    await waitFor(() => {
      expect(mockUseAgentConversation.getConversationHistory).toHaveBeenCalledWith(50, 24);
    });
  });

  it("shows connection lost warning when disconnected", () => {
    render(<ConversationPanel {...defaultProps} />);

    // The component starts with disconnected status
    expect(
      screen.getByText("Connection lost. Messages will be queued until reconnected."),
    ).toBeInTheDocument();
  });

  it("stops typing indicator after timeout", async () => {
    jest.useFakeTimers();
    const user = userEvent.setup();
    render(<ConversationPanel {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    await user.type(textarea, "Hello");

    // Fast forward time to trigger timeout
    jest.advanceTimersByTime(3000);

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith({
        type: "user_typing",
        conversation_id: "conv-123",
        is_typing: false,
      });
    });

    jest.useRealTimers();
  });

  it("calls onMessageSent callback when message is sent", async () => {
    const user = userEvent.setup();
    const onMessageSent = jest.fn();

    render(<ConversationPanel {...defaultProps} onSendMessage={onMessageSent} />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    await user.type(textarea, "Test message");

    fireEvent.keyDown(textarea, { key: "Enter", code: "Enter" });

    await waitFor(() => {
      expect(onMessageSent).toHaveBeenCalledWith(
        expect.objectContaining({
          content: "Test message",
          message_type: "user",
        }),
      );
    });
  });

  it("shows last activity timestamp", () => {
    render(<ConversationPanel {...defaultProps} />);

    expect(screen.getByText(/Last activity:/)).toBeInTheDocument();
  });
});
