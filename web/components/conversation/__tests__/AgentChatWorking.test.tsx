/**
 * Working test implementation for AgentChat - following TDD
 */

import React from "react";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AgentChat from "../AgentChat";
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

describe("AgentChat - Working Implementation", () => {
  const mockAgent: Agent = {
    id: "agent-1",
    name: "Test Agent 1",
    template: "test",
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
    avatar: "/test-avatar-1.png",
  };

  // Mock agents - keeping for future test cases
  // const mockAgents: Agent[] = [
  //   mockAgent,
  //   {
  //     id: "agent-2",
  //     name: "Test Agent 2",
  //     template: "test",
  //     status: "active",
  //     pymdp_config: {},
  //     beliefs: {},
  //     preferences: {},
  //     metrics: {},
  //     parameters: {},
  //     created_at: new Date().toISOString(),
  //     updated_at: new Date().toISOString(),
  //     inference_count: 0,
  //     total_steps: 0,
  //     avatar: "/test-avatar-2.png",
  //   },
  // ];

  // const mockChannels = [
  //   {
  //     id: "channel-1",
  //     name: "Test Channel 1",
  //     type: "group" as const,
  //     participants: ["agent-1", "agent-2"],
  //     unreadCount: 2,
  //     lastMessage: {
  //       id: "msg-1",
  //       agentId: "agent-2",
  //       content: "Hello from agent 2",
  //       timestamp: new Date().toISOString(),
  //       type: "text" as const,
  //     },
  //   },
  //   {
  //     id: "channel-2",
  //     name: "Direct Chat",
  //     type: "direct" as const,
  //     participants: ["agent-1", "agent-2"],
  //     unreadCount: 0,
  //   },
  // ];

  const defaultProps = {
    agent: mockAgent,
    messages: [
      {
        id: "msg-1",
        content: "Hello from agent 2",
        role: "agent" as const,
        timestamp: new Date().toISOString(),
      },
    ],
    onSendMessage: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders without crashing", () => {
    render(<AgentChat {...defaultProps} />);
    expect(screen.getByText("Test Agent 1")).toBeInTheDocument();
  });

  it("displays agent name in header", () => {
    render(<AgentChat {...defaultProps} />);

    expect(screen.getByText("Test Agent 1")).toBeInTheDocument();
  });

  it("displays messages", () => {
    render(<AgentChat {...defaultProps} />);

    expect(screen.getByText("Hello from agent 2")).toBeInTheDocument();
  });

  it("displays message input", () => {
    render(<AgentChat {...defaultProps} />);

    expect(screen.getByPlaceholderText("Type a message...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
  });

  it("shows connection status", () => {
    render(<AgentChat {...defaultProps} />);

    expect(screen.getByText("connected")).toBeInTheDocument();
  });

  it("has message input area", () => {
    render(<AgentChat {...defaultProps} />);

    expect(screen.getByPlaceholderText("Type a message...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  it("switches channels when clicked", async () => {
    const user = userEvent.setup();
    const onSendMessage = jest.fn();

    render(<AgentChat {...defaultProps} onSendMessage={onSendMessage} />);

    const directChatButton = screen.getByText("Direct Chat");
    await user.click(directChatButton);

    // Test channel switching logic here
  });

  it("sends typing indicator when typing", async () => {
    const user = userEvent.setup();
    render(<AgentChat {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type a message...");
    await user.type(textarea, "Hello");

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith({
        type: "agent_typing",
        agent_id: "agent-1",
        channel_id: "channel-1",
        is_typing: true,
      });
    });
  });

  it("sends message when Enter is pressed", async () => {
    const user = userEvent.setup();
    const onSendMessage = jest.fn();

    render(<AgentChat {...defaultProps} onSendMessage={onSendMessage} />);

    const textarea = screen.getByPlaceholderText("Type a message...");
    await user.type(textarea, "Test message");

    // Simulate Enter key press
    fireEvent.keyDown(textarea, { key: "Enter", code: "Enter" });

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "send_agent_message",
          data: expect.objectContaining({
            channel_id: "channel-1",
            message: expect.objectContaining({
              content: "Test message",
              agentId: "agent-1",
              type: "text",
            }),
          }),
        }),
      );
    });
  });

  it("sends message when send button is clicked", async () => {
    const user = userEvent.setup();
    render(<AgentChat {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type a message...");
    const sendButton = screen.getByRole("button", { name: /send/i });

    await user.type(textarea, "Test message");
    await user.click(sendButton);

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "send_agent_message",
        }),
      );
    });
  });

  it("does not send empty messages", async () => {
    const user = userEvent.setup();
    render(<AgentChat {...defaultProps} />);

    const sendButton = screen.getByRole("button", { name: /send/i });
    await user.click(sendButton);

    expect(mockWsClient.send).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "send_agent_message",
      }),
    );
  });

  it("allows shift+enter for new lines without sending", async () => {
    const user = userEvent.setup();
    render(<AgentChat {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type a message...");
    await user.type(textarea, "Line 1");

    // Simulate Shift+Enter
    fireEvent.keyDown(textarea, {
      key: "Enter",
      code: "Enter",
      shiftKey: true,
    });

    expect(mockWsClient.send).not.toHaveBeenCalledWith(
      expect.objectContaining({
        type: "send_agent_message",
      }),
    );
  });

  it("marks channel as read when selected", async () => {
    const user = userEvent.setup();
    render(<AgentChat {...defaultProps} />);

    const directChatButton = screen.getByText("Direct Chat");
    await user.click(directChatButton);

    expect(mockWsClient.send).toHaveBeenCalledWith({
      type: "mark_channel_read",
      agent_id: "agent-1",
      channel_id: "channel-2",
    });
  });

  it("establishes WebSocket connection on mount", async () => {
    render(<AgentChat {...defaultProps} />);

    await waitFor(() => {
      expect(mockWsClient.connect).toHaveBeenCalled();
    });

    expect(mockWsClient.send).toHaveBeenCalledWith({
      type: "subscribe_agent_chat",
      agent_id: "agent-1",
    });
  });

  it("subscribes to WebSocket events", async () => {
    render(<AgentChat {...defaultProps} />);

    await waitFor(() => {
      expect(mockWsClient.subscribe).toHaveBeenCalledWith(
        "agent_chat_message",
        expect.any(Function),
      );
      expect(mockWsClient.subscribe).toHaveBeenCalledWith(
        "agent_presence_update",
        expect.any(Function),
      );
      expect(mockWsClient.subscribe).toHaveBeenCalledWith("agent_typing", expect.any(Function));
    });
  });

  it("stops typing indicator after timeout", async () => {
    jest.useFakeTimers();
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    render(<AgentChat {...defaultProps} />);

    const textarea = screen.getByPlaceholderText("Type a message...");
    await user.type(textarea, "Hello");

    // Fast forward time to trigger timeout
    await act(async () => {
      jest.advanceTimersByTime(3000);
    });

    await waitFor(() => {
      expect(mockWsClient.send).toHaveBeenCalledWith({
        type: "agent_typing",
        agent_id: "agent-1",
        channel_id: "channel-1",
        is_typing: false,
      });
    });

    jest.useRealTimers();
  });
});
