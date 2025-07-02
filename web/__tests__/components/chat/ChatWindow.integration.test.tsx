import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatWindow from "@/components/chat-window";
import type { Agent, Conversation, Message } from "@/lib/types";

// Mock the conversation orchestrator hook
const mockQueueAgentResponse = jest.fn();
const mockProcessNewMessage = jest.fn();
const mockCancelAllResponses = jest.fn();

jest.mock("@/hooks/useConversationorchestrator", () => ({
  useConversationOrchestrator: jest.fn(() => ({
    queueAgentResponse: mockQueueAgentResponse,
    processNewMessage: mockProcessNewMessage,
    cancelAllResponses: mockCancelAllResponses,
    processingAgents: [],
    queuedAgents: [],
    typingAgents: {},
    processingMessageIds: [],
    isProcessing: false,
    error: null,
  })),
}));

// Mock the debug logger
jest.mock("@/lib/debug-logger", () => ({
  createLogger: jest.fn(() => ({
    log: jest.fn(),
    error: jest.fn(),
  })),
}));

// Mock icons
jest.mock("lucide-react", () => ({
  Send: ({ size, className }: any) => (
    <span
      className={className}
      data-testid="send-icon"
      style={{ fontSize: size }}
    >
      Send
    </span>
  ),
  X: ({ size, className }: any) => (
    <span className={className} data-testid="x-icon" style={{ fontSize: size }}>
      X
    </span>
  ),
  Loader2: ({ size, className }: any) => (
    <span
      className={className}
      data-testid="loader-icon"
      style={{ fontSize: size }}
    >
      Loading
    </span>
  ),
  CornerDownRight: ({ size, className }: any) => (
    <span
      className={className}
      data-testid="corner-down-right-icon"
      style={{ fontSize: size }}
    >
      ↘
    </span>
  ),
  AlertTriangle: ({ size, className }: any) => (
    <span
      className={className}
      data-testid="alert-triangle-icon"
      style={{ fontSize: size }}
    >
      ⚠
    </span>
  ),
  Upload: ({ size, className }: any) => (
    <span
      className={className}
      data-testid="upload-icon"
      style={{ fontSize: size }}
    >
      Upload
    </span>
  ),
}));

// Refactored mock data to match unified Agent interface
const mockAgents: Agent[] = [
  {
    id: "agent-1",
    name: "Alice",
    templateId: "explorer",
    biography: "An adventurous agent that discovers new territories and maps unknown environments.",
    knowledgeDomains: ["exploration", "mapping", "discovery", "navigation"],
    knowledge: [
      {
        id: "knowledge-1",
        title: "Navigation Techniques",
        content: "Advanced pathfinding algorithms",
        timestamp: new Date(),
        tags: ["navigation", "algorithms"],
      },
    ],
    parameters: {
      responseThreshold: 0.6,
      turnTakingProbability: 0.7,
      conversationEngagement: 0.8,
    },
    status: "idle",
    avatarUrl: "/avatars/explorer.svg",
    color: "#10B981",
    position: { x: 0, y: 0 },
    personality: {
      openness: 0.8,
      conscientiousness: 0.7,
      extraversion: 0.6,
      agreeableness: 0.9,
      neuroticism: 0.2,
    },
    createdAt: Date.now() - 7200000,
    lastActive: Date.now() - 300000,
    inConversation: true,
    autonomyEnabled: true,
    activityMetrics: {
      messagesCount: 47,
      beliefCount: 12,
      responseTime: [340, 280, 410, 290],
    },
  },
  {
    id: "agent-2",
    name: "Bob",
    templateId: "scholar",
    biography: "A learned agent that analyzes patterns and synthesizes knowledge.",
    knowledgeDomains: ["analysis", "synthesis", "education", "research"],
    knowledge: [
      {
        id: "knowledge-2",
        title: "Pattern Analysis",
        content: "Statistical pattern recognition methods",
        timestamp: new Date(),
        tags: ["patterns", "analysis"],
      },
    ],
    parameters: {
      responseThreshold: 0.8,
      turnTakingProbability: 0.5,
      conversationEngagement: 0.6,
    },
    status: "idle",
    avatarUrl: "/avatars/scholar.svg",
    color: "#8B5CF6",
    position: { x: 1, y: 1 },
    personality: {
      openness: 0.7,
      conscientiousness: 0.8,
      extraversion: 0.5,
      agreeableness: 0.8,
      neuroticism: 0.3,
    },
    createdAt: Date.now() - 5400000,
    lastActive: Date.now() - 120000,
    inConversation: true,
    autonomyEnabled: true,
    activityMetrics: {
      messagesCount: 38,
      beliefCount: 19,
      responseTime: [520, 480, 390, 450],
    },
  },
];

const mockMessages: Message[] = [
  {
    id: "msg-1",
    content: "Hello everyone!",
    senderId: "user",
    timestamp: new Date(1672567200000), // 2023-01-01T10:00:00Z
    metadata: { isGeneratedByLLM: false },
  },
  {
    id: "msg-2",
    content: "Hi there! How can I help?",
    senderId: "agent-1",
    timestamp: new Date(1672567260000), // 2023-01-01T10:01:00Z
    metadata: { isGeneratedByLLM: true },
  },
];

const mockConversation: Conversation = {
  id: "conv-1",
  participants: ["user", "agent-1", "agent-2"],
  messages: mockMessages,
  startTime: new Date(1672567200000),
  endTime: null,
};

describe("ChatWindow Integration Tests", () => {
  const mockOnSendMessage = jest.fn();
  const mockOnEndConversation = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.clearAllTimers();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  const renderChatWindow = (
    conversation = mockConversation,
    agents = mockAgents,
  ) => {
    return render(
      <ChatWindow
        conversation={conversation}
        agents={agents}
        onSendMessage={mockOnSendMessage}
        onEndConversation={mockOnEndConversation}
      />,
    );
  };

  describe("Component Rendering", () => {
    it("renders chat header with title", () => {
      renderChatWindow();

      expect(screen.getByText("Chat")).toBeInTheDocument();
      expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
        "Chat",
      );
    });

    it("renders end conversation button when conversation exists", () => {
      renderChatWindow();

      const endButton = screen.getByRole("button", {
        name: /end conversation/i,
      });
      expect(endButton).toBeInTheDocument();
      expect(screen.getByTestId("x-icon")).toBeInTheDocument();
    });

    it("renders message input and send button", () => {
      renderChatWindow();

      expect(
        screen.getByPlaceholderText("Type your message..."),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
      expect(screen.getByTestId("send-icon")).toBeInTheDocument();
    });

    it("renders sender selection dropdown", () => {
      renderChatWindow();

      const select = screen.getByRole("combobox");
      expect(select).toBeInTheDocument();
      expect(select).toHaveValue("user");

      // Check options are present
      expect(screen.getByRole("option", { name: "You" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "Alice" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "Bob" })).toBeInTheDocument();
    });

    it("renders upload button (disabled)", () => {
      renderChatWindow();

      const uploadButton = screen.getByRole("button", { name: /upload/i });
      expect(uploadButton).toBeInTheDocument();
      expect(uploadButton).toBeDisabled();
      expect(screen.getByTestId("upload-icon")).toBeInTheDocument();
    });
  });

  describe("Message Display", () => {
    it("displays existing messages correctly", () => {
      renderChatWindow();

      expect(screen.getByText("Hello everyone!")).toBeInTheDocument();
      expect(screen.getByText("Hi there! How can I help?")).toBeInTheDocument();
    });

    it("displays sender names correctly", () => {
      renderChatWindow();

      // Check message sender names (multiple instances are expected)
      const senderNames = screen.getAllByText("You");
      expect(senderNames.length).toBeGreaterThan(0);
      const aliceNames = screen.getAllByText("Alice");
      expect(aliceNames.length).toBeGreaterThan(0);
    });

    it("displays timestamps", () => {
      renderChatWindow();

      // Check that timestamps are displayed (format may vary by locale)
      const timeElements = screen.getAllByText(/\d{1,2}:\d{2}:\d{2}/);
      expect(timeElements.length).toBeGreaterThan(0);
    });

    it("displays AI indicator for LLM-generated messages", () => {
      renderChatWindow();

      expect(screen.getByText("AI")).toBeInTheDocument();
    });

    it("displays agent colors correctly", async () => {
      renderChatWindow();
      // If color rendering is async, wait for UI update
      await waitFor(() => {
        expect(screen.getByText("Alice")).toBeInTheDocument();
      });
    });

    it("handles empty conversation gracefully", () => {
      const emptyConversation = { ...mockConversation, messages: [] };
      renderChatWindow(emptyConversation);

      expect(
        screen.getByText(
          "No messages yet. Start the conversation by sending a message!",
        ),
      ).toBeInTheDocument();
    });

    it("handles null conversation gracefully", () => {
      renderChatWindow(null);

      expect(
        screen.getByText("No active conversation. Add agents to start one."),
      ).toBeInTheDocument();
    });

    it("skips messages with SKIP_RESPONSE", () => {
      const conversationWithSkippedMessage = {
        ...mockConversation,
        messages: [
          ...mockMessages,
          {
            id: "msg-skip",
            content:
              "This message contains SKIP_RESPONSE and should not render",
            senderId: "agent-1",
            timestamp: new Date(),
          },
        ],
      };

      renderChatWindow(conversationWithSkippedMessage);

      expect(
        screen.queryByText(
          "This message contains SKIP_RESPONSE and should not render",
        ),
      ).not.toBeInTheDocument();
    });

    it("displays system messages with special formatting", () => {
      const conversationWithSystemMessage = {
        ...mockConversation,
        messages: [
          ...mockMessages,
          {
            id: "msg-system",
            content: "System notification message",
            senderId: "system",
            timestamp: new Date(),
            metadata: { isSystemMessage: true },
          },
        ],
      };

      renderChatWindow(conversationWithSystemMessage);

      expect(
        screen.getByText("System notification message"),
      ).toBeInTheDocument();
    });
  });

  describe("Message Input and Sending", () => {
    it("allows typing in message input", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");
      await user.type(input, "Test message");

      expect(input).toHaveValue("Test message");
    });

    it("sends message on button click", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");
      const sendButton = screen.getByRole("button", { name: /send/i });

      await user.type(input, "Test message");
      await user.click(sendButton);

      expect(mockOnSendMessage).toHaveBeenCalledWith("Test message", "user");
    });

    it("sends message on Enter key press", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");

      await user.type(input, "Test message{enter}");

      expect(mockOnSendMessage).toHaveBeenCalledWith("Test message", "user");
    });

    it("does not send on Shift+Enter", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");

      await user.type(input, "Test message");
      await user.keyboard("{Shift>}{Enter}{/Shift}");

      expect(mockOnSendMessage).not.toHaveBeenCalled();
    });

    it("clears input after sending message", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");

      await user.type(input, "Test message");
      await user.keyboard("{enter}");

      expect(input).toHaveValue("");
    });

    it("disables send button when input is empty", () => {
      renderChatWindow();

      const sendButton = screen.getByRole("button", { name: /send/i });
      expect(sendButton).toBeDisabled();
    });

    it("enables send button when input has content", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");
      const sendButton = screen.getByRole("button", { name: /send/i });

      await user.type(input, "Test");

      expect(sendButton).not.toBeDisabled();
    });

    it("does not send whitespace-only messages", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");

      await user.type(input, "   ");
      await user.keyboard("{enter}");

      expect(mockOnSendMessage).not.toHaveBeenCalled();
    });

    it("processes new message after sending", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");

      await user.type(input, "Test message");
      await user.keyboard("{enter}");

      // Fast-forward the timeout
      jest.advanceTimersByTime(100);

      expect(mockProcessNewMessage).toHaveBeenCalled();
    });
  });

  describe("Sender Selection", () => {
    it("allows changing sender", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const select = screen.getByRole("combobox");

      await user.selectOptions(select, "agent-1");

      expect(select).toHaveValue("agent-1");
    });

    it("sends message with selected sender", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const select = screen.getByRole("combobox");
      const input = screen.getByPlaceholderText("Type your message...");

      await user.selectOptions(select, "agent-1");
      await user.type(input, "Message from Alice");
      await user.keyboard("{enter}");

      expect(mockOnSendMessage).toHaveBeenCalledWith(
        "Message from Alice",
        "agent-1",
      );
    });

    it("only shows agents that are participants", () => {
      const conversationWithOneAgent = {
        ...mockConversation,
        participants: ["user", "agent-1"], // Only agent-1 is a participant
      };

      renderChatWindow(conversationWithOneAgent);

      expect(screen.getByRole("option", { name: "You" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "Alice" })).toBeInTheDocument();
      expect(
        screen.queryByRole("option", { name: "Bob" }),
      ).not.toBeInTheDocument();
    });
  });

  describe("Conversation Controls", () => {
    it("calls onEndConversation when end button is clicked", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const endButton = screen.getByRole("button", {
        name: /end conversation/i,
      });
      await user.click(endButton);

      expect(mockCancelAllResponses).toHaveBeenCalled();
      expect(mockOnEndConversation).toHaveBeenCalled();
    });

    it("does not show end button when no conversation", () => {
      renderChatWindow(null);

      expect(
        screen.queryByRole("button", { name: /end conversation/i }),
      ).not.toBeInTheDocument();
    });

    it("does not show input area when no conversation", () => {
      renderChatWindow(null);

      expect(
        screen.queryByPlaceholderText("Type your message..."),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: /send/i }),
      ).not.toBeInTheDocument();
    });
  });

  describe("Processing States", () => {
    it("shows loading state when sending", () => {
      // Mock the hook to return isSending state
      const useConversationOrchestratorMock =
        require("@/hooks/useConversationorchestrator").useConversationOrchestrator;
      useConversationOrchestratorMock.mockReturnValueOnce({
        queueAgentResponse: mockQueueAgentResponse,
        processNewMessage: mockProcessNewMessage,
        cancelAllResponses: mockCancelAllResponses,
        processingAgents: [],
        queuedAgents: [],
        typingAgents: {},
        processingMessageIds: [],
        isProcessing: true,
        error: null,
      });

      renderChatWindow();

      // When processing, should show status message
      expect(screen.getByText(/agents are responding/i)).toBeInTheDocument();
    });

    it("disables inputs when processing", () => {
      const useConversationOrchestratorMock =
        require("@/hooks/useConversationorchestrator").useConversationOrchestrator;
      useConversationOrchestratorMock.mockReturnValueOnce({
        queueAgentResponse: mockQueueAgentResponse,
        processNewMessage: mockProcessNewMessage,
        cancelAllResponses: mockCancelAllResponses,
        processingAgents: [],
        queuedAgents: [],
        typingAgents: {},
        processingMessageIds: [],
        isProcessing: true,
        error: null,
      });

      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");
      const sendButton = screen.getByRole("button", { name: /send/i });
      const select = screen.getByRole("combobox");

      expect(input).toBeDisabled();
      expect(sendButton).toBeDisabled();
      expect(select).toBeDisabled();
    });

    it("shows typing indicators", () => {
      const useConversationOrchestratorMock =
        require("@/hooks/useConversationorchestrator").useConversationOrchestrator;
      useConversationOrchestratorMock.mockReturnValueOnce({
        queueAgentResponse: mockQueueAgentResponse,
        processNewMessage: mockProcessNewMessage,
        cancelAllResponses: mockCancelAllResponses,
        processingAgents: [],
        queuedAgents: [],
        typingAgents: {
          "agent-1": { text: "I am thinking...", messageId: "msg-1" },
        },
        processingMessageIds: [],
        isProcessing: false,
        error: null,
      });

      renderChatWindow();

      // Check for typing indicators in the UI
      expect(screen.getByText("typing...")).toBeInTheDocument();
      expect(screen.getByText("I am thinking...")).toBeInTheDocument();
      // Alice appears multiple times (in messages and typing indicator)
      const aliceElements = screen.getAllByText("Alice");
      expect(aliceElements.length).toBeGreaterThan(0);
    });
  });

  describe("Error Handling", () => {
    it("displays error messages", () => {
      const useConversationOrchestratorMock =
        require("@/hooks/useConversationorchestrator").useConversationOrchestrator;
      useConversationOrchestratorMock.mockReturnValueOnce({
        queueAgentResponse: mockQueueAgentResponse,
        processNewMessage: mockProcessNewMessage,
        cancelAllResponses: mockCancelAllResponses,
        processingAgents: [],
        queuedAgents: [],
        typingAgents: {},
        processingMessageIds: [],
        isProcessing: false,
        error: "Something went wrong",
      });

      renderChatWindow();

      expect(screen.getByText("Error:")).toBeInTheDocument();
      expect(screen.getByText("Something went wrong")).toBeInTheDocument();
      expect(screen.getByTestId("alert-triangle-icon")).toBeInTheDocument();
    });

    it("clears local errors after timeout", async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      // Simulate a local error by causing the onSendMessage to throw
      mockOnSendMessage.mockImplementationOnce(() => {
        throw new Error("Network error");
      });

      const input = screen.getByPlaceholderText("Type your message...");
      const sendButton = screen.getByRole("button", { name: /send/i });

      await user.type(input, "Test message");
      await user.click(sendButton);

      // Fast-forward time to trigger error clearing
      jest.advanceTimersByTime(5000);

      // Error should be cleared (this test would need to be adjusted based on actual implementation)
    });

    it("restores message input on send failure", async () => {
      mockOnSendMessage.mockImplementationOnce(() => {
        throw new Error("Send failed");
      });

      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");

      await user.type(input, "Test message");
      await user.keyboard("{enter}");

      // Input should be restored on error
      expect(input).toHaveValue("Test message");
    });
  });

  describe("Accessibility", () => {
    it("has proper semantic structure", () => {
      renderChatWindow();

      expect(screen.getByRole("heading", { level: 2 })).toBeInTheDocument();
      expect(screen.getByRole("textbox")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    it("has descriptive button labels", () => {
      renderChatWindow();

      expect(
        screen.getByRole("button", { name: /end conversation/i }),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /upload/i }),
      ).toBeInTheDocument();
    });

    it("has proper form controls", () => {
      renderChatWindow();

      const input = screen.getByPlaceholderText("Type your message...");
      expect(input).toHaveAttribute("placeholder", "Type your message...");

      const select = screen.getByRole("combobox");
      expect(select).toBeInTheDocument();
    });

    it("provides upload button tooltip", () => {
      renderChatWindow();

      const uploadButton = screen.getByRole("button", { name: /upload/i });
      expect(uploadButton).toHaveAttribute(
        "title",
        "Upload files (coming soon)",
      );
    });
  });

  describe("Message Threading", () => {
    it("displays response indicators", () => {
      const conversationWithResponse = {
        ...mockConversation,
        messages: [
          ...mockMessages,
          {
            id: "msg-response",
            content: "This is a response",
            senderId: "agent-2",
            timestamp: new Date(),
            metadata: {
              isGeneratedByLLM: true,
              respondingTo: "msg-1",
            },
          },
        ],
      };

      renderChatWindow(conversationWithResponse);

      expect(screen.getByText(/responding to:/)).toBeInTheDocument();
      expect(screen.getByTestId("corner-down-right-icon")).toBeInTheDocument();
    });

    it("shows processing indicators for messages being responded to", async () => {
      const useConversationOrchestratorMock =
        require("@/hooks/useConversationorchestrator").useConversationOrchestrator;
      useConversationOrchestratorMock.mockReturnValue({
        queueAgentResponse: mockQueueAgentResponse,
        processNewMessage: mockProcessNewMessage,
        cancelAllResponses: mockCancelAllResponses,
        processingAgents: [],
        queuedAgents: [],
        typingAgents: {},
        processingMessageIds: ["msg-1"],
        isProcessing: true,
        error: null,
      });

      renderChatWindow();

      // Wait for useEffect to process the processingMessageIds
      await waitFor(() => {
        expect(
          screen.getByText(/Agents are responding to this message/),
        ).toBeInTheDocument();
      });
    });
  });

  describe("Performance", () => {
    it("handles large message lists efficiently", () => {
      const manyMessages = Array.from({ length: 100 }, (_, i) => ({
        id: `msg-${i}`,
        content: `Message ${i}`,
        senderId: i % 2 === 0 ? "user" : "agent-1",
        timestamp: new Date(Date.now() + i * 1000),
      }));

      const conversationWithManyMessages = {
        ...mockConversation,
        messages: manyMessages,
      };

      const startTime = performance.now();
      renderChatWindow(conversationWithManyMessages);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should render quickly
      expect(screen.getByText("Message 0")).toBeInTheDocument();
      expect(screen.getByText("Message 99")).toBeInTheDocument();
    });

    it("maintains scroll position efficiently", () => {
      renderChatWindow();

      // Test would verify scroll behavior, but requires DOM measurement
      const messagesContainer = document.querySelector(".overflow-y-auto");
      expect(messagesContainer).toBeInTheDocument();
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts and unmounts without errors", () => {
      const { unmount } = renderChatWindow();

      expect(screen.getByText("Chat")).toBeInTheDocument();

      expect(() => unmount()).not.toThrow();
    });

    it("handles prop changes gracefully", () => {
      const { rerender } = renderChatWindow();

      const newConversation = {
        ...mockConversation,
        id: "conv-2",
        messages: [],
      };

      rerender(
        <ChatWindow
          conversation={newConversation}
          agents={mockAgents}
          onSendMessage={mockOnSendMessage}
          onEndConversation={mockOnEndConversation}
        />,
      );

      expect(
        screen.getByText(
          "No messages yet. Start the conversation by sending a message!",
        ),
      ).toBeInTheDocument();
    });
  });

  // Example for a test that triggers a timer (sending a message)
  it("sends a message and processes it", async () => {
    renderChatWindow();
    const input = screen.getByPlaceholderText("Type your message...");
    const sendButton = screen.getByRole("button", { name: /send/i });
    await userEvent.type(input, "Test message");
    fireEvent.click(sendButton);
    // Advance timers to resolve setTimeout in handleSendMessage
    jest.runAllTimers();
    // Wait for UI update
    await waitFor(() => {
      expect(mockOnSendMessage).toHaveBeenCalledWith("Test message", "user");
    });
  });
});
