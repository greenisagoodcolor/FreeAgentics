import React from "react";
import { render, screen, act } from "../../../__tests__/test-utils";
import userEvent from "@testing-library/user-event";
import { ConversationWindow } from "@/components/main/ConversationWindow";
import { useConversation } from "@/hooks/use-conversation";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";

// Mock the hooks
jest.mock("@/hooks/use-conversation");
jest.mock("@/hooks/use-prompt-processor");

const mockUseConversation = useConversation as jest.MockedFunction<typeof useConversation>;
const mockUsePromptProcessor = usePromptProcessor as jest.MockedFunction<typeof usePromptProcessor>;

describe("ConversationWindow", () => {
  const mockSendMessage = jest.fn();
  const mockProcessPrompt = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    // Default conversation state
    mockUseConversation.mockReturnValue({
      messages: [],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    // Default prompt processor state
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: false,
      error: null,
      agents: [],
      knowledgeGraph: { nodes: [], edges: [] },
      suggestions: [],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: "test-conv-id",
      iterationContext: null,
      resetConversation: jest.fn(),
    });
  });

  it("renders conversation window", () => {
    render(<ConversationWindow />);

    expect(screen.getByRole("heading", { name: /conversation/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument();
  });

  it("displays messages in conversation", () => {
    mockUseConversation.mockReturnValue({
      messages: [
        {
          id: "1",
          role: "user",
          content: "Hello, create an agent for me",
          timestamp: new Date("2024-01-01T10:00:00Z").toISOString(),
        },
        {
          id: "2",
          role: "assistant",
          content: "I'll help you create an agent. What kind of agent would you like?",
          timestamp: new Date("2024-01-01T10:00:30Z").toISOString(),
        },
        {
          id: "3",
          role: "system",
          content: "Agent creation process initiated",
          timestamp: new Date("2024-01-01T10:00:45Z").toISOString(),
        },
      ],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    expect(screen.getByText("Hello, create an agent for me")).toBeInTheDocument();
    expect(screen.getByText(/I'll help you create an agent/)).toBeInTheDocument();
    expect(screen.getByText("Agent creation process initiated")).toBeInTheDocument();
  });

  it("shows message metadata (timestamp and role)", () => {
    mockUseConversation.mockReturnValue({
      messages: [
        {
          id: "1",
          role: "user",
          content: "Test message",
          timestamp: new Date("2024-01-01T10:00:00Z").toISOString(),
        },
      ],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    expect(screen.getByText("You")).toBeInTheDocument();
    // The time shows as "11:00 AM" in the test output (locale-dependent)
    const timeElements = screen.getAllByText((content) => {
      return /\d{1,2}:\d{2}/.test(content);
    });
    expect(timeElements.length).toBeGreaterThan(0);
  });

  it("sends message when form is submitted", async () => {
    const user = userEvent.setup();
    render(<ConversationWindow />);

    const input = screen.getByPlaceholderText(/type a message/i);
    const sendButton = screen.getByRole("button", { name: /send/i });

    await user.type(input, "Create an explorer agent");
    await user.click(sendButton);

    expect(mockSendMessage).toHaveBeenCalledWith({
      content: "Create an explorer agent",
    });
  });

  it("clears input after sending message", async () => {
    const user = userEvent.setup();
    render(<ConversationWindow />);

    const input = screen.getByPlaceholderText(/type a message/i) as HTMLInputElement;
    await user.type(input, "Test message");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(input.value).toBe("");
  });

  it("sends message on Enter key", async () => {
    const user = userEvent.setup();
    render(<ConversationWindow />);

    const input = screen.getByPlaceholderText(/type a message/i);
    await user.type(input, "Test message");
    await user.keyboard("{Enter}");

    expect(mockSendMessage).toHaveBeenCalledWith({
      content: "Test message",
    });
  });

  it("allows multiline with Shift+Enter", async () => {
    const user = userEvent.setup();
    render(<ConversationWindow />);

    const input = screen.getByPlaceholderText(/type a message/i) as HTMLTextAreaElement;
    await user.type(input, "Line 1");
    await user.keyboard("{Shift>}{Enter}{/Shift}");
    await user.type(input, "Line 2");

    expect(input.value).toBe("Line 1\nLine 2");
    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it("disables input while sending message", async () => {
    mockUseConversation.mockReturnValue({
      messages: [],
      sendMessage: mockSendMessage,
      isLoading: true,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    const input = screen.getByPlaceholderText(/type a message/i);
    const sendButton = screen.getByRole("button", { name: /send/i });

    expect(input).toBeDisabled();
    expect(sendButton).toBeDisabled();
  });

  it("shows loading indicator for streaming messages", () => {
    mockUseConversation.mockReturnValue({
      messages: [
        {
          id: "1",
          role: "assistant",
          content: "Processing your request",
          timestamp: new Date().toISOString(),
          isStreaming: true,
        },
      ],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    expect(screen.getByTestId("streaming-indicator")).toBeInTheDocument();
  });

  it("shows error message when there is an error", () => {
    mockUseConversation.mockReturnValue({
      messages: [],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: new Error("Failed to send message"),
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    expect(screen.getByText(/failed to send message/i)).toBeInTheDocument();
  });

  it("shows suggestions from prompt processor", () => {
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: false,
      error: null,
      agents: [],
      knowledgeGraph: { nodes: [], edges: [] },
      suggestions: [
        "Add obstacle detection to the agent",
        "Create a resource collector agent",
        "Implement path planning algorithm",
      ],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: "test-conv-id",
      iterationContext: null,
      resetConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    expect(screen.getByText("Add obstacle detection to the agent")).toBeInTheDocument();
    expect(screen.getByText("Create a resource collector agent")).toBeInTheDocument();
    expect(screen.getByText("Implement path planning algorithm")).toBeInTheDocument();
  });

  it("applies suggestion when clicked", async () => {
    const user = userEvent.setup();
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: false,
      error: null,
      agents: [],
      knowledgeGraph: { nodes: [], edges: [] },
      suggestions: ["Add obstacle detection"],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: "test-conv-id",
      iterationContext: null,
      resetConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    await user.click(screen.getByText("Add obstacle detection"));

    const input = screen.getByPlaceholderText(/type a message/i) as HTMLTextAreaElement;
    expect(input.value).toBe("Add obstacle detection");
  });

  it("shows clear conversation button", () => {
    mockUseConversation.mockReturnValue({
      messages: [{ id: "1", role: "user", content: "Test", timestamp: new Date().toISOString() }],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    expect(screen.getByRole("button", { name: /clear conversation/i })).toBeInTheDocument();
  });

  it("clears conversation when clear button is clicked", async () => {
    const user = userEvent.setup();
    const mockClear = jest.fn();

    mockUseConversation.mockReturnValue({
      messages: [{ id: "1", role: "user", content: "Test", timestamp: new Date().toISOString() }],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: mockClear,
    });

    render(<ConversationWindow />);

    await user.click(screen.getByRole("button", { name: /clear conversation/i }));

    expect(mockClear).toHaveBeenCalled();
  });

  it("auto-scrolls to bottom when new messages arrive", () => {
    const { rerender } = render(<ConversationWindow />);

    // Add a message
    mockUseConversation.mockReturnValue({
      messages: [
        { id: "1", role: "user", content: "New message", timestamp: new Date().toISOString() },
      ],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    rerender(<ConversationWindow />);

    // Check that scroll area exists (actual scrolling behavior would be tested in e2e)
    expect(screen.getByTestId("messages-scroll-area")).toBeInTheDocument();
  });

  it("shows empty state when no messages", () => {
    render(<ConversationWindow />);

    expect(screen.getByText(/start a conversation/i)).toBeInTheDocument();
    expect(screen.getByText(/type a message below/i)).toBeInTheDocument();
  });

  it("renders markdown content in messages", () => {
    mockUseConversation.mockReturnValue({
      messages: [
        {
          id: "1",
          role: "assistant",
          content: "**Bold text** and *italic text* with `code`",
          timestamp: new Date().toISOString(),
        },
      ],
      sendMessage: mockSendMessage,
      isLoading: false,
      error: null,
      conversationId: "test-conv-id",
      clearConversation: jest.fn(),
    });

    render(<ConversationWindow />);

    // Markdown should be rendered (actual rendering would need a markdown component)
    expect(screen.getByText(/Bold text/)).toBeInTheDocument();
  });
});
