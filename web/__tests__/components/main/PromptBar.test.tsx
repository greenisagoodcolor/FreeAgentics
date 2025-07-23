import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PromptBar } from "@/components/main/PromptBar";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";

// Mock the hook
jest.mock("@/hooks/use-prompt-processor");

const mockProcessPrompt = jest.fn();
const mockUsePromptProcessor = usePromptProcessor as jest.MockedFunction<typeof usePromptProcessor>;

describe("PromptBar", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: false,
      error: null,
      agents: [],
      knowledgeGraph: null,
      suggestions: [],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: null,
      iterationContext: null,
      resetConversation: jest.fn(),
    });
  });

  it("renders prompt input area", () => {
    render(<PromptBar />);

    const input = screen.getByPlaceholderText(/enter your goal/i);
    expect(input).toBeInTheDocument();
    expect(input.tagName).toBe("TEXTAREA");
  });

  it("submits user goal on Enter", async () => {
    const user = userEvent.setup();
    render(<PromptBar />);

    const input = screen.getByPlaceholderText(/enter your goal/i);
    await user.type(input, "Find resources in the grid");
    await user.keyboard("{Enter}");

    expect(mockProcessPrompt).toHaveBeenCalledWith("Find resources in the grid");
  });

  it("submits user goal on Ctrl+Enter", async () => {
    const user = userEvent.setup();
    render(<PromptBar />);

    const input = screen.getByPlaceholderText(/enter your goal/i);
    await user.type(input, "Explore the environment");
    await user.keyboard("{Control>}{Enter}");

    expect(mockProcessPrompt).toHaveBeenCalledWith("Explore the environment");
  });

  it("shows processing state when submitting", () => {
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: true,
      error: null,
      agents: [],
      knowledgeGraph: null,
      suggestions: [],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: null,
      iterationContext: null,
      resetConversation: jest.fn(),
    });

    render(<PromptBar />);

    const input = screen.getByPlaceholderText(/enter your goal/i);
    expect(input).toBeDisabled();
    // Look for a loading indicator or processing text
  });

  it("shows error message when there is an error", () => {
    const errorMessage = "Failed to process prompt";
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: false,
      error: errorMessage,
      agents: [],
      knowledgeGraph: null,
      suggestions: [],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: null,
      iterationContext: null,
      resetConversation: jest.fn(),
    });

    render(<PromptBar />);

    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it("shows settings button", () => {
    render(<PromptBar />);

    const settingsButton = screen.getByLabelText(/settings/i);
    expect(settingsButton).toBeInTheDocument();
  });

  it("opens settings drawer when settings button is clicked", async () => {
    const user = userEvent.setup();
    render(<PromptBar />);

    const settingsButton = screen.getByLabelText(/settings/i);
    await user.click(settingsButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
  });

  it("shows conversation history snippets", () => {
    mockUsePromptProcessor.mockReturnValue({
      submitPrompt: mockProcessPrompt,
      isLoading: false,
      error: null,
      agents: [],
      knowledgeGraph: null,
      suggestions: [],
      retry: jest.fn(),
      fetchSuggestions: jest.fn(),
      conversationId: "test-conversation",
      iterationContext: {
        iteration_number: 2,
        total_agents: 1,
        kg_nodes: 5,
        conversation_summary: {
          iteration_count: 2,
          belief_evolution: { trend: "stable", stability: 0.8 },
        },
      },
      resetConversation: jest.fn(),
    });

    render(<PromptBar />);

    expect(screen.getByText(/create an explorer agent/i)).toBeInTheDocument();
    expect(screen.getByText(/add obstacle detection/i)).toBeInTheDocument();
  });

  it("clears input after successful submission", async () => {
    const user = userEvent.setup();
    render(<PromptBar />);

    const input = screen.getByPlaceholderText(/enter your goal/i) as HTMLTextAreaElement;
    await user.type(input, "Test prompt");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(input.value).toBe("");
    });
  });

  it("expands textarea on focus", async () => {
    const user = userEvent.setup();
    render(<PromptBar />);

    const input = screen.getByPlaceholderText(/enter your goal/i);
    expect(input).toHaveClass("h-10");

    await user.click(input);
    expect(input).toHaveClass("h-20");
  });
});
