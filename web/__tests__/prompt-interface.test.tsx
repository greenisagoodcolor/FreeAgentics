import React from "react";
import { render, screen, waitFor, act } from "./test-utils";
import userEvent from "@testing-library/user-event";
import { PromptInterface } from "@/components/prompt-interface";
import { apiClient } from "@/lib/api-client";

// Mock API client
jest.mock("@/lib/api-client", () => ({
  apiClient: {
    request: jest.fn(),
    processPrompt: jest.fn(),
    getSuggestions: jest.fn(),
  },
}));

// Mock WebSocket more properly
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = 1;
  send = jest.fn();
  close = jest.fn();
  addEventListener = jest.fn();
  removeEventListener = jest.fn();

  constructor() {
    this.readyState = MockWebSocket.OPEN;
  }
}

global.WebSocket = MockWebSocket as unknown as typeof WebSocket;

describe("PromptInterface", () => {
  const mockApiClient = apiClient;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Accessibility", () => {
    it("should have proper ARIA labels", () => {
      render(<PromptInterface />);

      expect(screen.getByRole("textbox", { name: /enter your prompt/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /send prompt/i })).toBeInTheDocument();
      expect(screen.getByRole("region", { name: /agent visualization/i })).toBeInTheDocument();
      expect(screen.getByRole("region", { name: /knowledge graph/i })).toBeInTheDocument();
    });

    it("should support keyboard navigation", async () => {
      const user = userEvent.setup();
      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });

      // Tab to input
      await user.tab();
      expect(input).toHaveFocus();

      // Type something to enable the button
      await user.type(input, "Test");

      const button = screen.getByRole("button", { name: /send prompt/i });

      // Tab to button
      await user.tab();
      expect(button).toHaveFocus();
    });

    it("should announce loading states to screen readers", async () => {
      // Create a delayed promise to ensure loading state is visible
      let resolvePromise: (value: unknown) => void;
      const delayedPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      mockApiClient.processPrompt = jest.fn().mockReturnValue(delayedPromise);

      render(<PromptInterface />);

      // Initially no loading announcement
      expect(screen.queryByRole("status")).not.toBeInTheDocument();

      // Type and submit
      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test prompt");
      await userEvent.click(button);

      // Loading announcement should appear
      await waitFor(() => {
        expect(screen.getByRole("status")).toHaveTextContent(/processing your prompt/i);
      });

      // Resolve the promise to clean up
      act(() => {
        resolvePromise({
          id: "test-123",
          prompt: "Test prompt",
          status: "processing",
          agents: [],
        });
      });
    });
  });

  describe("User Interactions", () => {
    it("should handle prompt submission", async () => {
      mockApiClient.processPrompt = jest.fn().mockResolvedValue({
        id: "test-123",
        prompt: "Test prompt",
        status: "processing",
        agents: [],
      });

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test prompt");
      await userEvent.click(button);

      await waitFor(() => {
        expect(mockApiClient.processPrompt).toHaveBeenCalledWith({
          prompt: "Test prompt",
          conversationId: undefined,
        });
      });

      // Input should be cleared after submission
      expect(input).toHaveValue("");
    });

    it("should disable submit button when prompt is empty", () => {
      render(<PromptInterface />);

      const button = screen.getByRole("button", { name: /send prompt/i });
      expect(button).toBeDisabled();
    });

    it("should enable submit button when prompt has content", async () => {
      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test");
      expect(button).toBeEnabled();

      await userEvent.clear(input);
      expect(button).toBeDisabled();
    });

    it("should submit on Enter key press", async () => {
      mockApiClient.processPrompt = jest.fn().mockResolvedValue({
        id: "test-123",
        prompt: "Test prompt",
        status: "processing",
        agents: [],
      });

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });

      await userEvent.type(input, "Test prompt");
      await userEvent.keyboard("{Enter}");

      await waitFor(() => {
        expect(mockApiClient.processPrompt).toHaveBeenCalled();
      });
    });

    it("should show suggestion list when typing", async () => {
      mockApiClient.getSuggestions = jest.fn().mockResolvedValue({
        success: true,
        data: ["How to create an agent?", "How does active inference work?"],
      });

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });

      await userEvent.type(input, "How");

      await waitFor(
        () => {
          expect(screen.getByRole("listbox", { name: /suggestions/i })).toBeInTheDocument();
        },
        { timeout: 2000 },
      );
    });

    it("should handle suggestion selection", async () => {
      mockApiClient.getSuggestions = jest.fn().mockResolvedValue({
        success: true,
        data: ["How to create an agent?", "How does active inference work?"],
      });

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });

      await userEvent.type(input, "How");

      await waitFor(
        () => {
          const suggestions = screen.getAllByRole("option");
          expect(suggestions.length).toBeGreaterThan(0);
        },
        { timeout: 2000 },
      );

      const firstSuggestion = screen.getAllByRole("option")[0];
      await userEvent.click(firstSuggestion);

      expect(input).toHaveValue(firstSuggestion.textContent);
    });
  });

  describe("Loading States", () => {
    it("should show loading indicator during processing", async () => {
      let resolvePromise: (value: any) => void;
      const delayedPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      mockApiClient.processPrompt = jest.fn().mockReturnValue(delayedPromise);

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test prompt");

      await act(async () => {
        await userEvent.click(button);
      });

      // Should show loading state immediately after click
      await waitFor(() => {
        expect(button).toHaveTextContent(/processing/i);
        expect(button).toBeDisabled();
      });

      // Clean up by resolving the promise
      act(() => {
        resolvePromise({
          success: true,
          data: {
            id: "test-123",
            prompt: "Test prompt",
            status: "completed",
            agents: [],
          },
        });
      });
    });

    it("should handle loading state transitions", async () => {
      let resolvePromise: (value: any) => void;
      const delayedPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      mockApiClient.processPrompt = jest.fn().mockReturnValue(delayedPromise);

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test prompt");

      await act(async () => {
        await userEvent.click(button);
      });

      // Should be loading
      await waitFor(() => {
        expect(button).toHaveTextContent(/processing/i);
      });

      // Resolve the promise
      act(() => {
        resolvePromise({
          success: true,
          data: {
            id: "test-123",
            prompt: "Test prompt",
            status: "completed",
            agents: [{ id: "agent-1", name: "Test Agent", status: "active" }],
          },
        });
      });

      // Should show completion state (button text changes back)
      await waitFor(() => {
        expect(button).toHaveTextContent(/send/i);
      });

      // Input should be cleared after successful submission
      expect(input).toHaveValue("");

      // Button should be disabled due to empty input (correct behavior)
      expect(button).toBeDisabled();

      // Type new text - button should become enabled again
      await userEvent.type(input, "New prompt");
      await waitFor(() => {
        expect(button).not.toBeDisabled();
      });
    });
  });

  describe("Error Handling", () => {
    it("should display error message on API failure", async () => {
      mockApiClient.processPrompt = jest.fn().mockRejectedValue(new Error("Internal server error"));

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test prompt");

      await act(async () => {
        await userEvent.click(button);
      });

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(/internal server error/i);
      });
    });

    it("should allow retry after error", async () => {
      mockApiClient.processPrompt = jest
        .fn()
        .mockRejectedValueOnce(new Error("Network error"))
        .mockResolvedValueOnce({
          success: true,
          data: {
            id: "test-123",
            prompt: "Test prompt",
            status: "completed",
            agents: [],
          },
        });

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      // First attempt - fails
      await userEvent.type(input, "Test prompt");

      await act(async () => {
        await userEvent.click(button);
      });

      await waitFor(() => {
        expect(screen.getByRole("alert")).toBeInTheDocument();
      });

      // Retry button should appear
      const retryButton = screen.getByRole("button", { name: /retry/i });

      await act(async () => {
        await userEvent.click(retryButton);
      });

      // Should succeed - error should be cleared
      await waitFor(
        () => {
          expect(screen.queryByRole("alert")).not.toBeInTheDocument();
        },
        { timeout: 3000 },
      );
    });

    it("should handle network timeout gracefully", async () => {
      mockApiClient.processPrompt = jest.fn().mockRejectedValue({
        detail: "Request timeout",
        status: 408,
      });

      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });
      const button = screen.getByRole("button", { name: /send prompt/i });

      await userEvent.type(input, "Test prompt");
      await userEvent.click(button);

      await waitFor(() => {
        const alert = screen.getByRole("alert");
        expect(alert).toBeInTheDocument();
        // Check for various possible error messages
        expect(alert.textContent).toMatch(/timeout|failed to process prompt/i);
      });
    });
  });

  describe("Real-time Updates", () => {
    // WebSocket functionality has been removed from the component
    it.skip("should establish WebSocket connection for real-time updates", async () => {
      // This test is skipped as WebSocket functionality has been moved
      // to a separate module and is no longer part of PromptInterface
    });

    it.skip("should update agent visualization on WebSocket message", async () => {
      // This test is skipped as WebSocket functionality has been moved
      // to a separate module and is no longer part of PromptInterface
    });

    it.skip("should handle WebSocket reconnection", async () => {
      // This test is skipped as WebSocket functionality has been moved
      // to a separate module and is no longer part of PromptInterface
    });
  });

  describe("Performance", () => {
    it("should render within performance budget", async () => {
      const startTime = performance.now();
      render(<PromptInterface />);
      const renderTime = performance.now() - startTime;

      expect(renderTime).toBeLessThan(100); // 100ms budget for initial render
    });

    it("should debounce suggestion requests", async () => {
      jest.useFakeTimers();
      mockApiClient.getSuggestions = jest.fn().mockResolvedValue({
        suggestions: ["How to test?"],
      });

      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      render(<PromptInterface />);

      const input = screen.getByRole("textbox", { name: /enter your prompt/i });

      // Type quickly
      await user.type(input, "How");

      // Should not make requests immediately
      expect(mockApiClient.getSuggestions).not.toHaveBeenCalled();

      // Fast-forward debounce timer
      await act(async () => {
        jest.advanceTimersByTime(300);
      });

      // Now should make request
      await waitFor(() => {
        expect(mockApiClient.getSuggestions).toHaveBeenCalledWith("How");
      });

      jest.useRealTimers();
    });
  });

  describe("Mobile Responsiveness", () => {
    it("should adapt layout for mobile screens", () => {
      // Mock mobile viewport
      global.innerWidth = 375;
      global.innerHeight = 667;

      render(<PromptInterface />);

      const container = screen.getByTestId("prompt-interface-container");
      expect(container).toHaveClass("flex-col"); // Should stack vertically on mobile
    });

    it.skip("should handle touch interactions", async () => {
      // This test is skipped due to jsdom limitations with touch events
      // Touch interactions are tested in e2e tests
    });
  });

  describe("Integration", () => {
    it.skip("should update knowledge graph on successful prompt processing", async () => {
      // This test is skipped as it requires complex canvas rendering setup
      // Knowledge graph functionality is tested in integration tests
    });
  });
});
