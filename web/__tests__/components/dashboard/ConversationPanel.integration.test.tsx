import React from "react";
import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Ensure we're using the real component
jest.unmock("@/app/dashboard/components/panels/ConversationPanel");
jest.unmock("@/app/dashboard/components/panels/ConversationPanel/index");

// Import after unmocking
import ConversationPanel from "@/app/dashboard/components/panels/ConversationPanel";

// Mock the hook with a controllable mock
const mockUseConversationData = jest.fn();
jest.mock("@/hooks/useConversationData", () => ({
  useConversationData: () => mockUseConversationData(),
}));

// Mock icons
jest.mock("lucide-react", () => ({
  MessageSquare: ({ className }: any) => (
    <span className={className} data-testid="message-square-icon">
      MessageSquare
    </span>
  ),
  Brain: ({ className }: any) => (
    <span className={className} data-testid="brain-icon">
      Brain
    </span>
  ),
  Network: ({ className }: any) => (
    <span className={className} data-testid="network-icon">
      Network
    </span>
  ),
  Activity: ({ className }: any) => (
    <span className={className} data-testid="activity-icon">
      Activity
    </span>
  ),
  ChevronRight: ({ className }: any) => (
    <span className={className} data-testid="chevron-right-icon">
      ChevronRight
    </span>
  ),
  AlertCircle: ({ className }: any) => (
    <span className={className} data-testid="alert-circle-icon">
      AlertCircle
    </span>
  ),
  Loader2: ({ className }: any) => (
    <span className={className} data-testid="loader-icon">
      Loader2
    </span>
  ),
}));

describe("ConversationPanel Integration Tests", () => {
  const mockMessages = [
    {
      id: "1",
      type: "llm" as const,
      content: "LLM processing user input",
      timestamp: "10:30:15",
      metadata: { model: "gpt-4" },
    },
    {
      id: "2",
      type: "gnn" as const,
      content: "Graph neural network activated",
      timestamp: "10:30:16",
      metadata: { nodes: 150 },
    },
    {
      id: "3",
      type: "pymdp" as const,
      content: "Active inference computation complete",
      timestamp: "10:30:17",
      metadata: { belief_update: true },
    },
    {
      id: "4",
      type: "system" as const,
      content: "System status update",
      timestamp: "10:30:18",
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    // Default mock return value
    mockUseConversationData.mockReturnValue({
      messages: [],
      isLoading: false,
      error: null,
      isConnected: true,
      addMessage: jest.fn(),
      clearMessages: jest.fn(),
    });
  });

  describe("Component Rendering", () => {
    it("renders header with connection status", () => {
      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("CONNECTED")).toBeInTheDocument();
      expect(screen.getByText("Agent Loop")).toBeInTheDocument();
      expect(screen.getByText("LLM")).toBeInTheDocument();
      expect(screen.getByText("GNN")).toBeInTheDocument();
      expect(screen.getByText("PyMDP")).toBeInTheDocument();
    });

    it("shows agent flow visualization", () => {
      render(<ConversationPanel view="executive" />);

      const chevronIcons = screen.getAllByTestId("chevron-right-icon");
      expect(chevronIcons).toHaveLength(2);

      // Verify the flow: LLM -> GNN -> PyMDP
      const flowContainer = screen.getByText("LLM").closest("div");
      expect(flowContainer).toContainHTML("LLM");
      expect(flowContainer).toContainHTML("GNN");
      expect(flowContainer).toContainHTML("PyMDP");
    });

    it("renders empty state when no messages", () => {
      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("No messages yet")).toBeInTheDocument();
      expect(
        screen.getByText("Waiting for agent activity..."),
      ).toBeInTheDocument();
      expect(screen.getByTestId("message-square-icon")).toBeInTheDocument();
    });

    it("displays loading state", () => {
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: true,
        error: null,
        isConnected: false,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      expect(
        screen.getByText("Connecting to agent system..."),
      ).toBeInTheDocument();
      expect(screen.getByTestId("loader-icon")).toBeInTheDocument();
    });

    it("displays error state", () => {
      const mockError = new Error("Connection failed");
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: false,
        error: mockError,
        isConnected: false,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      expect(
        screen.getByText("Failed to connect: Connection failed"),
      ).toBeInTheDocument();
      expect(screen.getAllByTestId("alert-circle-icon")).toHaveLength(2); // One in error message, one in status bar
    });

    it("renders disconnected status", () => {
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: false,
        error: null,
        isConnected: false,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("DISCONNECTED")).toBeInTheDocument();
      expect(screen.getByText("Disconnected")).toBeInTheDocument();
    });
  });

  describe("Message Display", () => {
    beforeEach(() => {
      mockUseConversationData.mockReturnValue({
        messages: mockMessages,
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });
    });

    it("renders all message types with correct content", () => {
      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("LLM processing user input")).toBeInTheDocument();
      expect(
        screen.getByText("Graph neural network activated"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Active inference computation complete"),
      ).toBeInTheDocument();
      expect(screen.getByText("System status update")).toBeInTheDocument();
    });

    it("displays correct message type labels", () => {
      render(<ConversationPanel view="executive" />);

      // Note: LLM, GNN, PyMDP appear in header AND in message labels
      expect(screen.getAllByText("LLM")).toHaveLength(2);
      expect(screen.getAllByText("GNN")).toHaveLength(2);
      expect(screen.getAllByText("PyMDP")).toHaveLength(2);
      expect(screen.getByText("SYSTEM")).toBeInTheDocument();
    });

    it("shows timestamps for each message", () => {
      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("10:30:15")).toBeInTheDocument();
      expect(screen.getByText("10:30:16")).toBeInTheDocument();
      expect(screen.getByText("10:30:17")).toBeInTheDocument();
      expect(screen.getByText("10:30:18")).toBeInTheDocument();
    });

    it("displays correct icons for each message type", () => {
      render(<ConversationPanel view="executive" />);

      expect(screen.getByTestId("brain-icon")).toBeInTheDocument();
      expect(screen.getByTestId("network-icon")).toBeInTheDocument();
      expect(screen.getAllByTestId("activity-icon")).toHaveLength(2); // One in message, one in status bar
      expect(screen.getByTestId("message-square-icon")).toBeInTheDocument(); // Only one in message (no empty state)
    });

    it("applies correct styling classes for message types", () => {
      render(<ConversationPanel view="executive" />);

      // Find the message containers (with border and padding)
      const llmMessage = screen
        .getByText("LLM processing user input")
        .closest(".border");
      expect(llmMessage).toHaveClass("text-blue-400");

      const gnnMessage = screen
        .getByText("Graph neural network activated")
        .closest(".border");
      expect(gnnMessage).toHaveClass("text-green-400");

      const pymdpMessage = screen
        .getByText("Active inference computation complete")
        .closest(".border");
      expect(pymdpMessage).toHaveClass("text-purple-400");

      const systemMessage = screen
        .getByText("System status update")
        .closest(".border");
      expect(systemMessage).toHaveClass("text-[var(--text-tertiary)]");
    });
  });

  describe("Status Bar", () => {
    it("shows ready status when connected", () => {
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("Ready")).toBeInTheDocument();
      expect(screen.getByTestId("activity-icon")).toBeInTheDocument();
    });

    it("shows disconnected status when not connected", () => {
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: false,
        error: null,
        isConnected: false,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("Disconnected")).toBeInTheDocument();
      expect(screen.getByTestId("alert-circle-icon")).toBeInTheDocument();
    });
  });

  describe("Connection Status Indicator", () => {
    it("shows animated pulse when connected", () => {
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      const indicator = screen.getByText("CONNECTED").previousElementSibling;
      expect(indicator).toHaveClass("bg-green-400", "animate-pulse");
    });

    it("shows red indicator when disconnected", () => {
      mockUseConversationData.mockReturnValue({
        messages: [],
        isLoading: false,
        error: null,
        isConnected: false,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      const indicator = screen.getByText("DISCONNECTED").previousElementSibling;
      expect(indicator).toHaveClass("bg-red-400");
      expect(indicator).not.toHaveClass("animate-pulse");
    });
  });

  describe("Message Updates", () => {
    it("updates when new messages are added", () => {
      const { rerender } = render(<ConversationPanel view="executive" />);

      expect(screen.getByText("No messages yet")).toBeInTheDocument();

      // Simulate new messages being added
      mockUseConversationData.mockReturnValue({
        messages: [mockMessages[0]],
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      rerender(<ConversationPanel view="executive" />);

      expect(screen.queryByText("No messages yet")).not.toBeInTheDocument();
      expect(screen.getByText("LLM processing user input")).toBeInTheDocument();
    });
  });

  describe("View Types", () => {
    it("accepts different view types", () => {
      const views: Array<"executive" | "technical" | "research" | "minimal"> = [
        "executive",
        "technical",
        "research",
        "minimal",
      ];

      views.forEach((view) => {
        const { unmount } = render(<ConversationPanel view={view} />);
        expect(screen.getByText("Agent Loop")).toBeInTheDocument();
        unmount();
      });
    });
  });

  describe("Error Handling", () => {
    it("handles different error types", () => {
      const testErrors = [
        new Error("Network timeout"),
        new Error("Authentication failed"),
        new Error("Server unavailable"),
      ];

      testErrors.forEach((error) => {
        mockUseConversationData.mockReturnValue({
          messages: [],
          isLoading: false,
          error,
          isConnected: false,
          addMessage: jest.fn(),
          clearMessages: jest.fn(),
        });

        const { unmount } = render(<ConversationPanel view="executive" />);

        expect(
          screen.getByText(`Failed to connect: ${error.message}`),
        ).toBeInTheDocument();
        unmount();
      });
    });

    it("handles messages with missing or invalid data", () => {
      const messagesWithMissingData = [
        {
          id: "1",
          type: "llm" as const,
          content: "",
          timestamp: "10:30:15",
        },
        {
          id: "2",
          type: "invalid" as any,
          content: "Test message",
          timestamp: "10:30:16",
        },
      ];

      mockUseConversationData.mockReturnValue({
        messages: messagesWithMissingData,
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      // Should not crash
      expect(() =>
        render(<ConversationPanel view="executive" />),
      ).not.toThrow();
    });
  });

  describe("Performance", () => {
    it("handles large number of messages", () => {
      const manyMessages = Array.from({ length: 1000 }, (_, i) => ({
        id: i.toString(),
        type: (i % 4 === 0
          ? "llm"
          : i % 4 === 1
            ? "gnn"
            : i % 4 === 2
              ? "pymdp"
              : "system") as const,
        content: `Message ${i}`,
        timestamp: `10:${String(i % 60).padStart(2, "0")}:${String(i % 60).padStart(2, "0")}`,
      }));

      mockUseConversationData.mockReturnValue({
        messages: manyMessages,
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      const { container } = render(<ConversationPanel view="executive" />);

      // Should render without performance issues
      expect(
        container.querySelectorAll('[class*="p-3 rounded-lg border"]'),
      ).toHaveLength(1000);
    });
  });

  describe("Auto-scroll Behavior", () => {
    it("calls scroll to bottom on message updates", () => {
      const mockScrollTo = jest.fn();
      const mockDiv = {
        scrollTop: 0,
        scrollHeight: 1000,
        set scrollTop(value: number) {
          mockScrollTo(value);
        },
        get scrollTop() {
          return 0;
        },
      };

      jest.spyOn(React, "useRef").mockReturnValue({ current: mockDiv as any });

      render(<ConversationPanel view="executive" />);

      // Simulate useEffect being called
      act(() => {
        mockUseConversationData.mockReturnValue({
          messages: [mockMessages[0]],
          isLoading: false,
          error: null,
          isConnected: true,
          addMessage: jest.fn(),
          clearMessages: jest.fn(),
        });
      });
    });
  });

  describe("Accessibility", () => {
    it("has proper semantic structure", () => {
      mockUseConversationData.mockReturnValue({
        messages: mockMessages,
        isLoading: false,
        error: null,
        isConnected: true,
        addMessage: jest.fn(),
        clearMessages: jest.fn(),
      });

      render(<ConversationPanel view="executive" />);

      // Messages should be contained in a scrollable region
      const messagesContainer = screen
        .getByText("LLM processing user input")
        .closest(".overflow-y-auto");
      expect(messagesContainer).toBeInTheDocument();
    });

    it("provides meaningful text for status indicators", () => {
      render(<ConversationPanel view="executive" />);

      expect(screen.getByText("CONNECTED")).toBeInTheDocument();
      expect(screen.getByText("Ready")).toBeInTheDocument();
    });
  });
});
