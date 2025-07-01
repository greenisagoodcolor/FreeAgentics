/**
 * PHASE 2: DEEP COMPONENT INTERACTION TESTS - MEMORYVIEWER
 *
 * Target: components/memoryviewer.tsx (2,272 lines)
 * Strategy: Test ALL component interactions, state changes, and business logic
 * Goal: Push coverage from 6.42% to 15%+ by testing the largest component thoroughly
 */

import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import { jest } from "@jest/globals";
import MemoryViewer, {
  type AgentToolPermissions,
} from "@/components/memoryviewer";
import type { Agent, Conversation, KnowledgeEntry } from "@/lib/types";

// Complete mock setup for all dependencies
jest.mock("@/hooks/use-toast", () => ({
  useToast: jest.fn(),
}));

jest.mock("@/contexts/llm-context", () => ({
  useLLM: jest.fn(),
}));

jest.mock("@/lib/utils", () => ({
  formatTimestamp: jest.fn((date) => date.toISOString()),
  extractTagsFromMarkdown: jest.fn((content) => content.match(/#(\w+)/g) || []),
  cn: jest.fn((...args) => args.filter(Boolean).join(" ")),
}));

jest.mock("@/lib/belief-extraction", () => ({
  parseBeliefs: jest.fn(),
  parseRefinedBeliefs: jest.fn(),
}));

jest.mock("@/lib/knowledge-export", () => ({
  exportAgentKnowledge: jest.fn(),
}));

jest.mock("@/lib/debug-logger", () => ({
  debugLog: jest.fn(),
  createLogger: jest.fn(() => ({
    log: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  })),
}));

// Mock all UI components to focus on logic
jest.mock("@/components/ui/button", () => ({
  Button: ({ children, onClick, variant, disabled, ...props }: any) => (
    <button
      onClick={onClick}
      disabled={disabled}
      data-variant={variant}
      {...props}
    >
      {children}
    </button>
  ),
}));

jest.mock("@/components/ui/textarea", () => ({
  Textarea: ({ value, onChange, placeholder, disabled, ...props }: any) => (
    <textarea
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      {...props}
    />
  ),
}));

jest.mock("@/components/ui/card", () => ({
  Card: ({ children, className, ...props }: any) => (
    <div className={className} {...props}>
      {children}
    </div>
  ),
  CardContent: ({ children, className, ...props }: any) => (
    <div className={className} {...props}>
      {children}
    </div>
  ),
  CardHeader: ({ children, className, ...props }: any) => (
    <div className={className} {...props}>
      {children}
    </div>
  ),
  CardTitle: ({ children, className, ...props }: any) => (
    <h3 className={className} {...props}>
      {children}
    </h3>
  ),
}));

jest.mock("@/components/ui/select", () => ({
  Select: ({ children, value, onValueChange, ...props }: any) => (
    <div data-testid="select-container" {...props}>
      <select value={value} onChange={(e) => onValueChange?.(e.target.value)}>
        {children}
      </select>
    </div>
  ),
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ value, children }: any) => (
    <option value={value}>{children}</option>
  ),
  SelectTrigger: ({ children }: any) => (
    <div data-testid="select-trigger">{children}</div>
  ),
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
}));

jest.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children, className, ...props }: any) => (
    <div className={className} {...props}>
      {children}
    </div>
  ),
}));

jest.mock("@/components/ui/input", () => ({
  Input: ({ value, onChange, placeholder, type, disabled, ...props }: any) => (
    <input
      type={type || "text"}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      {...props}
    />
  ),
}));

jest.mock("@/components/ui/badge", () => ({
  Badge: ({ children, variant, className, ...props }: any) => (
    <span className={className} data-variant={variant} {...props}>
      {children}
    </span>
  ),
}));

// Mock data structures
const createMockKnowledgeEntry = (
  id: string,
  overrides: Partial<KnowledgeEntry> = {},
): KnowledgeEntry => ({
  id,
  title: `Knowledge ${id}`,
  content: `Content for knowledge ${id}`,
  source: "user",
  timestamp: new Date("2023-01-01T12:00:00Z"),
  tags: ["tag1", "tag2"],
  metadata: {},
  ...overrides,
});

const createMockAgent = (
  id: string,
  overrides: Partial<Agent> = {},
): Agent => ({
  id,
  name: `Agent ${id}`,
  biography: `Biography for agent ${id}`,
  color: "#ff0000",
  position: { x: 0, y: 0 },
  knowledge: [createMockKnowledgeEntry(`knowledge-${id}`)],
  toolPermissions: {
    internetSearch: true,
    webScraping: false,
    wikipediaAccess: true,
    newsApi: false,
    academicSearch: true,
    documentRetrieval: false,
    imageGeneration: false,
    textSummarization: true,
    translation: false,
    codeExecution: false,
    calculator: true,
    knowledgeGraphQuery: false,
    factChecking: true,
    timelineGenerator: false,
    weatherData: false,
    mapLocationData: false,
    financialData: false,
    publicDatasets: false,
    memorySearch: true,
    crossAgentKnowledge: false,
    conversationAnalysis: true,
  },
  autonomyEnabled: true,
  inConversation: false,
  ...overrides,
});

const createMockConversation = (
  id: string,
  overrides: Partial<Conversation> = {},
): Conversation => ({
  id,
  participants: ["agent-1"],
  messages: [
    {
      id: `msg-${id}-1`,
      conversationId: id,
      senderId: "agent-1",
      content: `Message content for conversation ${id}`,
      timestamp: new Date("2023-01-01T12:00:00Z"),
      type: "text",
      metadata: {},
    },
  ],
  createdAt: new Date("2023-01-01T12:00:00Z"),
  updatedAt: new Date("2023-01-01T12:00:00Z"),
  title: `Conversation ${id}`,
  metadata: {},
  ...overrides,
});

describe("MemoryViewer - Deep Component Interactions", () => {
  let mockToast: jest.Mock;
  let mockUseLLM: jest.Mock;
  let mockOnAddKnowledge: jest.Mock;
  let mockOnUpdateAgent: jest.Mock;
  let mockOnDeleteKnowledge: jest.Mock;
  let mockOnUpdateKnowledge: jest.Mock;
  let mockOnClearSelectedKnowledgeNode: jest.Mock;
  let mockOnSelectAgent: jest.Mock;

  const baseAgent = createMockAgent("agent-1");
  const baseConversation = createMockConversation("conv-1");

  beforeEach(() => {
    jest.clearAllMocks();

    mockToast = jest.fn();
    mockUseLLM = jest.fn(() => ({
      isProcessing: false,
      setIsProcessing: jest.fn(),
      generateResponse: jest.fn(),
      extractBeliefs: jest.fn(),
      generateKnowledgeEntries: jest.fn(),
      llmClient: null,
    }));

    mockOnAddKnowledge = jest.fn();
    mockOnUpdateAgent = jest.fn();
    mockOnDeleteKnowledge = jest.fn();
    mockOnUpdateKnowledge = jest.fn();
    mockOnClearSelectedKnowledgeNode = jest.fn();
    mockOnSelectAgent = jest.fn();

    // Setup mocks
    const { useToast } = jest.requireMock("@/hooks/use-toast");
    const { useLLM } = jest.requireMock("@/contexts/llm-context");
    const { formatTimestamp, extractTagsFromMarkdown } =
      jest.requireMock("@/lib/utils");

    useToast.mockReturnValue({
      toast: mockToast,
      dismiss: jest.fn(),
      toasts: [],
    });

    useLLM.mockImplementation(mockUseLLM);
    formatTimestamp.mockImplementation((date) => date.toISOString());
    extractTagsFromMarkdown.mockImplementation(
      (content) => content.match(/#(\w+)/g) || [],
    );
  });

  const renderMemoryViewer = (
    props: Partial<React.ComponentProps<typeof MemoryViewer>> = {},
  ) => {
    const defaultProps = {
      selectedAgent: baseAgent,
      conversationHistory: [baseConversation],
      agents: [baseAgent],
      onAddKnowledge: mockOnAddKnowledge,
      onUpdateAgent: mockOnUpdateAgent,
      onDeleteKnowledge: mockOnDeleteKnowledge,
      onUpdateKnowledge: mockOnUpdateKnowledge,
      selectedKnowledgeNode: null,
      onClearSelectedKnowledgeNode: mockOnClearSelectedKnowledgeNode,
      onSelectAgent: mockOnSelectAgent,
      ...props,
    };

    return render(<MemoryViewer {...defaultProps} />);
  };

  describe("Component State Management", () => {
    test("manages tab navigation state correctly", () => {
      renderMemoryViewer();

      // Should start with Biography tab
      expect(
        screen.getByDisplayValue("Biography for agent agent-1"),
      ).toBeInTheDocument();

      // Navigate to Knowledge tab
      const knowledgeTab = screen.getByText("Knowledge");
      fireEvent.click(knowledgeTab);

      expect(screen.getByText("Knowledge agent-1")).toBeInTheDocument();
    });

    test("manages knowledge view state transitions", () => {
      renderMemoryViewer();

      // Navigate to knowledge tab
      fireEvent.click(screen.getByText("Knowledge"));

      // Should show knowledge list by default
      expect(screen.getByText("Knowledge agent-1")).toBeInTheDocument();

      // Switch to Add tab
      fireEvent.click(screen.getByText("Add"));
      expect(
        screen.getByPlaceholderText("Knowledge title..."),
      ).toBeInTheDocument();

      // Go back to List tab
      fireEvent.click(screen.getByText("List"));
      expect(screen.getByText("Knowledge agent-1")).toBeInTheDocument();
    });

    test("manages tool permissions editing state", () => {
      renderMemoryViewer();

      // Navigate to Tools tab
      fireEvent.click(screen.getByText("Tools"));

      // Check initial state
      expect(screen.getByText("Information Access Tools")).toBeInTheDocument();

      // Toggle a permission
      const internetSearchToggle = screen.getByLabelText("Internet Search");
      fireEvent.click(internetSearchToggle);

      // Should show unsaved changes indicator
      expect(screen.getByText("Unsaved Changes")).toBeInTheDocument();
    });

    test("manages belief extraction workflow state", () => {
      renderMemoryViewer();

      // Navigate to Inference tab
      fireEvent.click(screen.getByText("Inference"));

      expect(
        screen.getByText("Extract Beliefs from Conversation"),
      ).toBeInTheDocument();

      // Select a conversation
      const conversationSelect = screen.getByDisplayValue(
        "Select a conversation...",
      );
      fireEvent.change(conversationSelect, { target: { value: "conv-1" } });

      expect(
        screen.getByDisplayValue("Conversation conv-1"),
      ).toBeInTheDocument();
    });
  });

  describe("Biography Management Deep Testing", () => {
    test("handles biography editing workflow completely", async () => {
      renderMemoryViewer();

      const biographyTextarea = screen.getByDisplayValue(
        "Biography for agent agent-1",
      );

      // Test editing
      fireEvent.change(biographyTextarea, {
        target: { value: "New biography content" },
      });
      expect(biographyTextarea).toHaveValue("New biography content");

      // Test saving
      const saveButton = screen.getByText("Save Biography");
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(mockOnUpdateAgent).toHaveBeenCalledWith("agent-1", {
          biography: "New biography content",
        });
      });

      expect(mockToast).toHaveBeenCalledWith({
        title: "Success",
        description: "Biography saved successfully",
      });
    });

    test("handles biography save errors gracefully", async () => {
      mockOnUpdateAgent.mockRejectedValue(new Error("Save failed"));
      renderMemoryViewer();

      const biographyTextarea = screen.getByDisplayValue(
        "Biography for agent agent-1",
      );
      fireEvent.change(biographyTextarea, { target: { value: "New content" } });

      const saveButton = screen.getByText("Save Biography");
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: "Error",
          description: "Failed to save biography: Save failed",
          variant: "destructive",
        });
      });
    });

    test("disables save button when no changes made", () => {
      renderMemoryViewer();

      const saveButton = screen.getByText("Save Biography");
      expect(saveButton).toBeDisabled();

      // Make a change
      const biographyTextarea = screen.getByDisplayValue(
        "Biography for agent agent-1",
      );
      fireEvent.change(biographyTextarea, { target: { value: "Changed" } });

      expect(saveButton).not.toBeDisabled();
    });
  });

  describe("Knowledge Management Deep Testing", () => {
    test("handles knowledge creation workflow completely", async () => {
      renderMemoryViewer();

      // Navigate to knowledge and add
      fireEvent.click(screen.getByText("Knowledge"));
      fireEvent.click(screen.getByText("Add"));

      // Fill form
      const titleInput = screen.getByPlaceholderText("Knowledge title...");
      const contentTextarea = screen.getByPlaceholderText(
        "Knowledge content...",
      );
      const tagsInput = screen.getByPlaceholderText(
        "Tags (comma-separated)...",
      );

      fireEvent.change(titleInput, {
        target: { value: "New Knowledge Title" },
      });
      fireEvent.change(contentTextarea, {
        target: { value: "New knowledge content" },
      });
      fireEvent.change(tagsInput, { target: { value: "tag1, tag2, tag3" } });

      // Save
      const saveButton = screen.getByText("Save Knowledge");
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(mockOnAddKnowledge).toHaveBeenCalledWith(
          "agent-1",
          expect.objectContaining({
            title: "New Knowledge Title",
            content: "New knowledge content",
            tags: ["tag1", "tag2", "tag3"],
            source: "user",
          }),
        );
      });

      expect(mockToast).toHaveBeenCalledWith({
        title: "Success",
        description: "Knowledge saved successfully",
      });
    });

    test("handles knowledge editing workflow completely", async () => {
      const agentWithKnowledge = createMockAgent("agent-1", {
        knowledge: [
          createMockKnowledgeEntry("knowledge-1", {
            title: "Existing Knowledge",
            content: "Existing content",
            tags: ["existing", "tags"],
          }),
        ],
      });

      renderMemoryViewer({ selectedAgent: agentWithKnowledge });

      // Navigate to knowledge
      fireEvent.click(screen.getByText("Knowledge"));

      // Select knowledge entry
      const knowledgeItem = screen.getByText("Existing Knowledge");
      fireEvent.click(knowledgeItem);

      // Enter edit mode
      fireEvent.click(screen.getByText("Edit"));

      // Edit content
      const contentTextarea = screen.getByDisplayValue("Existing content");
      fireEvent.change(contentTextarea, {
        target: { value: "Updated content" },
      });

      // Save changes
      fireEvent.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(mockOnUpdateKnowledge).toHaveBeenCalledWith(
          "agent-1",
          "knowledge-1",
          {
            content: "Updated content",
          },
        );
      });
    });

    test("handles knowledge deletion workflow with confirmation", async () => {
      const agentWithKnowledge = createMockAgent("agent-1", {
        knowledge: [createMockKnowledgeEntry("knowledge-1")],
      });

      renderMemoryViewer({ selectedAgent: agentWithKnowledge });

      fireEvent.click(screen.getByText("Knowledge"));

      // Select and delete
      fireEvent.click(screen.getByText("Knowledge knowledge-1"));
      fireEvent.click(screen.getByText("Delete"));

      // Confirm deletion
      fireEvent.click(screen.getByText("Confirm Delete"));

      await waitFor(() => {
        expect(mockOnDeleteKnowledge).toHaveBeenCalledWith(
          "agent-1",
          "knowledge-1",
        );
      });
    });

    test("handles knowledge search functionality", () => {
      const agentWithMultipleKnowledge = createMockAgent("agent-1", {
        knowledge: [
          createMockKnowledgeEntry("knowledge-1", {
            title: "First Knowledge",
            content: "First content",
          }),
          createMockKnowledgeEntry("knowledge-2", {
            title: "Second Knowledge",
            content: "Second content",
          }),
          createMockKnowledgeEntry("knowledge-3", {
            title: "Third Knowledge",
            content: "Third content",
          }),
        ],
      });

      renderMemoryViewer({ selectedAgent: agentWithMultipleKnowledge });

      fireEvent.click(screen.getByText("Knowledge"));

      // Search for specific knowledge
      const searchInput = screen.getByPlaceholderText("Search knowledge...");
      fireEvent.change(searchInput, { target: { value: "First" } });

      expect(screen.getByText("First Knowledge")).toBeInTheDocument();
      expect(screen.queryByText("Second Knowledge")).not.toBeInTheDocument();
      expect(screen.queryByText("Third Knowledge")).not.toBeInTheDocument();
    });

    test("handles knowledge tag filtering", () => {
      const agentWithTaggedKnowledge = createMockAgent("agent-1", {
        knowledge: [
          createMockKnowledgeEntry("knowledge-1", {
            tags: ["science", "physics"],
          }),
          createMockKnowledgeEntry("knowledge-2", {
            tags: ["math", "algebra"],
          }),
          createMockKnowledgeEntry("knowledge-3", {
            tags: ["science", "chemistry"],
          }),
        ],
      });

      renderMemoryViewer({ selectedAgent: agentWithTaggedKnowledge });

      fireEvent.click(screen.getByText("Knowledge"));

      // Filter by tag
      const tagSelect = screen.getByDisplayValue("All Tags");
      fireEvent.change(tagSelect, { target: { value: "science" } });

      // Should show only science-tagged knowledge
      expect(screen.getByText("Knowledge knowledge-1")).toBeInTheDocument();
      expect(
        screen.queryByText("Knowledge knowledge-2"),
      ).not.toBeInTheDocument();
      expect(screen.getByText("Knowledge knowledge-3")).toBeInTheDocument();
    });
  });

  describe("Tool Permissions Deep Testing", () => {
    test("handles comprehensive tool permission updates", async () => {
      renderMemoryViewer();

      fireEvent.click(screen.getByText("Tools"));

      // Toggle multiple permissions
      const internetSearch = screen.getByLabelText("Internet Search");
      const calculator = screen.getByLabelText("Calculator");
      const memorySearch = screen.getByLabelText("Memory Search");

      fireEvent.click(internetSearch); // true -> false
      fireEvent.click(calculator); // true -> false
      fireEvent.click(memorySearch); // true -> false

      // Save changes
      fireEvent.click(screen.getByText("Save Tool Permissions"));

      await waitFor(() => {
        expect(mockOnUpdateAgent).toHaveBeenCalledWith("agent-1", {
          toolPermissions: expect.objectContaining({
            internetSearch: false,
            calculator: false,
            memorySearch: false,
            // Others should remain unchanged
            wikipediaAccess: true,
            academicSearch: true,
          }),
        });
      });
    });

    test("handles tool permission category grouping", () => {
      renderMemoryViewer();

      fireEvent.click(screen.getByText("Tools"));

      // Check all categories are present
      expect(screen.getByText("Information Access Tools")).toBeInTheDocument();
      expect(
        screen.getByText("Content Generation & Processing"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Knowledge & Reasoning Tools"),
      ).toBeInTheDocument();
      expect(screen.getByText("External Integrations")).toBeInTheDocument();
      expect(screen.getByText("Agent-Specific Tools")).toBeInTheDocument();
    });

    test("handles tool permission save errors", async () => {
      mockOnUpdateAgent.mockRejectedValue(
        new Error("Permission update failed"),
      );
      renderMemoryViewer();

      fireEvent.click(screen.getByText("Tools"));
      fireEvent.click(screen.getByLabelText("Internet Search"));
      fireEvent.click(screen.getByText("Save Tool Permissions"));

      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: "Error",
          description:
            "Failed to update tool permissions: Permission update failed",
          variant: "destructive",
        });
      });
    });
  });

  describe("Belief Extraction Deep Testing", () => {
    test("handles complete belief extraction workflow", async () => {
      const mockExtractBeliefs = jest
        .fn()
        .mockResolvedValue("Extracted beliefs text");
      mockUseLLM.mockReturnValue({
        isProcessing: false,
        setIsProcessing: jest.fn(),
        generateResponse: jest.fn(),
        extractBeliefs: mockExtractBeliefs,
        generateKnowledgeEntries: jest.fn(),
        llmClient: {},
      });

      renderMemoryViewer();

      fireEvent.click(screen.getByText("Inference"));

      // Select conversation
      const conversationSelect = screen.getByDisplayValue(
        "Select a conversation...",
      );
      fireEvent.change(conversationSelect, { target: { value: "conv-1" } });

      // Extract beliefs
      fireEvent.click(screen.getByText("Extract Beliefs"));

      await waitFor(() => {
        expect(mockExtractBeliefs).toHaveBeenCalledWith(
          baseConversation.messages.map((m) => m.content).join("\n"),
        );
      });

      // Check extracted beliefs are displayed
      expect(screen.getByText("Extracted beliefs text")).toBeInTheDocument();
    });

    test("handles belief extraction processing state", async () => {
      const mockSetIsProcessing = jest.fn();
      mockUseLLM.mockReturnValue({
        isProcessing: true,
        setIsProcessing: mockSetIsProcessing,
        generateResponse: jest.fn(),
        extractBeliefs: jest
          .fn()
          .mockImplementation(() => new Promise(() => {})), // Never resolves
        generateKnowledgeEntries: jest.fn(),
        llmClient: {},
      });

      renderMemoryViewer();

      fireEvent.click(screen.getByText("Inference"));

      const conversationSelect = screen.getByDisplayValue(
        "Select a conversation...",
      );
      fireEvent.change(conversationSelect, { target: { value: "conv-1" } });

      fireEvent.click(screen.getByText("Extract Beliefs"));

      // Should show processing state
      expect(screen.getByText("Extracting...")).toBeInTheDocument();
    });

    test("handles belief extraction errors", async () => {
      const mockExtractBeliefs = jest
        .fn()
        .mockRejectedValue(new Error("Extraction failed"));
      mockUseLLM.mockReturnValue({
        isProcessing: false,
        setIsProcessing: jest.fn(),
        generateResponse: jest.fn(),
        extractBeliefs: mockExtractBeliefs,
        generateKnowledgeEntries: jest.fn(),
        llmClient: {},
      });

      renderMemoryViewer();

      fireEvent.click(screen.getByText("Inference"));

      const conversationSelect = screen.getByDisplayValue(
        "Select a conversation...",
      );
      fireEvent.change(conversationSelect, { target: { value: "conv-1" } });

      fireEvent.click(screen.getByText("Extract Beliefs"));

      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: "Error",
          description: "Failed to extract beliefs: Extraction failed",
          variant: "destructive",
        });
      });
    });

    test("handles conversation selection edge cases", () => {
      const conversationHistory = [
        createMockConversation("conv-1", { title: "First Conversation" }),
        createMockConversation("conv-2", { title: "Second Conversation" }),
      ];

      renderMemoryViewer({ conversationHistory });

      fireEvent.click(screen.getByText("Inference"));

      const conversationSelect = screen.getByDisplayValue(
        "Select a conversation...",
      );

      // Should have all conversations as options
      fireEvent.change(conversationSelect, { target: { value: "conv-2" } });
      expect(
        screen.getByDisplayValue("Second Conversation"),
      ).toBeInTheDocument();
    });
  });

  describe("Knowledge Node Selection Deep Testing", () => {
    test("handles knowledge node selection from global graph", () => {
      const selectedKnowledgeNode = {
        type: "entry" as const,
        id: "knowledge-1",
        title: "Selected Knowledge",
      };

      renderMemoryViewer({ selectedKnowledgeNode });

      expect(screen.getByText("Knowledge Node Selection")).toBeInTheDocument();
      expect(screen.getByText("Selected Knowledge")).toBeInTheDocument();
    });

    test("handles tag node selection from global graph", () => {
      const selectedKnowledgeNode = {
        type: "tag" as const,
        id: "science",
        title: "Science Tag",
      };

      renderMemoryViewer({ selectedKnowledgeNode });

      expect(screen.getByText("Knowledge Node Selection")).toBeInTheDocument();
      expect(screen.getByText("Science Tag")).toBeInTheDocument();
    });

    test("handles clearing knowledge node selection", () => {
      const selectedKnowledgeNode = {
        type: "entry" as const,
        id: "knowledge-1",
        title: "Selected Knowledge",
      };

      renderMemoryViewer({ selectedKnowledgeNode });

      fireEvent.click(screen.getByText("Back"));

      expect(mockOnClearSelectedKnowledgeNode).toHaveBeenCalled();
    });
  });

  describe("Agent Selection and State Reset", () => {
    test("handles agent switching and state reset", () => {
      const { rerender } = renderMemoryViewer();

      // Go to knowledge tab and select an item
      fireEvent.click(screen.getByText("Knowledge"));
      fireEvent.click(screen.getByText("Knowledge knowledge-1"));

      // Switch to different agent
      const newAgent = createMockAgent("agent-2");
      rerender(
        <MemoryViewer
          selectedAgent={newAgent}
          conversationHistory={[baseConversation]}
          agents={[baseAgent, newAgent]}
          onAddKnowledge={mockOnAddKnowledge}
          onUpdateAgent={mockOnUpdateAgent}
          onDeleteKnowledge={mockOnDeleteKnowledge}
          onUpdateKnowledge={mockOnUpdateKnowledge}
          selectedKnowledgeNode={null}
          onClearSelectedKnowledgeNode={mockOnClearSelectedKnowledgeNode}
          onSelectAgent={mockOnSelectAgent}
        />,
      );

      // Should show new agent's biography
      expect(
        screen.getByDisplayValue("Biography for agent agent-2"),
      ).toBeInTheDocument();
    });

    test("handles null agent gracefully", () => {
      renderMemoryViewer({ selectedAgent: null });

      expect(
        screen.getByText("Select an agent to view their memory"),
      ).toBeInTheDocument();
    });
  });

  describe("Data Export Deep Testing", () => {
    test("handles knowledge export workflow", async () => {
      const { exportAgentKnowledge } = jest.requireMock(
        "@/lib/knowledge-export",
      );
      exportAgentKnowledge.mockResolvedValue("exported-data");

      renderMemoryViewer();

      fireEvent.click(screen.getByText("Knowledge"));
      fireEvent.click(screen.getByText("Export Knowledge"));

      await waitFor(() => {
        expect(exportAgentKnowledge).toHaveBeenCalledWith(baseAgent);
      });

      expect(mockToast).toHaveBeenCalledWith({
        title: "Success",
        description: "Knowledge exported successfully",
      });
    });

    test("handles export errors", async () => {
      const { exportAgentKnowledge } = jest.requireMock(
        "@/lib/knowledge-export",
      );
      exportAgentKnowledge.mockRejectedValue(new Error("Export failed"));

      renderMemoryViewer();

      fireEvent.click(screen.getByText("Knowledge"));
      fireEvent.click(screen.getByText("Export Knowledge"));

      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: "Error",
          description: "Failed to export knowledge: Export failed",
          variant: "destructive",
        });
      });
    });
  });

  describe("Performance and Edge Cases", () => {
    test("handles large knowledge datasets efficiently", () => {
      const largeKnowledgeSet = Array.from({ length: 500 }, (_, i) =>
        createMockKnowledgeEntry(`knowledge-${i}`, {
          title: `Knowledge ${i}`,
          content: `Content ${i}`,
          tags: [`tag-${i % 10}`, "common"],
        }),
      );

      const agentWithLargeKnowledge = createMockAgent("agent-1", {
        knowledge: largeKnowledgeSet,
      });

      const startTime = Date.now();
      renderMemoryViewer({ selectedAgent: agentWithLargeKnowledge });
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(1000);

      // Test search performance
      fireEvent.click(screen.getByText("Knowledge"));

      const searchInput = screen.getByPlaceholderText("Search knowledge...");
      fireEvent.change(searchInput, { target: { value: "Knowledge 100" } });

      expect(screen.getByText("Knowledge 100")).toBeInTheDocument();
    });

    test("handles malformed data gracefully", () => {
      const malformedAgent = {
        id: "malformed",
        name: "Malformed Agent",
        biography: "",
        knowledge: [
          { title: "Malformed Knowledge" }, // Missing required fields
        ],
      } as any;

      expect(() => {
        renderMemoryViewer({ selectedAgent: malformedAgent });
      }).not.toThrow();
    });

    test("handles missing dependencies gracefully", () => {
      const agentWithoutPermissions = {
        ...baseAgent,
        toolPermissions: null,
      } as any;

      renderMemoryViewer({ selectedAgent: agentWithoutPermissions });

      fireEvent.click(screen.getByText("Tools"));

      // Should use default permissions
      expect(screen.getByText("Information Access Tools")).toBeInTheDocument();
    });
  });
});
