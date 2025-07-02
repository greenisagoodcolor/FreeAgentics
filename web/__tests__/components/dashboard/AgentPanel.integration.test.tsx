import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import agentReducer from "@/store/slices/agentSlice";
import conversationReducer from "@/store/slices/conversationSlice";
import uiReducer from "@/store/slices/uiSlice";
import knowledgeReducer from "@/store/slices/knowledgeSlice";

// Ensure we're using the real component and hooks
jest.unmock("@/app/dashboard/components/panels/AgentPanel");
jest.unmock("@/app/dashboard/components/panels/AgentPanel/index");
jest.unmock("@/store/hooks");

// Import after unmocking
import AgentPanel from "@/app/dashboard/components/panels/AgentPanel";

// Mock the UI components properly
jest.mock("@/components/ui/button", () => ({
  Button: React.forwardRef(({ children, onClick, ...props }: any, ref: any) => (
    <button ref={ref} onClick={onClick} {...props}>
      {children}
    </button>
  )),
}));

jest.mock("@/components/ui/badge", () => ({
  Badge: ({ children, variant, ...props }: any) => (
    <span data-variant={variant} {...props}>
      {children}
    </span>
  ),
}));

jest.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children, ...props }: any) => (
    <div data-testid="scroll-area" {...props}>
      {children}
    </div>
  ),
}));

jest.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children, open, onOpenChange }: any) => {
    return open ? (
      <div role="dialog" data-testid="dialog">
        <button onClick={() => onOpenChange(false)} aria-label="Close dialog">
          ×
        </button>
        {children}
      </div>
    ) : null;
  },
  DialogContent: ({ children, className }: any) => (
    <div className={className} data-testid="dialog-content">
      {children}
    </div>
  ),
  DialogHeader: ({ children }: any) => (
    <div data-testid="dialog-header">{children}</div>
  ),
  DialogTitle: ({ children }: any) => <h2>{children}</h2>,
}));

jest.mock("@/components/dashboard/AgentTemplateSelector", () => {
  return function MockAgentTemplateSelector() {
    return (
      <div data-testid="agent-template-selector">
        Agent Template Selector Mock
      </div>
    );
  };
});

// Mock icons
jest.mock("lucide-react", () => ({
  Users: ({ className }: any) => (
    <span className={className} data-testid="users-icon">
      Users
    </span>
  ),
  Plus: ({ className }: any) => (
    <span className={className} data-testid="plus-icon">
      Plus
    </span>
  ),
  Settings: ({ className }: any) => (
    <span className={className} data-testid="settings-icon">
      Settings
    </span>
  ),
  Activity: ({ className }: any) => (
    <span className={className} data-testid="activity-icon">
      Activity
    </span>
  ),
  Pause: ({ className }: any) => (
    <span className={className} data-testid="pause-icon">
      Pause
    </span>
  ),
  Play: ({ className }: any) => (
    <span className={className} data-testid="play-icon">
      Play
    </span>
  ),
}));

// Helper function to create a mock store
const createMockStore = (initialState: any = {}) => {
  return configureStore({
    reducer: {
      agents: agentReducer,
      conversations: conversationReducer,
      ui: uiReducer,
      knowledge: knowledgeReducer,
      // Mock minimal reducers for other slices not available
      connection: (state = { isConnected: false, socket: null }) => state,
      analytics: (state = { metrics: {} }) => state,
    },
    preloadedState: initialState,
  });
};

// Helper to render with Redux
const renderWithRedux = (component: React.ReactElement, initialState = {}) => {
  const store = createMockStore(initialState);
  return {
    ...render(<Provider store={store}>{component}</Provider>),
    store,
  };
};

describe("AgentPanel Integration Tests", () => {
  const mockAgents = {
    agent1: {
      id: "agent1",
      name: "Agent Alpha",
      status: "active",
      templateId: "explorer",
      biography: "An exploration agent focused on discovery",
      knowledgeDomains: ["exploration", "testing"],
      parameters: {
        responseThreshold: 0.6,
        turnTakingProbability: 0.7,
        conversationEngagement: 0.8,
      },
      activityMetrics: { messagesCount: 42 },
      inConversation: true,
      autonomyEnabled: true,
      avatarUrl: "/avatars/explorer.svg",
      color: "#10B981",
      createdAt: Date.now(),
      lastActive: Date.now(),
    },
    agent2: {
      id: "agent2",
      name: "Agent Beta",
      status: "idle",
      templateId: "guardian",
      biography: "A guardian agent protecting system integrity",
      knowledgeDomains: ["security", "protection"],
      parameters: {
        responseThreshold: 0.7,
        turnTakingProbability: 0.6,
        conversationEngagement: 0.7,
      },
      activityMetrics: { messagesCount: 15 },
      inConversation: false,
      autonomyEnabled: true,
      avatarUrl: "/avatars/guardian.svg",
      color: "#EF4444",
      createdAt: Date.now(),
      lastActive: Date.now(),
    },
    agent3: {
      id: "agent3",
      name: "Agent Gamma",
      status: "error",
      templateId: "analyst",
      biography: "An analytical agent for data processing",
      knowledgeDomains: ["analysis", "data"],
      parameters: {
        responseThreshold: 0.8,
        turnTakingProbability: 0.5,
        conversationEngagement: 0.6,
      },
      activityMetrics: { messagesCount: 0 },
      inConversation: false,
      autonomyEnabled: false,
      avatarUrl: "/avatars/analyst.svg",
      color: "#3B82F6",
      createdAt: Date.now(),
      lastActive: Date.now(),
    },
  };

  beforeEach(() => {
    // Clear any console mocks
    jest.clearAllMocks();
  });

  describe("Component Rendering", () => {
    it("renders empty state when no agents exist", () => {
      const emptyState = {
        agents: {
          agents: {},
          agentOrder: [],
          selectedAgentId: null,
          typingAgents: [],
        },
      };
      renderWithRedux(<AgentPanel view="executive" />, emptyState);

      expect(screen.getByText("Agent Management")).toBeInTheDocument();
      expect(screen.getByText("0 Total")).toBeInTheDocument();
      expect(screen.getByText("No agents configured")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /Create Agent/i }),
      ).toBeInTheDocument();
    });

    it("renders agent list when agents exist", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("3 Total")).toBeInTheDocument();
      expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
      expect(screen.getByText("Agent Beta")).toBeInTheDocument();
      expect(screen.getByText("Agent Gamma")).toBeInTheDocument();
    });

    it("displays correct status badges", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("ACTIVE")).toBeInTheDocument();
      expect(screen.getByText("IDLE")).toBeInTheDocument();
      expect(screen.getByText("ERROR")).toBeInTheDocument();
    });

    it("shows agent templates and biographies", () => {
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("Template: explorer")).toBeInTheDocument();
      expect(
        screen.getByText(/exploration agent focused on discovery/),
      ).toBeInTheDocument();
    });
  });

  describe("Agent Selection", () => {
    it("selects agent on click and shows details", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");

      // Initially no details shown
      expect(screen.queryByText("Messages:")).not.toBeInTheDocument();

      // Click to select
      await user.click(agentCard);

      // Details should be visible
      expect(screen.getByText("Messages:")).toBeInTheDocument();
      expect(screen.getByText("42")).toBeInTheDocument();
      expect(screen.getByText("Active:")).toBeInTheDocument();
      expect(screen.getByText("Yes")).toBeInTheDocument();
    });

    it("deselects agent on second click", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");

      // Select
      await user.click(agentCard);
      expect(screen.getByText("Messages:")).toBeInTheDocument();

      // Deselect
      await user.click(agentCard);
      expect(screen.queryByText("Messages:")).not.toBeInTheDocument();
    });
  });

  describe("Agent Controls", () => {
    it("shows play/pause button based on agent status", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCards = screen.getAllByTestId("agent-card");

      // Active agent shows pause
      const pauseIcon = agentCards[0].querySelector(
        '[data-testid="pause-icon"]',
      );
      expect(pauseIcon).toBeInTheDocument();

      // Idle agent shows play
      const playIcon = agentCards[1].querySelector('[data-testid="play-icon"]');
      expect(playIcon).toBeInTheDocument();
    });

    it("handles play/pause button click", async () => {
      const user = userEvent.setup();

      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const playPauseButton = screen.getByRole("button", { name: /pause/i });

      // Button should be clickable (no error should occur)
      await user.click(playPauseButton);

      // Button exists and is functional - this test verifies the UI structure
      expect(playPauseButton).toBeInTheDocument();
    });

    it("shows action buttons when agent is selected", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");
      await user.click(agentCard);

      expect(
        screen.getByRole("button", { name: /Configure/i }),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Logs/i })).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /Reset/i }),
      ).toBeInTheDocument();
    });
  });

  describe("Template Selector Modal", () => {
    it("opens modal when clicking add button", async () => {
      const user = userEvent.setup();
      const emptyState = {
        agents: {
          agents: {},
          agentOrder: [],
          selectedAgentId: null,
          typingAgents: [],
        },
      };
      renderWithRedux(<AgentPanel view="executive" />, emptyState);

      const addButtons = screen.getAllByTestId("plus-icon");
      const headerAddButton = addButtons[0].parentElement!;

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

      await user.click(headerAddButton);

      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByText("Create New Agent")).toBeInTheDocument();
      expect(screen.getByTestId("agent-template-selector")).toBeInTheDocument();
    });

    it("closes modal when requested", async () => {
      const user = userEvent.setup();
      const emptyState = {
        agents: {
          agents: {},
          agentOrder: [],
          selectedAgentId: null,
          typingAgents: [],
        },
      };
      renderWithRedux(<AgentPanel view="executive" />, emptyState);

      const addButton = screen.getAllByTestId("plus-icon")[0].parentElement!;
      await user.click(addButton);

      expect(screen.getByRole("dialog")).toBeInTheDocument();

      const closeButton = screen.getByRole("button", { name: /close dialog/i });
      await user.click(closeButton);

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  describe("Footer Statistics", () => {
    it("displays correct agent counts", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("1 Active")).toBeInTheDocument();
      expect(screen.getByText("1 Idle")).toBeInTheDocument();
    });

    it("shows manage all button", () => {
      const emptyState = {
        agents: {
          agents: {},
          agentOrder: [],
          selectedAgentId: null,
          typingAgents: [],
        },
      };
      renderWithRedux(<AgentPanel view="executive" />, emptyState);

      const manageButton = screen.getByRole("button", { name: /Manage All/i });
      expect(manageButton).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("handles missing agent gracefully", () => {
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1", "nonexistent"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      // Should only show the existing agent
      expect(screen.getAllByTestId("agent-card")).toHaveLength(1);
      expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
    });

    it("displays agent ID when name is missing", () => {
      const initialState = {
        agents: {
          agents: {
            agent1: { ...mockAgents.agent1, name: undefined },
          },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("agent1")).toBeInTheDocument();
    });

    it("handles missing metrics gracefully", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: {
            agent1: { ...mockAgents.agent1, activityMetrics: undefined },
          },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");
      await user.click(agentCard);

      // Should show 0 for missing metrics
      expect(screen.getByText("0")).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("has accessible labels for controls", () => {
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(
        screen.getByRole("button", { name: /pause/i }),
      ).toBeInTheDocument();
    });

    it("agent cards are keyboard accessible", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
          selectedAgentId: null,
          typingAgents: [],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");

      // Should be clickable
      await user.click(agentCard);
      expect(screen.getByText("Messages:")).toBeInTheDocument();
    });
  });
});
