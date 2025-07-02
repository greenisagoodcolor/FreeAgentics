import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import agentReducer from "@/store/slices/agentSlice";

// Unmock the AgentPanel component
jest.unmock("@/app/dashboard/components/panels/AgentPanel");

import AgentPanel from "@/app/dashboard/components/panels/AgentPanel";

// Mock the UI components
jest.mock("@/components/ui/button", () => ({
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

jest.mock("@/components/ui/badge", () => ({
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}));

jest.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}));

jest.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children, open }: any) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: any) => <div>{children}</div>,
  DialogHeader: ({ children }: any) => <div>{children}</div>,
  DialogTitle: ({ children }: any) => <h2>{children}</h2>,
}));

jest.mock("@/components/dashboard/AgentTemplateSelector", () => {
  return function MockAgentTemplateSelector() {
    return (
      <div data-testid="agent-template-selector">Agent Template Selector</div>
    );
  };
});

// Mock icons
jest.mock("lucide-react", () => ({
  Users: () => <span data-testid="users-icon">Users</span>,
  Plus: () => <span data-testid="plus-icon">Plus</span>,
  Settings: () => <span data-testid="settings-icon">Settings</span>,
  Activity: () => <span data-testid="activity-icon">Activity</span>,
  Pause: () => <span data-testid="pause-icon">Pause</span>,
  Play: () => <span data-testid="play-icon">Play</span>,
}));

// Helper function to create a mock store
const createMockStore = (initialState: any = {}) => {
  return configureStore({
    reducer: agentReducer,
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

describe("AgentPanel Component", () => {
  const mockAgents = {
    agent1: {
      name: "Agent Alpha",
      status: "active",
      templateId: "explorer",
      biography: "An exploration agent focused on discovery",
      activityMetrics: { messagesCount: 42 },
      inConversation: true,
    },
    agent2: {
      name: "Agent Beta",
      status: "idle",
      templateId: "guardian",
      biography: "A guardian agent protecting system integrity",
      activityMetrics: { messagesCount: 15 },
      inConversation: false,
    },
    agent3: {
      name: "Agent Gamma",
      status: "error",
      templateId: "analyst",
      activityMetrics: { messagesCount: 0 },
      inConversation: false,
    },
  };

  describe("Rendering", () => {
    it("renders with no agents", () => {
      renderWithRedux(<AgentPanel view="executive" />);

      expect(screen.getByText("Agent Management")).toBeInTheDocument();
      expect(screen.getByText("0 Total")).toBeInTheDocument();
      expect(screen.getByText("No agents configured")).toBeInTheDocument();
      expect(screen.getByText("Create Agent")).toBeInTheDocument();
    });

    it("renders with agents", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("3 Total")).toBeInTheDocument();
      expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
      expect(screen.getByText("Agent Beta")).toBeInTheDocument();
      expect(screen.getByText("Agent Gamma")).toBeInTheDocument();
    });

    it("displays agent status correctly", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("ACTIVE")).toBeInTheDocument();
      expect(screen.getByText("IDLE")).toBeInTheDocument();
      expect(screen.getByText("ERROR")).toBeInTheDocument();
    });

    it("shows agent biographies", () => {
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(
        screen.getByText(/exploration agent focused on discovery/),
      ).toBeInTheDocument();
    });

    it("displays footer statistics", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("1 Active")).toBeInTheDocument();
      expect(screen.getByText("1 Idle")).toBeInTheDocument();
    });
  });

  describe("Interactions", () => {
    it("selects and deselects agents on click", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");

      // Initially not selected
      expect(agentCard).not.toHaveClass("border-[var(--accent-primary)]");

      // Select agent
      await user.click(agentCard);
      expect(agentCard).toHaveClass("border-[var(--accent-primary)]");

      // Deselect agent
      await user.click(agentCard);
      expect(agentCard).not.toHaveClass("border-[var(--accent-primary)]");
    });

    it("shows expanded details when agent is selected", async () => {
      const user = userEvent.setup();
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");

      // Details not visible initially
      expect(screen.queryByText("Messages:")).not.toBeInTheDocument();

      // Select agent
      await user.click(agentCard);

      // Details now visible
      expect(screen.getByText("Messages:")).toBeInTheDocument();
      expect(screen.getByText("42")).toBeInTheDocument();
      expect(screen.getByText("Active:")).toBeInTheDocument();
      expect(screen.getByText("Yes")).toBeInTheDocument();
      expect(screen.getByText("Configure")).toBeInTheDocument();
      expect(screen.getByText("Logs")).toBeInTheDocument();
      expect(screen.getByText("Reset")).toBeInTheDocument();
    });

    it("opens template selector modal when clicking add button", async () => {
      const user = userEvent.setup();
      renderWithRedux(<AgentPanel view="executive" />);

      const addButton = screen.getAllByTestId("plus-icon")[0].parentElement!;

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

      await user.click(addButton);

      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByText("Create New Agent")).toBeInTheDocument();
      expect(screen.getByTestId("agent-template-selector")).toBeInTheDocument();
    });

    it("opens template selector from empty state", async () => {
      const user = userEvent.setup();
      renderWithRedux(<AgentPanel view="executive" />);

      const createButton = screen.getByText("Create Agent");

      await user.click(createButton);

      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByText("Create New Agent")).toBeInTheDocument();
    });

    it("toggles play/pause button based on agent status", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      // Active agent shows pause icon
      const agentCards = screen.getAllByTestId("agent-card");
      const activeCard = agentCards[0];
      expect(
        activeCard.querySelector('[data-testid="pause-icon"]'),
      ).toBeInTheDocument();

      // Idle agent shows play icon
      const idleCard = agentCards[1];
      expect(
        idleCard.querySelector('[data-testid="play-icon"]'),
      ).toBeInTheDocument();
    });
  });

  describe("Agent Status", () => {
    it("shows correct status colors", () => {
      const initialState = {
        agents: {
          agents: mockAgents,
          agentOrder: ["agent1", "agent2", "agent3"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCards = screen.getAllByTestId("agent-card");

      // Check status indicators
      expect(agentCards[0].querySelector(".bg-green-500")).toBeInTheDocument(); // active
      expect(agentCards[1].querySelector(".bg-yellow-500")).toBeInTheDocument(); // idle
      expect(agentCards[2].querySelector(".bg-red-500")).toBeInTheDocument(); // error
    });

    it("handles unknown status gracefully", () => {
      const initialState = {
        agents: {
          agents: {
            agent1: { ...mockAgents.agent1, status: "unknown" },
          },
          agentOrder: ["agent1"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
      expect(
        screen.getByTestId("agent-card").querySelector(".bg-gray-500"),
      ).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("handles missing agent data gracefully", () => {
      const initialState = {
        agents: {
          agents: { agent1: mockAgents.agent1 },
          agentOrder: ["agent1", "missingAgent"], // Reference to non-existent agent
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      // Should only render the existing agent
      expect(screen.getByText("Agent Alpha")).toBeInTheDocument();
      expect(screen.getAllByTestId("agent-card")).toHaveLength(1);
    });

    it("handles agents without names", () => {
      const initialState = {
        agents: {
          agents: {
            agent1: { ...mockAgents.agent1, name: undefined },
          },
          agentOrder: ["agent1"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      // Should display agent ID when name is missing
      expect(screen.getByText("agent1")).toBeInTheDocument();
    });

    it("handles missing activity metrics", () => {
      const initialState = {
        agents: {
          agents: {
            agent1: { ...mockAgents.agent1, activityMetrics: undefined },
          },
          agentOrder: ["agent1"],
        },
      };

      renderWithRedux(<AgentPanel view="executive" />, initialState);

      const agentCard = screen.getByTestId("agent-card");
      fireEvent.click(agentCard);

      // Should show 0 when metrics are missing
      expect(screen.getByText("0")).toBeInTheDocument();
    });
  });

  describe("Props", () => {
    it("accepts different view types", () => {
      const views: Array<"executive" | "technical" | "research" | "minimal"> = [
        "executive",
        "technical",
        "research",
        "minimal",
      ];

      views.forEach((view) => {
        const { unmount } = renderWithRedux(<AgentPanel view={view} />);
        expect(screen.getByText("Agent Management")).toBeInTheDocument();
        unmount();
      });
    });
  });

  describe("Performance", () => {
    it("handles large number of agents", () => {
      const manyAgents: any = {};
      const agentOrder: string[] = [];

      // Create 100 agents
      for (let i = 0; i < 100; i++) {
        const id = `agent${i}`;
        manyAgents[id] = {
          name: `Agent ${i}`,
          status: i % 3 === 0 ? "active" : i % 3 === 1 ? "idle" : "error",
          templateId: "test",
        };
        agentOrder.push(id);
      }

      const initialState = {
        agents: {
          agents: manyAgents,
          agentOrder,
        },
      };

      const { container } = renderWithRedux(
        <AgentPanel view="executive" />,
        initialState,
      );

      expect(screen.getByText("100 Total")).toBeInTheDocument();
      expect(
        container.querySelectorAll('[data-testid="agent-card"]'),
      ).toHaveLength(100);
    });
  });
});
