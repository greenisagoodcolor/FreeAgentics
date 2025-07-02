import React from "react";
import { render, screen } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import agentReducer from "@/store/slices/agentSlice";
import conversationReducer from "@/store/slices/conversationSlice";
import uiReducer from "@/store/slices/uiSlice";
import knowledgeReducer from "@/store/slices/knowledgeSlice";

// Ensure we're using the real component
jest.unmock("@/app/dashboard/components/panels/MetricsPanel");
jest.unmock("@/app/dashboard/components/panels/MetricsPanel/index");

// Import after unmocking
import MetricsPanel from "@/app/dashboard/components/panels/MetricsPanel";
import { useAppSelector } from "@/store/hooks";

// Get the mocked function
const mockUseAppSelector = useAppSelector as jest.MockedFunction<
  typeof useAppSelector
>;

// Mock Next.js router
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock Redux store hooks
jest.mock("@/store/hooks", () => ({
  useAppSelector: jest.fn(),
  useAppDispatch: () => jest.fn(),
}));

// Mock icons
jest.mock("lucide-react", () => ({
  Activity: ({ className }: any) => (
    <span className={className} data-testid="activity-icon">
      Activity
    </span>
  ),
  Users: ({ className }: any) => (
    <span className={className} data-testid="users-icon">
      Users
    </span>
  ),
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
  TrendingUp: ({ className }: any) => (
    <span className={className} data-testid="trending-up-icon">
      TrendingUp
    </span>
  ),
  Zap: ({ className }: any) => (
    <span className={className} data-testid="zap-icon">
      Zap
    </span>
  ),
  Presentation: ({ className }: any) => (
    <span className={className} data-testid="presentation-icon">
      Presentation
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
      analytics: (state = { metrics: {} }) => state,
      connection: (state = { isConnected: false, socket: null }) => state,
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

describe("MetricsPanel Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset to default mock implementation
    mockUseAppSelector.mockImplementation((selector) => {
      const defaultMockState = {
        agents: { agents: {} },
        conversations: { conversations: {} },
        analytics: { metrics: {} },
      };
      return selector(defaultMockState);
    });
  });

  describe("Component Rendering", () => {
    it("renders all metric cards with correct labels", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
      expect(screen.getByText("ACTIVE AGENTS")).toBeInTheDocument();
      expect(screen.getByText("TOTAL MESSAGES")).toBeInTheDocument();
      expect(screen.getByText("AVG RESPONSE")).toBeInTheDocument();
      expect(screen.getByText("KNOWLEDGE NODES")).toBeInTheDocument();
      expect(screen.getByText("SYSTEM HEALTH")).toBeInTheDocument();
    });

    it("renders all metric icons", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByTestId("users-icon")).toBeInTheDocument();
      expect(screen.getByTestId("activity-icon")).toBeInTheDocument();
      expect(screen.getByTestId("message-square-icon")).toBeInTheDocument();
      expect(screen.getByTestId("zap-icon")).toBeInTheDocument();
      expect(screen.getByTestId("brain-icon")).toBeInTheDocument();
      expect(screen.getByTestId("trending-up-icon")).toBeInTheDocument();
    });

    it("applies correct styling classes", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      // Find the outer container by going up multiple levels
      const metricsLabel = screen.getByText("TOTAL AGENTS");
      const outerContainer = metricsLabel.closest(
        '[class*="h-full bg-[var(--bg-primary)]"]',
      );
      expect(outerContainer).toHaveClass(
        "h-full",
        "bg-[var(--bg-primary)]",
        "p-4",
      );

      const grid = outerContainer?.querySelector(".grid");
      expect(grid).toHaveClass("grid", "grid-cols-7", "gap-4", "h-full");
    });

    it("renders metric cards with correct structure", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      // Find the metric card container by traversing up from the label
      const metricLabel = screen.getByText("TOTAL AGENTS");
      const metricCard = metricLabel.closest(
        '[class*="bg-[var(--bg-secondary)]"]',
      );
      expect(metricCard).toHaveClass(
        "bg-[var(--bg-secondary)]",
        "border",
        "border-[var(--bg-tertiary)]",
        "rounded-lg",
        "p-4",
        "flex",
        "flex-col",
        "justify-between",
      );
    });
  });

  describe("Metric Calculations", () => {
    it("calculates total agents correctly", () => {
      const mockAgents = {
        "agent-1": { id: "agent-1", name: "Agent 1", status: "active" },
        "agent-2": { id: "agent-2", name: "Agent 2", status: "inactive" },
        "agent-3": { id: "agent-3", name: "Agent 3", status: "active" },
      };

      // Mock the useAppSelector calls
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: mockAgents },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getByText("3")).toBeInTheDocument();
      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
    });

    it("calculates active agents correctly", () => {
      const mockAgents = {
        "agent-1": { id: "agent-1", name: "Agent 1", status: "active" },
        "agent-2": { id: "agent-2", name: "Agent 2", status: "inactive" },
        "agent-3": { id: "agent-3", name: "Agent 3", status: "active" },
      };

      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: mockAgents },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getByText("2")).toBeInTheDocument();
      expect(screen.getByText("67%")).toBeInTheDocument(); // 2/3 * 100 = 67%
    });

    it("calculates total messages correctly", () => {
      const stateWithConversations = {
        conversations: {
          conversations: {
            "conv-1": {
              id: "conv-1",
              messages: [
                { id: "msg-1", content: "Hello" },
                { id: "msg-2", content: "Hi there" },
              ],
            },
            "conv-2": {
              id: "conv-2",
              messages: [{ id: "msg-3", content: "How are you?" }],
            },
          },
        },
      };

      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: {} },
          conversations: stateWithConversations.conversations,
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getByText("3")).toBeInTheDocument();
    });

    it("displays average response time from analytics", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: {} },
          conversations: { conversations: {} },
          analytics: {
            metrics: {
              averageResponseTime: 250.7,
            },
          },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getByText("251ms")).toBeInTheDocument();
    });

    it("handles empty state gracefully", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: {
            agents: {},
            agentOrder: [],
            selectedAgentId: null,
            typingAgents: [],
          },
          conversations: {
            conversations: {},
            active: null,
          },
          analytics: {
            metrics: {},
          },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getAllByText("0")).toHaveLength(3); // Total agents, active agents, messages
      expect(screen.getByText("0ms")).toBeInTheDocument(); // Avg response time
    });

    it("handles missing conversations gracefully", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: {} },
          conversations: {
            conversations: {
              "conv-1": {
                id: "conv-1",
                // No messages property
              },
            },
          },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getAllByText("0")).toHaveLength(3); // Should handle missing messages
    });
  });

  describe("Static Metrics", () => {
    it("displays static knowledge nodes metric", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByText("1,247")).toBeInTheDocument();
      expect(screen.getByText("KNOWLEDGE NODES")).toBeInTheDocument();
      expect(screen.getByText("+89")).toBeInTheDocument();
    });

    it("displays static system health metric", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByText("99.9%")).toBeInTheDocument();
      expect(screen.getByText("SYSTEM HEALTH")).toBeInTheDocument();
      expect(screen.getByText("+0.1%")).toBeInTheDocument();
    });

    it("displays static change values for hardcoded metrics", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByText("+2")).toBeInTheDocument(); // Total agents change
      expect(screen.getByText("+156")).toBeInTheDocument(); // Total messages change
      expect(screen.getByText("-12ms")).toBeInTheDocument(); // Avg response change
    });
  });

  describe("Trend Indicators", () => {
    it("applies correct trend colors for up trends", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      const upTrends = screen.getAllByText(/^\+/); // All positive changes
      upTrends.forEach((trend) => {
        expect(trend).toHaveClass("text-green-400");
      });
    });

    it("applies correct trend colors for down trends", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      const downTrend = screen.getByText("-12ms");
      expect(downTrend).toHaveClass("text-red-400");
    });

    it("applies correct icon colors", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByTestId("users-icon")).toHaveClass("text-blue-400");
      expect(screen.getByTestId("activity-icon")).toHaveClass("text-green-400");
      expect(screen.getByTestId("message-square-icon")).toHaveClass(
        "text-purple-400",
      );
      expect(screen.getByTestId("zap-icon")).toHaveClass("text-yellow-400");
      expect(screen.getByTestId("brain-icon")).toHaveClass("text-indigo-400");
      expect(screen.getByTestId("trending-up-icon")).toHaveClass(
        "text-emerald-400",
      );
    });
  });

  describe("View Types", () => {
    it("accepts different view types", () => {
      const views = ["executive", "technical", "research", "minimal"];

      views.forEach((view) => {
        const { unmount } = renderWithRedux(<MetricsPanel view={view} />);
        expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
        unmount();
      });
    });

    it("maintains functionality across view types", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: {
            agents: {
              "agent-1": { id: "agent-1", status: "active" },
            },
          },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      const views = ["executive", "technical", "research", "minimal"];

      views.forEach((view) => {
        const { unmount } = render(<MetricsPanel view={view} />);
        expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
        expect(screen.getByText("100%")).toBeInTheDocument(); // Active percentage
        unmount();
      });
    });
  });

  describe("Redux Integration", () => {
    it("updates when agents state changes", () => {
      // First setup - initial state
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: {
            agents: {
              "agent-1": { id: "agent-1", status: "active" },
            },
          },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      const { rerender } = render(<MetricsPanel view="executive" />);

      // Verify initial state - check for TOTAL AGENTS label to be more specific
      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
      expect(screen.getByText("100%")).toBeInTheDocument();

      // Update the mock for the rerender
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: {
            agents: {
              "agent-1": { id: "agent-1", status: "active" },
              "agent-2": { id: "agent-2", status: "inactive" },
            },
          },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      rerender(<MetricsPanel view="executive" />);

      // Verify updated state - 1 active out of 2 total = 50%
      expect(screen.getByText("50%")).toBeInTheDocument();
    });

    it("handles null or undefined state gracefully", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: null,
          conversations: null,
          analytics: null,
        };
        return selector(mockState);
      });

      expect(() => {
        render(<MetricsPanel view="executive" />);
      }).not.toThrow();

      expect(screen.getAllByText("0")).toHaveLength(3); // Should default to 0 for multiple metrics
    });
  });

  describe("Accessibility", () => {
    it("has proper semantic structure", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      const metrics = screen.getAllByText(
        /AGENTS|MESSAGES|RESPONSE|NODES|HEALTH/,
      );
      expect(metrics).toHaveLength(6);

      metrics.forEach((metric) => {
        expect(metric).toBeInTheDocument();
      });
    });

    it("provides meaningful metric labels", () => {
      renderWithRedux(<MetricsPanel view="executive" />);

      const labels = [
        "TOTAL AGENTS",
        "ACTIVE AGENTS",
        "TOTAL MESSAGES",
        "AVG RESPONSE",
        "KNOWLEDGE NODES",
        "SYSTEM HEALTH",
      ];

      labels.forEach((label) => {
        expect(screen.getByText(label)).toBeInTheDocument();
      });
    });

    it("uses appropriate font styles for readability", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: {} },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      const metricValues = screen.getAllByText("0"); // Multiple metric values
      expect(metricValues[0]).toHaveClass(
        "text-2xl",
        "font-bold",
        "font-mono",
        "text-[var(--text-primary)]",
      );

      const metricLabel = screen.getByText("TOTAL AGENTS");
      expect(metricLabel).toHaveClass(
        "text-xs",
        "text-[var(--text-secondary)]",
        "font-mono",
      );
    });
  });

  describe("Performance", () => {
    it("renders efficiently with large datasets", () => {
      const largeAgents = {};
      for (let i = 0; i < 1000; i++) {
        largeAgents[`agent-${i}`] = {
          id: `agent-${i}`,
          status: i % 2 === 0 ? "active" : "inactive",
        };
      }

      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: largeAgents },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      const startTime = performance.now();
      render(<MetricsPanel view="executive" />);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should render in under 1 second
      expect(screen.getByText("1000")).toBeInTheDocument(); // Total agents
      expect(screen.getByText("50%")).toBeInTheDocument(); // Active percentage
    });

    it("handles rapid state updates efficiently", () => {
      // Initial setup
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: {} },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      const { rerender } = render(<MetricsPanel view="executive" />);

      for (let i = 0; i < 10; i++) {
        mockUseAppSelector.mockImplementation((selector) => {
          const mockState = {
            agents: {
              agents: {
                [`agent-${i}`]: { id: `agent-${i}`, status: "active" },
              },
            },
            conversations: { conversations: {} },
            analytics: { metrics: {} },
          };
          return selector(mockState);
        });

        rerender(<MetricsPanel view="executive" />);
      }

      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
      expect(screen.getByText("100%")).toBeInTheDocument();
    });
  });

  describe("Error Handling", () => {
    it("handles invalid agent data gracefully", () => {
      const stateWithInvalidData = {
        agents: {
          agents: {
            "agent-1": null,
            "agent-2": undefined,
            "agent-3": { id: "agent-3" }, // Missing status
          },
        },
      };

      expect(() => {
        renderWithRedux(
          <MetricsPanel view="executive" />,
          stateWithInvalidData,
        );
      }).not.toThrow();
    });

    it("handles invalid conversation data gracefully", () => {
      const stateWithInvalidConversations = {
        conversations: {
          conversations: {
            "conv-1": null,
            "conv-2": { id: "conv-2", messages: null },
            "conv-3": { id: "conv-3", messages: "invalid" },
          },
        },
      };

      expect(() => {
        renderWithRedux(
          <MetricsPanel view="executive" />,
          stateWithInvalidConversations,
        );
      }).not.toThrow();
    });

    it("handles missing props gracefully", () => {
      expect(() => {
        renderWithRedux(<MetricsPanel view={undefined as any} />);
      }).not.toThrow();
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts and unmounts without errors", () => {
      const { unmount } = renderWithRedux(<MetricsPanel view="executive" />);

      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();

      expect(() => unmount()).not.toThrow();
    });

    it("maintains state consistency across re-renders", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: {
            agents: {
              "agent-1": { id: "agent-1", status: "active" },
            },
          },
          conversations: { conversations: {} },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      const { rerender } = render(<MetricsPanel view="executive" />);

      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();

      rerender(<MetricsPanel view="technical" />);

      expect(screen.getByText("TOTAL AGENTS")).toBeInTheDocument();
    });
  });

  describe("Localization", () => {
    it("formats large numbers correctly", () => {
      mockUseAppSelector.mockImplementation((selector) => {
        const mockState = {
          agents: { agents: {} },
          conversations: {
            conversations: {
              "conv-1": {
                id: "conv-1",
                messages: new Array(12000).fill({ id: "msg", content: "test" }),
              },
            },
          },
          analytics: { metrics: {} },
        };
        return selector(mockState);
      });

      render(<MetricsPanel view="executive" />);

      expect(screen.getByText("12,000")).toBeInTheDocument();
    });

    it("handles zero division correctly", () => {
      const stateWithNoAgents = {
        agents: {
          agents: {},
        },
      };

      renderWithRedux(<MetricsPanel view="executive" />, stateWithNoAgents);

      expect(screen.getByText("0%")).toBeInTheDocument(); // Should not show NaN%
    });
  });
});
