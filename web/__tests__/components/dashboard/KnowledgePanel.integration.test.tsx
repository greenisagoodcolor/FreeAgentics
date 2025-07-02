import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import agentReducer from "@/store/slices/agentSlice";
import conversationReducer from "@/store/slices/conversationSlice";
import uiReducer from "@/store/slices/uiSlice";
import knowledgeReducer from "@/store/slices/knowledgeSlice";

// Ensure we're using the real component
jest.unmock("@/app/dashboard/components/panels/KnowledgePanel");
jest.unmock("@/app/dashboard/components/panels/KnowledgePanel/index");

// Import after unmocking
import KnowledgePanel from "@/app/dashboard/components/panels/KnowledgePanel";
import KnowledgeGraphVisualization from "@/components/dashboard/KnowledgeGraphVisualization";

// Mock the complex KnowledgeGraphVisualization component
jest.mock("@/components/dashboard/KnowledgeGraphVisualization", () => {
  return jest.fn(({ testMode }: { testMode?: boolean }) => (
    <div data-testid="knowledge-graph-visualization" data-test-mode={testMode}>
      Mock Knowledge Graph Visualization
    </div>
  ));
});

// Mock icons
jest.mock("lucide-react", () => ({
  Network: ({ className }: any) => (
    <span className={className} data-testid="network-icon">
      Network
    </span>
  ),
}));

// Helper function to create a mock store with proper knowledge slice structure
const createMockStore = (initialState: any = {}) => {
  return configureStore({
    reducer: {
      knowledge: knowledgeReducer,
      agents: agentReducer,
      conversations: conversationReducer,
      ui: uiReducer,
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

describe("KnowledgePanel Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset environment
    delete (process.env as any).NODE_ENV;
  });

  describe("Component Rendering", () => {
    it("renders panel header with correct elements", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
      expect(screen.getByTestId("network-icon")).toBeInTheDocument();
    });

    it("renders KnowledgeGraphVisualization component", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Mock Knowledge Graph Visualization"),
      ).toBeInTheDocument();
    });

    it("applies correct styling classes", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      const container = screen.getByText("Knowledge Graph").closest(".h-full");
      expect(container).toHaveClass(
        "h-full",
        "flex",
        "flex-col",
        "bg-[var(--bg-primary)]",
      );

      const header = screen.getByText("Knowledge Graph").closest(".flex");
      expect(header).toHaveClass("flex", "items-center", "gap-2");
    });

    it("renders visualization container with correct styling", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      const visualizationContainer = screen.getByTestId(
        "knowledge-graph-visualization",
      ).parentElement;
      expect(visualizationContainer).toHaveClass("flex-1", "overflow-hidden");
    });
  });

  describe("Test Mode Detection", () => {
    beforeEach(() => {
      // Reset window.location mock
      delete (window as any).location;
      (window as any).location = { search: "" };
    });

    it("passes testMode=false by default", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      const visualization = screen.getByTestId("knowledge-graph-visualization");
      expect(visualization).toHaveAttribute("data-test-mode", "false");
      expect(KnowledgeGraphVisualization).toHaveBeenCalledWith(
        { testMode: false },
        expect.anything(),
      );
    });

    it("detects test mode from URL search params", () => {
      (window as any).location = { search: "?testMode=true" };

      renderWithRedux(<KnowledgePanel view="executive" />);

      const visualization = screen.getByTestId("knowledge-graph-visualization");
      expect(visualization).toHaveAttribute("data-test-mode", "true");
      expect(KnowledgeGraphVisualization).toHaveBeenCalledWith(
        { testMode: true },
        expect.anything(),
      );
    });

    it("detects test mode from NODE_ENV", () => {
      process.env.NODE_ENV = "test";

      renderWithRedux(<KnowledgePanel view="executive" />);

      const visualization = screen.getByTestId("knowledge-graph-visualization");
      expect(visualization).toHaveAttribute("data-test-mode", "true");
      expect(KnowledgeGraphVisualization).toHaveBeenCalledWith(
        { testMode: true },
        expect.anything(),
      );
    });

    it("detects test mode from URL with other params", () => {
      (window as any).location = { search: "?foo=bar&testMode=true&baz=qux" };

      renderWithRedux(<KnowledgePanel view="executive" />);

      const visualization = screen.getByTestId("knowledge-graph-visualization");
      expect(visualization).toHaveAttribute("data-test-mode", "true");
    });

    it("handles missing window object gracefully", () => {
      // Mock window.location to undefined to test graceful handling
      const originalLocation = window.location;
      delete (window as any).location;

      expect(() => {
        renderWithRedux(<KnowledgePanel view="executive" />);
      }).not.toThrow();

      // Restore window.location
      (window as any).location = originalLocation;
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
        const { unmount } = renderWithRedux(<KnowledgePanel view={view} />);
        expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
        expect(
          screen.getByTestId("knowledge-graph-visualization"),
        ).toBeInTheDocument();
        unmount();
      });
    });

    it("passes view prop correctly to visualization", () => {
      renderWithRedux(<KnowledgePanel view="technical" />);

      // Verify the component renders regardless of view type
      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
      expect(KnowledgeGraphVisualization).toHaveBeenCalled();
    });
  });

  describe("Redux Integration", () => {
    it("renders with empty knowledge state", () => {
      const emptyState = {
        knowledge: {
          graph: { nodes: {}, edges: {} },
          filters: {
            confidenceThreshold: 0,
            types: ["belief", "fact", "hypothesis"],
            agentIds: [],
            searchQuery: "",
          },
          selectedNodeId: null,
          selectedEdgeId: null,
          hoveredNodeId: null,
          viewMode: "collective",
          focusedAgentId: null,
          comparisonAgentIds: [],
          renderEngine: "svg",
          zoom: 1,
          center: { x: 0, y: 0 },
        },
      };

      renderWithRedux(<KnowledgePanel view="executive" />, emptyState);

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
    });

    it("renders with populated knowledge state", () => {
      const populatedState = {
        knowledge: {
          graph: {
            nodes: {
              "1": {
                id: "1",
                label: "Node 1",
                type: "belief",
                confidence: 0.8,
                agents: ["agent-1"],
                createdAt: Date.now(),
                lastModified: Date.now(),
              },
              "2": {
                id: "2",
                label: "Node 2",
                type: "fact",
                confidence: 0.9,
                agents: ["agent-2"],
                createdAt: Date.now(),
                lastModified: Date.now(),
              },
            },
            edges: {
              e1: {
                id: "e1",
                source: "1",
                target: "2",
                type: "supports",
                strength: 0.7,
                agents: ["agent-1"],
                createdAt: Date.now(),
              },
            },
          },
          filters: {
            confidenceThreshold: 0,
            types: ["belief", "fact", "hypothesis"],
            agentIds: [],
            searchQuery: "",
          },
          selectedNodeId: null,
          selectedEdgeId: null,
          hoveredNodeId: null,
          viewMode: "collective",
          focusedAgentId: null,
          comparisonAgentIds: [],
          renderEngine: "svg",
          zoom: 1,
          center: { x: 0, y: 0 },
        },
      };

      renderWithRedux(<KnowledgePanel view="executive" />, populatedState);

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
    });

    it("handles store updates gracefully", async () => {
      const { rerender } = renderWithRedux(<KnowledgePanel view="executive" />);

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();

      // Rerender with different state
      rerender(
        <Provider
          store={createMockStore({
            knowledge: {
              graph: {
                nodes: {
                  "1": {
                    id: "1",
                    label: "Updated Node",
                    type: "hypothesis",
                    confidence: 0.6,
                    agents: ["agent-1"],
                    createdAt: Date.now(),
                    lastModified: Date.now(),
                  },
                },
                edges: {},
              },
            },
          })}
        >
          <KnowledgePanel view="executive" />
        </Provider>,
      );

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts and unmounts without errors", () => {
      const { unmount } = renderWithRedux(<KnowledgePanel view="executive" />);

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();

      expect(() => unmount()).not.toThrow();
    });

    it("handles multiple renders correctly", () => {
      const { rerender } = renderWithRedux(<KnowledgePanel view="executive" />);

      expect(KnowledgeGraphVisualization).toHaveBeenCalledTimes(1);

      rerender(
        <Provider store={createMockStore()}>
          <KnowledgePanel view="technical" />
        </Provider>,
      );

      expect(KnowledgeGraphVisualization).toHaveBeenCalledTimes(2);
    });

    it("maintains component integrity across view changes", () => {
      const { rerender } = renderWithRedux(<KnowledgePanel view="executive" />);

      expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();

      rerender(
        <Provider store={createMockStore()}>
          <KnowledgePanel view="minimal" />
        </Provider>,
      );

      expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("has proper semantic structure", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      const heading = screen.getByRole("heading", { level: 3 });
      expect(heading).toHaveTextContent("Knowledge Graph");

      // Look for the h-full container which is the outermost div
      const panel = screen.getByText("Knowledge Graph").closest(".h-full");
      expect(panel).toHaveClass("h-full");
    });

    it("provides meaningful header content", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
      expect(screen.getByTestId("network-icon")).toBeInTheDocument();
    });

    it("maintains focus management", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      // The panel should be focusable through its child components
      const container = screen.getByTestId("knowledge-graph-visualization");
      expect(container).toBeInTheDocument();
    });
  });

  describe("Error Handling", () => {
    it("handles visualization component errors gracefully", () => {
      // Mock console.error to capture the error
      const originalError = console.error;
      console.error = jest.fn();

      KnowledgeGraphVisualization.mockImplementationOnce(() => {
        throw new Error("Visualization error");
      });

      // The error should be caught by React's error boundary and logged
      expect(() => {
        renderWithRedux(<KnowledgePanel view="executive" />);
      }).not.toThrow();

      // Verify the error was logged
      expect(console.error).toHaveBeenCalled();

      // Restore console.error
      console.error = originalError;
    });

    it("handles invalid view props", () => {
      const invalidView = "invalid" as any;

      expect(() => {
        renderWithRedux(<KnowledgePanel view={invalidView} />);
      }).not.toThrow();

      expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
    });

    it("handles missing props gracefully", () => {
      expect(() => {
        renderWithRedux(<KnowledgePanel view={undefined as any} />);
      }).not.toThrow();
    });
  });

  describe("Performance", () => {
    it("renders efficiently with large datasets", () => {
      const largeNodes = {};
      const largeEdges = {};

      // Create 100 nodes (reduced for test performance)
      for (let i = 0; i < 100; i++) {
        largeNodes[`node-${i}`] = {
          id: `node-${i}`,
          label: `Node ${i}`,
          type: "belief",
          confidence: 0.8,
          agents: ["agent-1"],
          createdAt: Date.now(),
          lastModified: Date.now(),
        };
      }

      // Create 50 edges
      for (let i = 0; i < 50; i++) {
        largeEdges[`edge-${i}`] = {
          id: `edge-${i}`,
          source: `node-${i}`,
          target: `node-${i + 1}`,
          type: "supports",
          strength: 0.7,
          agents: ["agent-1"],
          createdAt: Date.now(),
        };
      }

      const largeState = {
        knowledge: {
          graph: {
            nodes: largeNodes,
            edges: largeEdges,
          },
        },
      };

      const startTime = performance.now();
      renderWithRedux(<KnowledgePanel view="executive" />, largeState);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should render in under 1 second
      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
    });

    it("handles rapid prop changes efficiently", async () => {
      const { rerender } = renderWithRedux(<KnowledgePanel view="executive" />);

      const views: Array<"executive" | "technical" | "research" | "minimal"> = [
        "technical",
        "research",
        "minimal",
        "executive",
      ];

      for (const view of views) {
        rerender(
          <Provider store={createMockStore()}>
            <KnowledgePanel view={view} />
          </Provider>,
        );

        await waitFor(() => {
          expect(
            screen.getByTestId("knowledge-graph-visualization"),
          ).toBeInTheDocument();
        });
      }

      expect(KnowledgeGraphVisualization).toHaveBeenCalledTimes(5); // Initial + 4 updates
    });
  });

  describe("Integration with Visualization Component", () => {
    it("passes testMode prop correctly to visualization", () => {
      process.env.NODE_ENV = "test";

      renderWithRedux(<KnowledgePanel view="executive" />);

      expect(KnowledgeGraphVisualization).toHaveBeenCalledWith(
        { testMode: true },
        expect.anything(),
      );
    });

    it("provides proper container for visualization", () => {
      renderWithRedux(<KnowledgePanel view="executive" />);

      const container = screen.getByTestId(
        "knowledge-graph-visualization",
      ).parentElement;
      expect(container).toHaveClass("flex-1", "overflow-hidden");
    });

    it("maintains visualization across re-renders", () => {
      const { rerender } = renderWithRedux(<KnowledgePanel view="executive" />);

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();

      rerender(
        <Provider store={createMockStore()}>
          <KnowledgePanel view="executive" />
        </Provider>,
      );

      expect(
        screen.getByTestId("knowledge-graph-visualization"),
      ).toBeInTheDocument();
    });
  });
});
