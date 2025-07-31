import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { KnowledgeGraphView } from "./KnowledgeGraphView";
import { useKnowledgeGraph } from "@/hooks/use-knowledge-graph";

// Mock the knowledge graph hook
jest.mock("@/hooks/use-knowledge-graph");

// Mock D3 to avoid issues in test environment
jest.mock("d3", () => ({
  select: jest.fn(() => ({
    selectAll: jest.fn(() => ({
      remove: jest.fn(),
    })),
    attr: jest.fn(() => ({
      attr: jest.fn(),
    })),
    call: jest.fn(),
    append: jest.fn(() => ({
      selectAll: jest.fn(() => ({
        data: jest.fn(() => ({
          enter: jest.fn(() => ({
            append: jest.fn(() => ({
              attr: jest.fn(() => ({
                attr: jest.fn(() => ({
                  attr: jest.fn(),
                })),
              })),
            })),
          })),
        })),
      })),
    })),
  })),
  zoom: jest.fn(() => ({
    scaleExtent: jest.fn(() => ({
      on: jest.fn(),
    })),
    transform: jest.fn(),
    scaleBy: jest.fn(),
  })),
  forceSimulation: jest.fn(() => ({
    force: jest.fn(() => ({
      force: jest.fn(() => ({
        force: jest.fn(() => ({
          force: jest.fn(),
        })),
      })),
    })),
    on: jest.fn(),
    stop: jest.fn(),
    alpha: jest.fn(() => ({
      restart: jest.fn(),
    })),
    alphaTarget: jest.fn(() => ({
      restart: jest.fn(),
    })),
  })),
  forceLink: jest.fn(() => ({
    id: jest.fn(() => ({
      distance: jest.fn(),
    })),
  })),
  forceManyBody: jest.fn(() => ({
    strength: jest.fn(),
  })),
  forceCenter: jest.fn(),
  forceCollide: jest.fn(() => ({
    radius: jest.fn(),
  })),
  drag: jest.fn(() => ({
    on: jest.fn(() => ({
      on: jest.fn(() => ({
        on: jest.fn(),
      })),
    })),
  })),
  zoomIdentity: {},
}));

describe("KnowledgeGraphView", () => {
  const mockClearGraph = jest.fn();

  const mockNodes = [
    { id: "1", label: "Agent 1", type: "agent" as const, x: 100, y: 100 },
    { id: "2", label: "Belief 1", type: "belief" as const, x: 200, y: 200 },
  ];

  const mockEdges = [{ source: "1", target: "2" }];

  beforeEach(() => {
    jest.clearAllMocks();
    (useKnowledgeGraph as jest.Mock).mockReturnValue({
      nodes: mockNodes,
      edges: mockEdges,
      isLoading: false,
      error: null,
      clearGraph: mockClearGraph,
    });
  });

  it("should render loading state", () => {
    (useKnowledgeGraph as jest.Mock).mockReturnValue({
      nodes: [],
      edges: [],
      isLoading: true,
      error: null,
      clearGraph: mockClearGraph,
    });

    render(<KnowledgeGraphView />);
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("should render error state", () => {
    (useKnowledgeGraph as jest.Mock).mockReturnValue({
      nodes: [],
      edges: [],
      isLoading: false,
      error: new Error("Failed to load graph"),
      clearGraph: mockClearGraph,
    });

    render(<KnowledgeGraphView />);
    expect(screen.getByText("Failed to load graph")).toBeInTheDocument();
  });

  it("should render nodes and display node count", () => {
    render(<KnowledgeGraphView />);
    expect(screen.getByText("2 nodes, 1 edges")).toBeInTheDocument();
  });

  it("should open a details side-sheet when clicking a node", () => {
    render(<KnowledgeGraphView />);

    // Find the graph container
    const graphContainer = screen.getByTestId("graph-container");
    expect(graphContainer).toBeInTheDocument();

    // Since D3 creates SVG elements dynamically, we need to simulate a node click
    // In a real implementation, this would trigger opening a side sheet
    // For now, this test will fail as the feature isn't implemented
    const nodeDetailsSheet = screen.queryByTestId("node-details-sheet");
    expect(nodeDetailsSheet).not.toBeInTheDocument();

    // Simulate clicking on a node (this would be done via D3 in the actual implementation)
    // This test expects a side sheet to appear after clicking
    fireEvent.click(graphContainer);

    // This assertion will fail until we implement the feature
    expect(screen.getByTestId("node-details-sheet")).toBeInTheDocument();
    expect(screen.getByText("Node Details")).toBeInTheDocument();
  });
});
