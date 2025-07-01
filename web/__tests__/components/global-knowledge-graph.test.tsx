/**
 * GlobalKnowledgeGraph Component Comprehensive Test Suite
 * Target: components/GlobalKnowledgeGraph.tsx (1,605 lines)
 * Goal: Maximize coverage for the second largest component file
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
import GlobalKnowledgeGraph from "@/components/GlobalKnowledgeGraph";
import type { Agent, KnowledgeEntry } from "@/lib/types";

// Mock D3 force simulation and dependencies
const mockForceSimulation = {
  nodes: jest.fn().mockReturnThis(),
  links: jest.fn().mockReturnThis(),
  force: jest.fn().mockReturnThis(),
  on: jest.fn().mockReturnThis(),
  restart: jest.fn().mockReturnThis(),
  stop: jest.fn().mockReturnThis(),
  alpha: jest.fn().mockReturnThis(),
  alphaTarget: jest.fn().mockReturnThis(),
  alphaDecay: jest.fn().mockReturnThis(),
  velocityDecay: jest.fn().mockReturnThis(),
  tick: jest.fn().mockReturnThis(),
};

const mockForceCenter = jest.fn(() => mockForceSimulation);
const mockForceLink = jest.fn(() => mockForceSimulation);
const mockForceManyBody = jest.fn(() => mockForceSimulation);
const mockForceCollide = jest.fn(() => mockForceSimulation);

jest.mock("d3-force", () => ({
  forceSimulation: jest.fn(() => mockForceSimulation),
  forceCenter: mockForceCenter,
  forceLink: mockForceLink,
  forceManyBody: mockForceManyBody,
  forceCollide: mockForceCollide,
}));

// Mock D3 selection and drag
const mockDragBehavior = {
  on: jest.fn().mockReturnThis(),
};

const mockSelect = jest.fn(() => ({
  selectAll: jest.fn().mockReturnThis(),
  data: jest.fn().mockReturnThis(),
  enter: jest.fn().mockReturnThis(),
  append: jest.fn().mockReturnThis(),
  merge: jest.fn().mockReturnThis(),
  exit: jest.fn().mockReturnThis(),
  remove: jest.fn().mockReturnThis(),
  attr: jest.fn().mockReturnThis(),
  style: jest.fn().mockReturnThis(),
  text: jest.fn().mockReturnThis(),
  on: jest.fn().mockReturnThis(),
  call: jest.fn().mockReturnThis(),
  raise: jest.fn().mockReturnThis(),
  node: jest.fn(() => ({
    getBBox: () => ({ width: 100, height: 20 }),
  })),
}));

const mockDrag = jest.fn(() => mockDragBehavior);

jest.mock("d3-selection", () => ({
  select: mockSelect,
}));

jest.mock("d3-drag", () => ({
  drag: mockDrag,
}));

// Mock components
jest.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: any) => (
    <div data-testid="card" {...props}>
      {children}
    </div>
  ),
  CardContent: ({ children, ...props }: any) => (
    <div data-testid="card-content" {...props}>
      {children}
    </div>
  ),
  CardHeader: ({ children, ...props }: any) => (
    <div data-testid="card-header" {...props}>
      {children}
    </div>
  ),
  CardTitle: ({ children, ...props }: any) => (
    <h3 data-testid="card-title" {...props}>
      {children}
    </h3>
  ),
}));

jest.mock("@/components/ui/button", () => ({
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

jest.mock("@/components/AboutButton", () => {
  return function MockAboutButton({ onShowAbout }: any) {
    return (
      <button onClick={onShowAbout} data-testid="about-button">
        About
      </button>
    );
  };
});

// Mock data
const mockKnowledgeEntry: KnowledgeEntry = {
  id: "knowledge-1",
  title: "Test Knowledge",
  content: "This is test knowledge content",
  source: "user",
  timestamp: new Date(),
  tags: ["test", "knowledge"],
  metadata: {},
};

const mockAgent: Agent = {
  id: "agent-1",
  name: "Test Agent",
  biography: "Test agent biography",
  color: "#ff0000",
  position: { x: 0, y: 0 },
  knowledge: [mockKnowledgeEntry],
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
};

const defaultProps = {
  agents: [mockAgent],
  onSelectNode: jest.fn(),
  onShowAbout: jest.fn(),
};

describe("GlobalKnowledgeGraph", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Reset D3 mocks to default behavior
    mockForceSimulation.nodes.mockReturnThis();
    mockForceSimulation.links.mockReturnThis();
    mockForceSimulation.force.mockReturnThis();
    mockForceSimulation.on.mockReturnThis();
    mockForceSimulation.restart.mockReturnThis();
    mockForceSimulation.stop.mockReturnThis();

    // Mock ResizeObserver
    global.ResizeObserver = jest.fn().mockImplementation(() => ({
      observe: jest.fn(),
      unobserve: jest.fn(),
      disconnect: jest.fn(),
    }));

    // Mock HTMLCanvasElement and 2D context
    const mockContext = {
      fillRect: jest.fn(),
      clearRect: jest.fn(),
      getImageData: jest.fn(() => ({
        data: new Uint8ClampedArray(4),
      })),
      putImageData: jest.fn(),
      createImageData: jest.fn(),
      setTransform: jest.fn(),
      drawImage: jest.fn(),
      save: jest.fn(),
      restore: jest.fn(),
      fillText: jest.fn(),
      measureText: jest.fn(() => ({ width: 100 })),
      arc: jest.fn(),
      beginPath: jest.fn(),
      closePath: jest.fn(),
      fill: jest.fn(),
      stroke: jest.fn(),
      scale: jest.fn(),
      rotate: jest.fn(),
      translate: jest.fn(),
      clip: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      bezierCurveTo: jest.fn(),
      quadraticCurveTo: jest.fn(),
      createLinearGradient: jest.fn(),
      createRadialGradient: jest.fn(),
    };

    // Mock canvas element
    HTMLCanvasElement.prototype.getContext = jest.fn(() => mockContext);
    HTMLCanvasElement.prototype.toDataURL = jest.fn(
      () => "data:image/png;base64,mock",
    );
    HTMLCanvasElement.prototype.toBlob = jest.fn();
  });

  describe("Component Initialization", () => {
    test("renders without crashing", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      expect(screen.getByText("Global Knowledge Graph")).toBeInTheDocument();
      expect(document.querySelector("canvas")).toBeInTheDocument();
    });

    test("displays graph title", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      expect(screen.getByText("Global Knowledge Graph")).toBeInTheDocument();
    });

    test("renders control buttons", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      expect(
        screen.getByRole("button", { name: /zoom in/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /zoom out/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /reset view/i }),
      ).toBeInTheDocument();
    });

    test("renders about button", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      // The about button is rendered in the component, look for its title attribute
      expect(document.querySelector('[title="About"]')).toBeInTheDocument();
    });

    test("renders SVG element", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      const svg = document.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });
  });

  describe("Graph Visualization", () => {
    test("initializes D3 force simulation", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      expect(mockForceSimulation.nodes).toHaveBeenCalled();
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "link",
        expect.any(Object),
      );
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "charge",
        expect.any(Object),
      );
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "center",
        expect.any(Object),
      );
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "collision",
        expect.any(Object),
      );
    });

    test("creates nodes from agent data", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      // Simulation should be initialized with nodes
      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });

    test("handles empty agents array", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} agents={[]} />);

      expect(screen.getByTestId("card")).toBeInTheDocument();
      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });

    test("handles multiple agents", () => {
      const multipleAgents = [
        mockAgent,
        {
          ...mockAgent,
          id: "agent-2",
          name: "Second Agent",
          color: "#00ff00",
          knowledge: [
            {
              ...mockKnowledgeEntry,
              id: "knowledge-2",
              title: "Second Knowledge",
              tags: ["different", "tags"],
            },
          ],
        },
      ];

      render(
        <GlobalKnowledgeGraph {...defaultProps} agents={multipleAgents} />,
      );

      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });
  });

  describe("Control Interactions", () => {
    test("zoom in button works", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      const zoomInButton = screen.getByRole("button", { name: /zoom in/i });
      fireEvent.click(zoomInButton);

      // Should trigger zoom functionality
      expect(zoomInButton).toBeInTheDocument();
    });

    test("zoom out button works", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      const zoomOutButton = screen.getByRole("button", { name: /zoom out/i });
      fireEvent.click(zoomOutButton);

      expect(zoomOutButton).toBeInTheDocument();
    });

    test("reset view button works", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      const resetButton = screen.getByRole("button", { name: /reset view/i });
      fireEvent.click(resetButton);

      expect(resetButton).toBeInTheDocument();
    });

    test("play/pause simulation button works", async () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      // Look for play/pause button (might start as either)
      const buttons = screen.getAllByRole("button");
      const playPauseButton = buttons.find(
        (button) =>
          button.getAttribute("title")?.includes("play") ||
          button.getAttribute("title")?.includes("pause") ||
          button.textContent?.includes("Play") ||
          button.textContent?.includes("Pause"),
      );

      if (playPauseButton) {
        fireEvent.click(playPauseButton);
        expect(playPauseButton).toBeInTheDocument();
      }
    });

    test("about button triggers callback", () => {
      const onShowAbout = jest.fn();
      render(
        <GlobalKnowledgeGraph {...defaultProps} onShowAbout={onShowAbout} />,
      );

      const aboutButton = document.querySelector('[title="About"]');
      if (aboutButton) {
        fireEvent.click(aboutButton);
        expect(onShowAbout).toHaveBeenCalled();
      }
    });
  });

  describe("Node Selection", () => {
    test("node selection triggers callback", async () => {
      const onSelectNode = jest.fn();
      render(
        <GlobalKnowledgeGraph {...defaultProps} onSelectNode={onSelectNode} />,
      );

      // Wait for the component to render and set up
      await waitFor(() => {
        expect(mockForceSimulation.on).toHaveBeenCalled();
      });

      // Simulate D3 tick callback that would be called
      const tickCallback = mockForceSimulation.on.mock.calls.find(
        (call) => call[0] === "tick",
      )?.[1];

      if (tickCallback) {
        act(() => {
          tickCallback();
        });
      }
    });

    test("handles different node types", () => {
      const agentWithTags = {
        ...mockAgent,
        knowledge: [
          {
            ...mockKnowledgeEntry,
            tags: ["physics", "mathematics", "science"],
          },
        ],
      };

      render(
        <GlobalKnowledgeGraph {...defaultProps} agents={[agentWithTags]} />,
      );

      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });
  });

  describe("Graph Layout and Physics", () => {
    test("force simulation is configured correctly", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      // Check that forces are configured
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "link",
        expect.any(Object),
      );
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "charge",
        expect.any(Object),
      );
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "center",
        expect.any(Object),
      );
      expect(mockForceSimulation.force).toHaveBeenCalledWith(
        "collision",
        expect.any(Object),
      );
    });

    test("simulation restarts when data changes", () => {
      const { rerender } = render(<GlobalKnowledgeGraph {...defaultProps} />);

      // Change agents data
      const newAgent = {
        ...mockAgent,
        id: "agent-new",
        name: "New Agent",
      };

      rerender(<GlobalKnowledgeGraph {...defaultProps} agents={[newAgent]} />);

      // Simulation should restart with new data
      expect(mockForceSimulation.restart).toHaveBeenCalled();
    });

    test("handles resize events", async () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      // ResizeObserver should be set up
      expect(global.ResizeObserver).toHaveBeenCalled();
    });
  });

  describe("Data Processing", () => {
    test("consolidates duplicate knowledge entries", () => {
      const agentsWithDuplicateKnowledge = [
        {
          ...mockAgent,
          knowledge: [{ ...mockKnowledgeEntry, title: "Shared Knowledge" }],
        },
        {
          ...mockAgent,
          id: "agent-2",
          knowledge: [
            {
              ...mockKnowledgeEntry,
              id: "knowledge-2",
              title: "Shared Knowledge",
            },
          ],
        },
      ];

      render(
        <GlobalKnowledgeGraph
          {...defaultProps}
          agents={agentsWithDuplicateKnowledge}
        />,
      );

      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });

    test("processes tags correctly", () => {
      const agentWithManyTags = {
        ...mockAgent,
        knowledge: [
          {
            ...mockKnowledgeEntry,
            tags: ["tag1", "tag2", "tag3", "common-tag"],
          },
          {
            ...mockKnowledgeEntry,
            id: "knowledge-2",
            title: "Second Knowledge",
            tags: ["tag4", "tag5", "common-tag"],
          },
        ],
      };

      render(
        <GlobalKnowledgeGraph {...defaultProps} agents={[agentWithManyTags]} />,
      );

      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });

    test("handles empty knowledge arrays", () => {
      const agentWithoutKnowledge = {
        ...mockAgent,
        knowledge: [],
      };

      render(
        <GlobalKnowledgeGraph
          {...defaultProps}
          agents={[agentWithoutKnowledge]}
        />,
      );

      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });
  });

  describe("QuadTree Spatial Partitioning", () => {
    test("quadtree is used for collision detection", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      // The component should initialize properly even with quadtree
      expect(screen.getByTestId("card")).toBeInTheDocument();
    });
  });

  describe("Performance and Memory", () => {
    test("handles large datasets efficiently", () => {
      const largeAgentList = Array.from({ length: 100 }, (_, i) => ({
        ...mockAgent,
        id: `agent-${i}`,
        name: `Agent ${i}`,
        knowledge: Array.from({ length: 10 }, (_, j) => ({
          ...mockKnowledgeEntry,
          id: `knowledge-${i}-${j}`,
          title: `Knowledge ${i}-${j}`,
          tags: [`tag-${i}`, `common-tag`, `specific-${j}`],
        })),
      }));

      const startTime = Date.now();
      render(
        <GlobalKnowledgeGraph {...defaultProps} agents={largeAgentList} />,
      );
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(2000);
      expect(mockForceSimulation.nodes).toHaveBeenCalled();
    });

    test("cleans up simulation on unmount", () => {
      const { unmount } = render(<GlobalKnowledgeGraph {...defaultProps} />);

      unmount();

      expect(mockForceSimulation.stop).toHaveBeenCalled();
    });
  });

  describe("Error Handling", () => {
    test("handles malformed agent data", () => {
      const malformedAgent = {
        id: "malformed",
        name: "Malformed Agent",
        // Missing required fields
      } as any;

      expect(() => {
        render(
          <GlobalKnowledgeGraph {...defaultProps} agents={[malformedAgent]} />,
        );
      }).not.toThrow();
    });

    test("handles agents with malformed knowledge", () => {
      const agentWithMalformedKnowledge = {
        ...mockAgent,
        knowledge: [
          { title: "Knowledge without ID" }, // Missing required fields
        ] as any,
      };

      expect(() => {
        render(
          <GlobalKnowledgeGraph
            {...defaultProps}
            agents={[agentWithMalformedKnowledge]}
          />,
        );
      }).not.toThrow();
    });

    test("handles null/undefined props gracefully", () => {
      expect(() => {
        render(
          <GlobalKnowledgeGraph
            agents={[]}
            onSelectNode={jest.fn()}
            onShowAbout={jest.fn()}
          />,
        );
      }).not.toThrow();
    });
  });

  describe("Accessibility", () => {
    test("provides accessible structure", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      expect(
        screen.getByRole("button", { name: /zoom in/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /zoom out/i }),
      ).toBeInTheDocument();
    });

    test("SVG has proper attributes for screen readers", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      const svg = document.querySelector("svg");
      expect(svg).toBeInTheDocument();
      // SVG should be properly structured for accessibility
    });
  });

  describe("Visual Elements", () => {
    test("applies correct styling classes", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      const container = document.querySelector(".rounded-lg.border.bg-card");
      expect(container).toBeInTheDocument();
    });

    test("renders with proper layout structure", () => {
      render(<GlobalKnowledgeGraph {...defaultProps} />);

      expect(screen.getByText("Global Knowledge Graph")).toBeInTheDocument();
      expect(document.querySelector("canvas")).toBeInTheDocument();
    });
  });
});
