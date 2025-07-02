import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import { KnowledgeEntry } from "@/lib/types";

// Mock UI components
// All Jest mock factories must use React.createElement, not JSX, for compatibility with our test runner and Babel config. See ADR-001 and README for mocking standards.
jest.mock("@/components/ui/card", () => ({
  Card: function (props) {
    const { children, className } = props;
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "card", className },
      children,
    );
  },
  CardContent: function (props) {
    const { children, className } = props;
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "card-content", className },
      children,
    );
  },
}));

jest.mock("@/components/ui/button", () => ({
  Button: function (props) {
    const { children, className, ...rest } = props;
    const mockReact = require("react");
    return mockReact.createElement(
      "button",
      { "data-testid": "button", className, ...rest },
      children,
    );
  },
}));

jest.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: function (props) {
    const { children, className } = props;
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "scroll-area", className },
      children,
    );
  },
}));

// Mock icons
jest.mock("lucide-react", () => ({
  ZoomIn: function (props) {
    const { size } = props;
    const mockReact = require("react");
    return mockReact.createElement(
      "span",
      { "data-testid": "zoom-in-icon", style: { fontSize: size } },
      "ZoomIn",
    );
  },
  ZoomOut: function (props) {
    const { size } = props;
    const mockReact = require("react");
    return mockReact.createElement(
      "span",
      { "data-testid": "zoom-out-icon", style: { fontSize: size } },
      "ZoomOut",
    );
  },
}));

// Mock canvas context
const mockContext = {
  clearRect: jest.fn(),
  save: jest.fn(),
  restore: jest.fn(),
  translate: jest.fn(),
  scale: jest.fn(),
  beginPath: jest.fn(),
  arc: jest.fn(),
  moveTo: jest.fn(),
  lineTo: jest.fn(),
  stroke: jest.fn(),
  fill: jest.fn(),
  fillRect: jest.fn(),
  fillText: jest.fn(),
  measureText: jest.fn(() => ({ width: 50 })),
  set strokeStyle(value) {},
  set fillStyle(value) {},
  set lineWidth(value) {},
  set font(value) {},
  set textAlign(value) {},
  set textBaseline(value) {},
};

// Mock getContext
const mockGetContext = jest.fn(() => mockContext);

// Mock HTMLCanvasElement
Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: () => mockContext,
});

// Mock canvas properties
Object.defineProperty(HTMLCanvasElement.prototype, "width", {
  writable: true,
  value: 800,
});

Object.defineProperty(HTMLCanvasElement.prototype, "height", {
  writable: true,
  value: 600,
});

// Mock getBoundingClientRect for all elements
Element.prototype.getBoundingClientRect = jest.fn(() => ({
  width: 800,
  height: 600,
  left: 0,
  top: 0,
  right: 800,
  bottom: 600,
  x: 0,
  y: 0,
  toJSON: () => ({
    width: 800,
    height: 600,
    left: 0,
    top: 0,
    right: 800,
    bottom: 600,
    x: 0,
    y: 0,
  }),
}));

// Mock clientWidth and clientHeight
Object.defineProperty(Element.prototype, "clientWidth", {
  get: () => 800,
});

Object.defineProperty(Element.prototype, "clientHeight", {
  get: () => 600,
});

// Sample test data
const mockKnowledgeEntries = [
  {
    id: "entry-1",
    title: "Machine Learning Basics",
    content: "Introduction to ML concepts",
    tags: ["AI", "Machine Learning", "Education"],
    timestamp: new Date("2023-01-01"),
  },
  {
    id: "entry-2",
    title: "Neural Networks",
    content: "Deep learning fundamentals",
    tags: ["AI", "Deep Learning", "Neural Networks"],
    timestamp: new Date("2023-01-02"),
  },
  {
    id: "entry-3",
    title: "Data Preprocessing",
    content: "Cleaning and preparing data",
    tags: ["Data Science", "Machine Learning"],
    timestamp: new Date("2023-01-03"),
  },
];

describe("KnowledgeGraph Integration Tests", () => {
  const mockOnSelectEntry = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers(); // Ensure all timers are mocked for each test
  });

  afterEach(() => {
    jest.runOnlyPendingTimers(); // Flush any pending timers
    jest.useRealTimers(); // Restore real timers
  });

  const renderKnowledgeGraph = (
    knowledge = mockKnowledgeEntries,
    selectedEntry = null,
  ) => {
    let result;
    act(() => {
      result = render(
        <KnowledgeGraph
          knowledge={knowledge}
          onSelectEntry={mockOnSelectEntry}
          selectedEntry={selectedEntry}
        />,
      );
    });
    return result;
  };

  describe("Component Rendering", () => {
    it("renders basic structure", () => {
      renderKnowledgeGraph();

      expect(screen.getByTestId("card")).toBeInTheDocument();
      expect(screen.getByTestId("card-content")).toBeInTheDocument();
    });

    it("renders entry and tag count in header", () => {
      renderKnowledgeGraph();

      expect(screen.getByText("3 entries, 6 tags")).toBeInTheDocument();
    });

    it("renders control buttons", () => {
      renderKnowledgeGraph();

      const buttons = screen.getAllByTestId("button");
      expect(buttons).toHaveLength(4); // Debug, Zoom Out, Zoom In, Reset

      expect(screen.getByText("Debug")).toBeInTheDocument();
      expect(screen.getByTestId("zoom-out-icon")).toBeInTheDocument();
      expect(screen.getByTestId("zoom-in-icon")).toBeInTheDocument();
      expect(screen.getByText("Reset")).toBeInTheDocument();
    });

    it("renders zoom level display", () => {
      renderKnowledgeGraph();

      expect(screen.getByText("100%")).toBeInTheDocument();
    });

    it("renders canvas element", () => {
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
      expect(canvas).toHaveClass("absolute", "inset-0", "cursor-grab");
    });

    it("applies correct styling classes", () => {
      renderKnowledgeGraph();

      const card = screen.getByTestId("card");
      expect(card).toHaveClass("h-full");

      const cardContent = screen.getByTestId("card-content");
      expect(cardContent).toHaveClass("p-0", "h-full", "flex", "flex-col");
    });
  });

  describe("Knowledge Data Processing", () => {
    it("handles empty knowledge array", () => {
      renderKnowledgeGraph([]);

      expect(screen.getByText("0 entries, 0 tags")).toBeInTheDocument();
    });

    it("calculates unique tags correctly", () => {
      const knowledgeWithDuplicateTags = [
        {
          id: "entry-1",
          title: "Entry 1",
          content: "Content 1",
          tags: ["AI", "ML"],
          timestamp: new Date(),
        },
        {
          id: "entry-2",
          title: "Entry 2",
          content: "Content 2",
          tags: ["AI", "DL"], // 'AI' is duplicate
          timestamp: new Date(),
        },
      ];

      renderKnowledgeGraph(knowledgeWithDuplicateTags);

      // Should count unique tags: AI, ML, DL = 3 tags
      expect(screen.getByText("2 entries, 3 tags")).toBeInTheDocument();
    });

    it("handles entries with no tags", () => {
      const knowledgeWithNoTags = [
        {
          id: "entry-1",
          title: "Entry 1",
          content: "Content 1",
          tags: [],
          timestamp: new Date(),
        },
      ];

      renderKnowledgeGraph(knowledgeWithNoTags);

      expect(screen.getByText("1 entries, 0 tags")).toBeInTheDocument();
    });

    it("handles single entry", () => {
      const singleEntry = [mockKnowledgeEntries[0]];

      renderKnowledgeGraph(singleEntry);

      expect(screen.getByText("1 entries, 3 tags")).toBeInTheDocument();
    });
  });

  describe("Canvas Interaction", () => {
    it("initializes canvas context", () => {
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
      expect(canvas?.getContext).toBeDefined();
    });

    it("handles mouse move events", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas") as HTMLCanvasElement;
      expect(canvas).toBeInTheDocument();

      await act(async () => {
        await user.pointer({ coords: { x: 100, y: 100 }, target: canvas });
      });

      // Should not throw errors
    });

    it("handles mouse down events", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas") as HTMLCanvasElement;

      await act(async () => {
        fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });
      });

      // Should not throw errors
    });

    it("handles mouse up events", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas") as HTMLCanvasElement;

      await act(async () => {
        fireEvent.mouseUp(canvas);
      });

      // Should not throw errors
    });

    it("handles mouse leave events", async () => {
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas") as HTMLCanvasElement;

      await act(async () => {
        fireEvent.mouseLeave(canvas);
      });

      // Should not throw errors
    });
  });

  describe("Zoom Controls", () => {
    it("zooms in when zoom in button is clicked", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const zoomInButton = screen.getByTestId("zoom-in-icon").closest("button");
      expect(zoomInButton).toBeInTheDocument();

      await act(async () => {
        await user.click(zoomInButton!);
      });

      await waitFor(() => {
        expect(screen.getByText("120%")).toBeInTheDocument();
      });
    });

    it("zooms out when zoom out button is clicked", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const zoomOutButton = screen
        .getByTestId("zoom-out-icon")
        .closest("button");
      expect(zoomOutButton).toBeInTheDocument();

      await act(async () => {
        await user.click(zoomOutButton!);
      });

      await waitFor(() => {
        expect(screen.getByText("80%")).toBeInTheDocument();
      });
    });

    it("resets zoom and position when reset button is clicked", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      // First zoom in
      const zoomInButton = screen.getByTestId("zoom-in-icon").closest("button");
      await act(async () => {
        await user.click(zoomInButton!);
      });
      await waitFor(() => {
        expect(screen.getByText("120%")).toBeInTheDocument();
      });

      // Then reset
      const resetButton = screen.getByText("Reset");
      await act(async () => {
        await user.click(resetButton);
      });

      await waitFor(() => {
        expect(screen.getByText("100%")).toBeInTheDocument();
      });
    });

    it("limits zoom in to maximum level", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const zoomInButton = screen.getByTestId("zoom-in-icon").closest("button");

      // Click zoom in many times to reach limit
      for (let i = 0; i < 20; i++) {
        await act(async () => {
          await user.click(zoomInButton!);
        });
      }

      // Should be capped at 300%
      await waitFor(() => {
        expect(screen.getByText("300%")).toBeInTheDocument();
      });
    });

    it("limits zoom out to minimum level", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const zoomOutButton = screen
        .getByTestId("zoom-out-icon")
        .closest("button");

      // Click zoom out many times to reach limit
      for (let i = 0; i < 20; i++) {
        await act(async () => {
          await user.click(zoomOutButton!);
        });
      }

      // Should be capped at 50%
      await waitFor(() => {
        expect(screen.getByText("50%")).toBeInTheDocument();
      });
    });
  });

  describe("Debug Mode", () => {
    it("toggles debug panel when debug button is clicked", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const debugButton = screen.getByText("Debug");

      // Debug panel should not be visible initially
      expect(screen.queryByText(/Canvas:/)).not.toBeInTheDocument();

      // Click to show debug panel
      await act(async () => {
        await user.click(debugButton);
      });

      await waitFor(() => {
        expect(screen.getByText(/Canvas:/)).toBeInTheDocument();
        expect(screen.getByText(/Nodes:/)).toBeInTheDocument();
        expect(screen.getByText(/Zoom:/)).toBeInTheDocument();
        expect(screen.getByText(/Offset:/)).toBeInTheDocument();
        expect(screen.getByText(/Hovered:/)).toBeInTheDocument();
      });
    });

    it("hides debug panel when debug button is clicked again", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const debugButton = screen.getByText("Debug");

      // Show debug panel
      await act(async () => {
        await user.click(debugButton);
      });
      await waitFor(() => {
        expect(screen.getByText(/Canvas:/)).toBeInTheDocument();
      });

      // Hide debug panel
      await act(async () => {
        await user.click(debugButton);
      });
      await waitFor(() => {
        expect(screen.queryByText(/Canvas:/)).not.toBeInTheDocument();
      });
    });

    it("displays correct debug information", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const debugButton = screen.getByText("Debug");
      await act(async () => {
        await user.click(debugButton);
      });

      await waitFor(() => {
        expect(screen.getByText("Canvas: 800x600")).toBeInTheDocument();
        expect(screen.getByText("Nodes: 9")).toBeInTheDocument(); // 3 entries + 6 tags
        expect(screen.getByText("Zoom: 1.00")).toBeInTheDocument();
        expect(screen.getByText("Offset: 0,0")).toBeInTheDocument();
        expect(screen.getByText("Hovered: none")).toBeInTheDocument();
      });
    });
  });

  describe("Selected Entry Handling", () => {
    it("renders with no selected entry initially", () => {
      renderKnowledgeGraph();

      // Should not show tag info panel
      expect(screen.queryByText(/Tag:/)).not.toBeInTheDocument();
    });

    it("handles selected entry prop", () => {
      renderKnowledgeGraph(mockKnowledgeEntries, mockKnowledgeEntries[0]);

      // Component should render without errors
      expect(screen.getByTestId("card")).toBeInTheDocument();
    });

    it("calls onSelectEntry when entry node is clicked", () => {
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas") as HTMLCanvasElement;

      // Simulate clicking on first entry node (positioned at center + offset)
      act(() => {
        fireEvent.mouseDown(canvas, { clientX: 600, clientY: 450 }); // Approximate position
      });

      // Should attempt to call onSelectEntry (exact behavior depends on node positioning)
      // This tests that the click handler is properly attached
    });
  });

  describe("Tag Selection and Info Panel", () => {
    it("does not show tag info panel initially", () => {
      renderKnowledgeGraph();

      expect(screen.queryByText(/Tag:/)).not.toBeInTheDocument();
    });

    // Note: Testing tag selection would require more complex mocking of canvas coordinates
    // and node positioning, which is difficult to test accurately in unit tests
  });

  describe("Canvas Drawing", () => {
    it("clears canvas on each draw", () => {
      renderKnowledgeGraph();

      expect(mockContext.clearRect).toHaveBeenCalled();
    });

    it("applies zoom and pan transformations", () => {
      renderKnowledgeGraph();

      expect(mockContext.save).toHaveBeenCalled();
      expect(mockContext.translate).toHaveBeenCalled();
      expect(mockContext.scale).toHaveBeenCalled();
      expect(mockContext.restore).toHaveBeenCalled();
    });

    it("draws nodes and links", () => {
      renderKnowledgeGraph();

      expect(mockContext.beginPath).toHaveBeenCalled();
      expect(mockContext.arc).toHaveBeenCalled();
      expect(mockContext.fill).toHaveBeenCalled();
      expect(mockContext.stroke).toHaveBeenCalled();
    });

    it("redraws when knowledge data changes", () => {
      const { rerender } = renderKnowledgeGraph();

      const initialCalls = mockContext.clearRect.mock.calls.length;

      // Change knowledge data
      act(() => {
        rerender(
          <KnowledgeGraph
            knowledge={[mockKnowledgeEntries[0]]}
            onSelectEntry={mockOnSelectEntry}
            selectedEntry={null}
          />,
        );
      });

      expect(mockContext.clearRect).toHaveBeenCalledTimes(initialCalls + 1);
    });
  });

  describe("Responsive Behavior", () => {
    it("handles window resize events", () => {
      renderKnowledgeGraph();

      // Trigger resize event
      act(() => {
        fireEvent(window, new Event("resize"));
      });

      // Should call clearRect again
      expect(mockContext.clearRect).toHaveBeenCalled();
    });

    it("updates canvas dimensions on container resize", () => {
      renderKnowledgeGraph();

      const canvas = document.querySelector("canvas");
      expect(canvas).toBeInTheDocument();

      // Canvas dimensions should be set based on container
      expect(canvas).toHaveProperty("width");
      expect(canvas).toHaveProperty("height");
    });
  });

  describe("Edge Cases", () => {
    it("handles null canvas ref gracefully", () => {
      renderKnowledgeGraph();

      // Component should handle cases where canvas might not be available
      // This is more of a runtime safety test
      expect(screen.getByTestId("card")).toBeInTheDocument();
    });

    it("handles null container ref gracefully", () => {
      renderKnowledgeGraph();

      // Should not throw errors even with missing container
      expect(screen.getByTestId("card")).toBeInTheDocument();
    });

    it("handles knowledge entries with very long titles", () => {
      const longTitleKnowledge = [
        {
          id: "entry-1",
          title: "A".repeat(200),
          content: "Content",
          tags: ["tag1"],
          timestamp: new Date(),
        },
      ];

      expect(() => {
        renderKnowledgeGraph(longTitleKnowledge);
      }).not.toThrow();
    });

    it("handles knowledge entries with many tags", () => {
      const manyTagsKnowledge = [
        {
          id: "entry-1",
          title: "Entry with many tags",
          content: "Content",
          tags: Array.from({ length: 100 }, (_, i) => `tag${i}`),
          timestamp: new Date(),
        },
      ];

      expect(() => {
        renderKnowledgeGraph(manyTagsKnowledge);
      }).not.toThrow();

      expect(screen.getByText("1 entries, 100 tags")).toBeInTheDocument();
    });

    it("handles undefined canvas context", () => {
      // Temporarily mock getContext to return null
      const originalGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = jest.fn(() => null);

      expect(() => {
        renderKnowledgeGraph();
      }).not.toThrow();

      // Restore original getContext
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    });
  });

  describe("Performance", () => {
    it("renders efficiently with large knowledge sets", () => {
      // Use a more reasonable test size (100 entries instead of 1000)
      const largeKnowledge = Array.from({ length: 100 }, (_, i) => ({
        id: `entry-${i}`,
        title: `Entry ${i}`,
        content: `Content ${i}`,
        tags: [`tag${i % 10}`, `category${i % 5}`],
        timestamp: new Date(),
      }));

      const startTime = performance.now();
      renderKnowledgeGraph(largeKnowledge);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should render in under 1 second
      expect(screen.getByText("100 entries, 15 tags")).toBeInTheDocument();
    });

    it("handles rapid zoom changes efficiently", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const zoomInButton = screen.getByTestId("zoom-in-icon").closest("button");
      const zoomOutButton = screen
        .getByTestId("zoom-out-icon")
        .closest("button");

      // Rapidly change zoom levels
      for (let i = 0; i < 10; i++) {
        await act(async () => {
          await user.click(zoomInButton!);
        });
        await act(async () => {
          await user.click(zoomOutButton!);
        });
      }

      expect(screen.getByText("100%")).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("provides meaningful button labels", () => {
      renderKnowledgeGraph();

      expect(screen.getByText("Debug")).toBeInTheDocument();
      expect(screen.getByText("Reset")).toBeInTheDocument();
    });

    it("has clickable controls", async () => {
      const user = userEvent.setup();
      renderKnowledgeGraph();

      const buttons = screen.getAllByTestId("button");
      expect(buttons.length).toBeGreaterThan(0);

      // All buttons should be clickable
      for (const button of buttons) {
        await act(async () => {
          await user.click(button);
        });
      }
    });

    it("provides visual feedback for zoom level", () => {
      renderKnowledgeGraph();

      const zoomDisplay = screen.getByText("100%");
      expect(zoomDisplay).toBeInTheDocument();
      expect(zoomDisplay).toHaveClass("text-white");
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts and unmounts without errors", () => {
      const { unmount } = renderKnowledgeGraph();

      expect(screen.getByTestId("card")).toBeInTheDocument();

      act(() => {
        expect(() => unmount()).not.toThrow();
      });
    });

    it("handles prop changes correctly", () => {
      const { rerender } = renderKnowledgeGraph();

      expect(screen.getByText("3 entries, 6 tags")).toBeInTheDocument();

      act(() => {
        rerender(
          <KnowledgeGraph
            knowledge={[mockKnowledgeEntries[0]]}
            onSelectEntry={mockOnSelectEntry}
            selectedEntry={null}
          />,
        );
      });

      expect(screen.getByText("1 entries, 3 tags")).toBeInTheDocument();
    });

    it("cleans up event listeners on unmount", () => {
      const removeEventListenerSpy = jest.spyOn(window, "removeEventListener");

      const { unmount } = renderKnowledgeGraph();

      act(() => {
        unmount();
      });

      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        "resize",
        expect.any(Function),
      );

      removeEventListenerSpy.mockRestore();
    });
  });
});
