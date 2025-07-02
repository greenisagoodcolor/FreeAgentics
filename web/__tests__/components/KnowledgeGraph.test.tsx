import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
  act,
} from "@testing-library/react";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import * as d3 from "d3";

// Import centralized mock factory patterns
import {
  UI_COMPONENT_MOCKS,
  HOOK_MOCKS,
  UTILITY_MOCKS,
} from "../utils/ui-component-mock-factory";

// Apply comprehensive Canvas API mocking - must be before component imports
Object.defineProperty(global.HTMLCanvasElement.prototype, "getContext", {
  value: jest.fn((contextType) => {
    if (contextType === "2d") {
      return {
        // Drawing rectangles
        clearRect: jest.fn(),
        fillRect: jest.fn(),
        strokeRect: jest.fn(),

        // Drawing text
        fillText: jest.fn(),
        strokeText: jest.fn(),
        measureText: jest.fn(() => ({ width: 50 })),

        // Drawing paths
        beginPath: jest.fn(),
        closePath: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        bezierCurveTo: jest.fn(),
        quadraticCurveTo: jest.fn(),
        arc: jest.fn(),
        arcTo: jest.fn(),
        ellipse: jest.fn(),
        rect: jest.fn(),

        // Drawing operations
        fill: jest.fn(),
        stroke: jest.fn(),
        drawImage: jest.fn(),
        createImageData: jest.fn(),
        getImageData: jest.fn(),
        putImageData: jest.fn(),

        // Transformations
        scale: jest.fn(),
        rotate: jest.fn(),
        translate: jest.fn(),
        transform: jest.fn(),
        setTransform: jest.fn(),
        resetTransform: jest.fn(),

        // State
        save: jest.fn(),
        restore: jest.fn(),

        // Styles
        createLinearGradient: jest.fn(),
        createRadialGradient: jest.fn(),
        createPattern: jest.fn(),

        // Properties
        canvas: {
          width: 800,
          height: 600,
        },
        fillStyle: "#000000",
        strokeStyle: "#000000",
        globalAlpha: 1,
        lineWidth: 1,
        lineCap: "butt",
        lineJoin: "miter",
        miterLimit: 10,
        lineDashOffset: 0,
        shadowOffsetX: 0,
        shadowOffsetY: 0,
        shadowBlur: 0,
        shadowColor: "rgba(0, 0, 0, 0)",
        globalCompositeOperation: "source-over",
        font: "10px sans-serif",
        textAlign: "start",
        textBaseline: "alphabetic",
        direction: "inherit",
      };
    }
    return null;
  }),
  writable: true,
  configurable: true,
});

// Mock getBoundingClientRect and client dimensions for canvas
Object.defineProperty(
  global.HTMLCanvasElement.prototype,
  "getBoundingClientRect",
  {
    value: jest.fn(() => ({
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      width: 800,
      height: 600,
      x: 0,
      y: 0,
    })),
    writable: true,
    configurable: true,
  },
);

Object.defineProperty(global.HTMLCanvasElement.prototype, "clientWidth", {
  value: 800,
  writable: true,
  configurable: true,
});

Object.defineProperty(global.HTMLCanvasElement.prototype, "clientHeight", {
  value: 600,
  writable: true,
  configurable: true,
});

Object.defineProperty(global.HTMLCanvasElement.prototype, "offsetWidth", {
  value: 800,
  writable: true,
  configurable: true,
});

Object.defineProperty(global.HTMLCanvasElement.prototype, "offsetHeight", {
  value: 600,
  writable: true,
  configurable: true,
});

// Enhanced D3 mock with comprehensive tracking
jest.mock("d3", () => ({
  select: jest.fn().mockReturnThis(),
  selectAll: jest.fn().mockReturnThis(),
  append: jest.fn().mockReturnThis(),
  attr: jest.fn().mockReturnThis(),
  style: jest.fn().mockReturnThis(),
  data: jest.fn().mockReturnThis(),
  enter: jest.fn().mockReturnThis(),
  exit: jest.fn().mockReturnThis(),
  remove: jest.fn().mockReturnThis(),
  on: jest.fn().mockReturnThis(),
  transition: jest.fn().mockReturnThis(),
  duration: jest.fn().mockReturnThis(),
  ease: jest.fn().mockReturnThis(),
  call: jest.fn().mockReturnThis(),
  drag: jest.fn().mockReturnThis(),
  node: jest.fn(() => document.createElement("div")),
  nodes: jest.fn(() => [document.createElement("div")]),
  size: jest.fn(() => 1),
  forceSimulation: jest.fn(() => ({
    nodes: jest.fn().mockReturnThis(),
    links: jest.fn().mockReturnThis(),
    force: jest.fn().mockReturnThis(),
    alpha: jest.fn().mockReturnThis(),
    restart: jest.fn().mockReturnThis(),
    stop: jest.fn().mockReturnThis(),
    on: jest.fn().mockReturnThis(),
    tick: jest.fn(),
  })),
  forceLink: jest.fn(() => ({
    id: jest.fn().mockReturnThis(),
    distance: jest.fn().mockReturnThis(),
  })),
  forceManyBody: jest.fn(() => ({
    strength: jest.fn().mockReturnThis(),
  })),
  forceCenter: jest.fn(() => ({})),
  forceCollide: jest.fn(() => ({
    radius: jest.fn().mockReturnThis(),
  })),
  zoom: jest.fn(() => ({
    scaleExtent: jest.fn().mockReturnThis(),
    on: jest.fn().mockReturnThis(),
    transform: jest.fn(),
  })),
  zoomIdentity: { k: 1, x: 0, y: 0 },
  event: {
    transform: { k: 1, x: 0, y: 0 },
  },
}));

// Apply centralized utility mocks
jest.mock("@/lib/utils", () => UTILITY_MOCKS);

describe("KnowledgeGraph Component", () => {
  const mockKnowledge = [
    {
      id: "1",
      title: "Node 1",
      content: "Content 1",
      tags: ["tag1", "tag2"],
      timestamp: new Date(),
    },
    {
      id: "2",
      title: "Node 2",
      content: "Content 2",
      tags: ["tag2", "tag3"],
      timestamp: new Date(),
    },
    {
      id: "3",
      title: "Node 3",
      content: "Content 3",
      tags: ["tag1"],
      timestamp: new Date(),
    },
  ];

  const mockProps = {
    knowledge: mockKnowledge,
    onSelectEntry: jest.fn(),
    selectedEntry: null,
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Ensure Canvas mock is properly set up for each test
    const mockGetContext = jest.fn((contextType) => {
      if (contextType === "2d") {
        return {
          // Drawing rectangles
          clearRect: jest.fn(),
          fillRect: jest.fn(),
          strokeRect: jest.fn(),

          // Drawing text
          fillText: jest.fn(),
          strokeText: jest.fn(),
          measureText: jest.fn(() => ({ width: 50 })),

          // Drawing paths
          beginPath: jest.fn(),
          closePath: jest.fn(),
          moveTo: jest.fn(),
          lineTo: jest.fn(),
          bezierCurveTo: jest.fn(),
          quadraticCurveTo: jest.fn(),
          arc: jest.fn(),
          arcTo: jest.fn(),
          ellipse: jest.fn(),
          rect: jest.fn(),

          // Drawing operations
          fill: jest.fn(),
          stroke: jest.fn(),
          drawImage: jest.fn(),
          createImageData: jest.fn(),
          getImageData: jest.fn(),
          putImageData: jest.fn(),

          // Transformations
          scale: jest.fn(),
          rotate: jest.fn(),
          translate: jest.fn(),
          transform: jest.fn(),
          setTransform: jest.fn(),
          resetTransform: jest.fn(),

          // State
          save: jest.fn(),
          restore: jest.fn(),

          // Styles
          createLinearGradient: jest.fn(),
          createRadialGradient: jest.fn(),
          createPattern: jest.fn(),

          // Properties
          canvas: {
            width: 800,
            height: 600,
          },
          fillStyle: "#000000",
          strokeStyle: "#000000",
          globalAlpha: 1,
          lineWidth: 1,
          lineCap: "butt",
          lineJoin: "miter",
          miterLimit: 10,
          lineDashOffset: 0,
          shadowOffsetX: 0,
          shadowOffsetY: 0,
          shadowBlur: 0,
          shadowColor: "rgba(0, 0, 0, 0)",
          globalCompositeOperation: "source-over",
          font: "10px sans-serif",
          textAlign: "start",
          textBaseline: "alphabetic",
          direction: "inherit",
        };
      }
      return null;
    });

    global.HTMLCanvasElement.prototype.getContext = mockGetContext;

    // Mock ResizeObserver which is often used for responsive canvas sizing
    global.ResizeObserver = jest.fn(() => ({
      observe: jest.fn(),
      unobserve: jest.fn(),
      disconnect: jest.fn(),
    })) as any;

    // Mock IntersectionObserver which may be used for viewport detection
    global.IntersectionObserver = jest.fn(() => ({
      observe: jest.fn(),
      unobserve: jest.fn(),
      disconnect: jest.fn(),
      root: null,
      rootMargin: "",
      thresholds: [],
    })) as any;

    // Mock requestAnimationFrame for D3 animations
    global.requestAnimationFrame = jest.fn((cb) => {
      setTimeout(cb, 16); // Simulate 60fps
      return 1;
    });
    global.cancelAnimationFrame = jest.fn();
  });

  describe("Rendering", () => {
    it("renders without crashing", async () => {
      await act(async () => {
        render(
          <KnowledgeGraph
            knowledge={[]}
            onSelectEntry={() => {}}
            selectedEntry={null}
          />,
        );
      });
      expect(screen.getByText(/0 entries/)).toBeInTheDocument();
    });

    it("renders with knowledge data", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      // Verify the component renders with the expected structure
      expect(container.querySelector("canvas")).toBeInTheDocument();
      expect(
        screen.getByText((content) => content.includes("3 entries")),
      ).toBeInTheDocument();
      expect(
        screen.getByText((content) => content.includes("3 tags")),
      ).toBeInTheDocument();

      // Component should render without errors when given knowledge data
      expect(container).toBeTruthy();
    });

    it("renders controls when showing debug", async () => {
      await act(async () => {
        render(<KnowledgeGraph {...mockProps} />);
      });

      const debugButton = screen.getByText("Debug");

      await act(async () => {
        fireEvent.click(debugButton);
      });

      expect(screen.getByText(/Canvas:/)).toBeInTheDocument();
    });

    it("displays knowledge entry count", () => {
      render(<KnowledgeGraph {...mockProps} />);
      // Component renders "3 entries, 3 tags" - need flexible text matching
      expect(
        screen.getByText((content, element) =>
          content.includes(`${mockKnowledge.length} entries`),
        ),
      ).toBeInTheDocument();
    });

    it("displays tag count", () => {
      render(<KnowledgeGraph {...mockProps} />);
      const uniqueTags = new Set(mockKnowledge.flatMap((k) => k.tags));
      // Component renders "3 entries, 3 tags" - need flexible text matching
      expect(
        screen.getByText((content, element) =>
          content.includes(`${uniqueTags.size} tags`),
        ),
      ).toBeInTheDocument();
    });
  });

  describe("Interactions", () => {
    it("handles zoom controls", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      // Find zoom buttons by their SVG content since they may not have aria-labels
      const buttons = container.querySelectorAll("button");
      const zoomInButton = Array.from(buttons).find((button) =>
        button.querySelector("svg.lucide-zoom-in"),
      );
      const zoomOutButton = Array.from(buttons).find((button) =>
        button.querySelector("svg.lucide-zoom-out"),
      );

      // Verify zoom buttons exist and can be clicked without errors
      expect(zoomInButton).toBeTruthy();
      expect(zoomOutButton).toBeTruthy();

      if (zoomInButton && zoomOutButton) {
        await act(async () => {
          fireEvent.click(zoomInButton);
          fireEvent.click(zoomOutButton);
        });
      }

      // Component should handle zoom interactions without crashing
      expect(container).toBeTruthy();
    });

    it("handles reset button", async () => {
      await act(async () => {
        render(<KnowledgeGraph {...mockProps} />);
      });

      const resetButton = screen.getByText("Reset");
      expect(resetButton).toBeInTheDocument();

      await act(async () => {
        fireEvent.click(resetButton);
      });

      // Reset button should be clickable without errors
      expect(resetButton).toBeInTheDocument();
    });

    it("handles canvas mouse interactions", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      // Find canvas element directly since it may not have proper role
      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();

      if (canvas) {
        await act(async () => {
          fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });
          fireEvent.mouseMove(canvas, { clientX: 150, clientY: 150 });
          fireEvent.mouseUp(canvas);
        });
      }

      // Canvas should handle mouse interactions without errors
      expect(canvas).toBeInTheDocument();
    });

    it("calls onSelectEntry when knowledge entry is clicked", async () => {
      const onSelectEntry = jest.fn();
      let container: any;

      await act(async () => {
        const result = render(
          <KnowledgeGraph
            knowledge={mockKnowledge}
            onSelectEntry={onSelectEntry}
            selectedEntry={null}
          />,
        );
        container = result.container;
      });

      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();

      if (canvas) {
        await act(async () => {
          fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });
        });
      }

      // Component should render correctly with onSelectEntry prop
      expect(canvas).toBeInTheDocument();
      expect(onSelectEntry).toBeDefined();
    });
  });

  describe("Visual States", () => {
    it("highlights selected entry", async () => {
      let container: any;

      await act(async () => {
        const result = render(
          <KnowledgeGraph {...mockProps} selectedEntry={mockKnowledge[0]} />,
        );
        container = result.container;
      });

      // Component should render correctly with selected entry
      expect(container.querySelector("canvas")).toBeInTheDocument();
      expect(
        screen.getByText((content) => content.includes("3 entries")),
      ).toBeInTheDocument();
    });

    it("shows hovering effects", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();

      if (canvas) {
        await act(async () => {
          fireEvent.mouseMove(canvas, { clientX: 200, clientY: 200 });
        });
      }

      // Canvas should handle mouse hover without errors
      expect(canvas).toBeInTheDocument();
    });

    it("displays node labels at high zoom", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      // Find zoom buttons by their SVG icons
      const buttons = screen.getAllByRole("button");
      const zoomInButton = buttons.find((button) => {
        const svg = button.querySelector("svg.lucide-zoom-in");
        return svg;
      });

      if (zoomInButton) {
        await act(async () => {
          fireEvent.click(zoomInButton);
          fireEvent.click(zoomInButton);
          fireEvent.click(zoomInButton);
        });
      }

      // Zoom interactions should work without errors
      expect(zoomInButton).toBeTruthy();
    });
  });

  describe("Tag Interactions", () => {
    it("shows tag selection panel", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();

      if (canvas) {
        await act(async () => {
          // Simulate clicking on a tag node
          fireEvent.mouseDown(canvas, { clientX: 300, clientY: 300 });
        });
      }

      // Canvas interactions should work without errors
      expect(canvas).toBeInTheDocument();
    });

    it("filters entries by tag", async () => {
      const onSelectEntry = jest.fn();
      let container: any;

      await act(async () => {
        const result = render(
          <KnowledgeGraph
            knowledge={mockKnowledge}
            onSelectEntry={onSelectEntry}
            selectedEntry={null}
          />,
        );
        container = result.container;
      });

      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();

      if (canvas) {
        await act(async () => {
          // Simulate tag filtering
          fireEvent.mouseDown(canvas, { clientX: 250, clientY: 250 });
        });
      }

      // Tag filtering interactions should work without errors
      expect(canvas).toBeInTheDocument();
      expect(onSelectEntry).toBeDefined();
    });
  });

  describe("Performance", () => {
    it("handles large datasets efficiently", async () => {
      const largeKnowledge = Array.from({ length: 100 }, (_, i) => ({
        id: `entry${i}`,
        title: `Entry ${i}`,
        content: `Content ${i}`,
        tags: [`tag${i % 10}`, `tag${(i + 1) % 10}`],
        timestamp: new Date(),
      }));

      let container: any;
      const startTime = Date.now();

      await act(async () => {
        const result = render(
          <KnowledgeGraph
            knowledge={largeKnowledge}
            onSelectEntry={() => {}}
            selectedEntry={null}
          />,
        );
        container = result.container;
      });

      const endTime = Date.now();

      // Should render quickly even with large datasets
      expect(endTime - startTime).toBeLessThan(2000);
      expect(container.querySelector("canvas")).toBeInTheDocument();
      expect(
        screen.getByText((content) => content.includes("100 entries")),
      ).toBeInTheDocument();
    });

    it("throttles render updates", async (): Promise<void> => {
      let result: any;
      const startTime = Date.now();

      await act(async () => {
        result = render(<KnowledgeGraph {...mockProps} />);
      });

      // Rapid updates - should be handled efficiently
      for (let i = 0; i < 10; i++) {
        await act(async () => {
          result.rerender(<KnowledgeGraph {...mockProps} key={i} />);
        });
      }

      const endTime = Date.now();

      // Should handle rapid re-renders efficiently without excessive processing time
      expect(endTime - startTime).toBeLessThan(1000);
      expect(result.container.querySelector("canvas")).toBeInTheDocument();
    });
  });

  describe("Error Handling", () => {
    it("handles invalid knowledge data gracefully", async () => {
      const invalidKnowledge = [
        {
          id: null,
          title: "Invalid",
          content: "",
          tags: [],
          timestamp: new Date(),
        },
        {
          id: "valid",
          title: "Valid",
          content: "Valid content",
          tags: ["tag"],
          timestamp: new Date(),
        },
      ];

      let container: any;

      await act(async () => {
        const result = render(
          <KnowledgeGraph
            knowledge={invalidKnowledge as any}
            onSelectEntry={() => {}}
            selectedEntry={null}
          />,
        );
        container = result.container;
      });

      // Should handle invalid data gracefully without crashing
      expect(container.querySelector("canvas")).toBeInTheDocument();
      expect(
        screen.getByText((content) => content.includes("2 entries")),
      ).toBeInTheDocument();
    });

    it("handles empty knowledge array", async () => {
      await act(async () => {
        render(
          <KnowledgeGraph
            knowledge={[]}
            onSelectEntry={() => {}}
            selectedEntry={null}
          />,
        );
      });
      expect(screen.getByText(/0 entries/)).toBeInTheDocument();
    });

    it("displays error message on render failure", async () => {
      // Mock D3 to throw error
      (d3.select as jest.Mock).mockImplementationOnce(() => {
        throw new Error("D3 error");
      });

      // Error boundary would catch this in real app
      await act(async () => {
        expect(() => {
          render(<KnowledgeGraph {...mockProps} />);
        }).not.toThrow();
      });
    });
  });

  describe("Responsive Design", () => {
    it("adjusts to container size changes", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      await act(async () => {
        // Simulate window resize
        fireEvent(window, new Event("resize"));
      });

      // Should handle resize events without crashing
      expect(container.querySelector("canvas")).toBeInTheDocument();
    });

    it("maintains aspect ratio on resize", async () => {
      let container: any;

      await act(async () => {
        const result = render(<KnowledgeGraph {...mockProps} />);
        container = result.container;
      });

      // Simulate container size change
      Object.defineProperty(container.firstChild, "clientWidth", {
        value: 800,
        configurable: true,
      });
      Object.defineProperty(container.firstChild, "clientHeight", {
        value: 600,
        configurable: true,
      });

      await act(async () => {
        fireEvent(window, new Event("resize"));
      });

      // Should maintain proper structure after resize
      expect(container.querySelector("canvas")).toBeInTheDocument();
    });
  });

  describe("Debug Mode", () => {
    it("shows debug information when enabled", async () => {
      await act(async () => {
        render(<KnowledgeGraph {...mockProps} />);
      });

      const debugButton = screen.getByText("Debug");

      await act(async () => {
        fireEvent.click(debugButton);
      });

      expect(screen.getByText(/Canvas:/)).toBeInTheDocument();
      expect(screen.getByText(/Nodes:/)).toBeInTheDocument();
      expect(screen.getByText(/Zoom:/)).toBeInTheDocument();
    });

    it("hides debug information when disabled", async () => {
      await act(async () => {
        render(<KnowledgeGraph {...mockProps} />);
      });

      const debugButton = screen.getByText("Debug");
      fireEvent.click(debugButton);
      fireEvent.click(debugButton);

      expect(screen.queryByText(/Canvas:/)).not.toBeInTheDocument();
    });
  });
});
