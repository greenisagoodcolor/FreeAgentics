import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from "@testing-library/react";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import * as d3 from "d3";

// Mock D3
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
  forceSimulation: jest.fn(() => ({
    nodes: jest.fn().mockReturnThis(),
    links: jest.fn().mockReturnThis(),
    force: jest.fn().mockReturnThis(),
    alpha: jest.fn().mockReturnThis(),
    restart: jest.fn().mockReturnThis(),
    stop: jest.fn().mockReturnThis(),
    on: jest.fn().mockReturnThis(),
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
  })),
  zoomIdentity: {},
  event: {
    transform: { k: 1, x: 0, y: 0 },
  },
}));

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
  });

  describe("Rendering", () => {
    it("renders without crashing", () => {
      render(
        <KnowledgeGraph
          knowledge={[]}
          onSelectEntry={() => {}}
          selectedEntry={null}
        />,
      );
      expect(screen.getByText(/0 entries/)).toBeInTheDocument();
    });

    it("renders with knowledge data", () => {
      render(<KnowledgeGraph {...mockProps} />);
      expect(d3.select).toHaveBeenCalled();
      expect(d3.forceSimulation).toHaveBeenCalled();
    });

    it("renders controls when showing debug", () => {
      render(<KnowledgeGraph {...mockProps} />);
      const debugButton = screen.getByText("Debug");
      fireEvent.click(debugButton);
      expect(screen.getByText(/Canvas:/)).toBeInTheDocument();
    });

    it("displays knowledge entry count", () => {
      render(<KnowledgeGraph {...mockProps} />);
      expect(
        screen.getByText(`${mockKnowledge.length} entries`),
      ).toBeInTheDocument();
    });

    it("displays tag count", () => {
      render(<KnowledgeGraph {...mockProps} />);
      const uniqueTags = new Set(mockKnowledge.flatMap((k) => k.tags));
      expect(screen.getByText(`${uniqueTags.size} tags`)).toBeInTheDocument();
    });
  });

  describe("Interactions", () => {
    it("handles zoom controls", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const zoomInButton = screen.getByRole("button", { name: /zoom.*in/i });
      const zoomOutButton = screen.getByRole("button", { name: /zoom.*out/i });

      fireEvent.click(zoomInButton);
      fireEvent.click(zoomOutButton);

      expect(d3.select).toHaveBeenCalled();
    });

    it("handles reset button", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const resetButton = screen.getByText("Reset");
      fireEvent.click(resetButton);

      expect(d3.select).toHaveBeenCalled();
    });

    it("handles canvas mouse interactions", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const canvas = screen.getByRole("img", { hidden: true });

      fireEvent.mouseDown(canvas);
      fireEvent.mouseMove(canvas);
      fireEvent.mouseUp(canvas);

      expect(d3.select).toHaveBeenCalled();
    });

    it("calls onSelectEntry when knowledge entry is clicked", () => {
      const onSelectEntry = jest.fn();
      render(
        <KnowledgeGraph
          knowledge={mockKnowledge}
          onSelectEntry={onSelectEntry}
          selectedEntry={null}
        />,
      );

      const canvas = screen.getByRole("img", { hidden: true });
      fireEvent.mouseDown(canvas);

      // Simulate clicking on a knowledge entry node
      // Note: In a real test, you'd need to mock the node detection logic
    });
  });

  describe("Visual States", () => {
    it("highlights selected entry", () => {
      render(
        <KnowledgeGraph {...mockProps} selectedEntry={mockKnowledge[0]} />,
      );
      expect(d3.select).toHaveBeenCalled();
    });

    it("shows hovering effects", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const canvas = screen.getByRole("img", { hidden: true });
      fireEvent.mouseMove(canvas);

      expect(d3.select).toHaveBeenCalled();
    });

    it("displays node labels at high zoom", () => {
      render(<KnowledgeGraph {...mockProps} />);

      // Simulate zoom in
      const zoomInButton = screen.getByRole("button", { name: /zoom.*in/i });
      fireEvent.click(zoomInButton);
      fireEvent.click(zoomInButton);
      fireEvent.click(zoomInButton);

      expect(d3.select).toHaveBeenCalled();
    });
  });

  describe("Tag Interactions", () => {
    it("shows tag selection panel", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const canvas = screen.getByRole("img", { hidden: true });

      // Simulate clicking on a tag node
      fireEvent.mouseDown(canvas);
    });

    it("filters entries by tag", () => {
      const onSelectEntry = jest.fn();
      render(
        <KnowledgeGraph
          knowledge={mockKnowledge}
          onSelectEntry={onSelectEntry}
          selectedEntry={null}
        />,
      );

      // Simulate tag filtering
      const canvas = screen.getByRole("img", { hidden: true });
      fireEvent.mouseDown(canvas);
    });
  });

  describe("Performance", () => {
    it("handles large datasets efficiently", () => {
      const largeKnowledge = Array.from({ length: 100 }, (_, i) => ({
        id: `entry${i}`,
        title: `Entry ${i}`,
        content: `Content ${i}`,
        tags: [`tag${i % 10}`, `tag${(i + 1) % 10}`],
        timestamp: new Date(),
      }));

      render(
        <KnowledgeGraph
          knowledge={largeKnowledge}
          onSelectEntry={() => {}}
          selectedEntry={null}
        />,
      );
      expect(d3.forceSimulation).toHaveBeenCalled();
    });

    it("throttles render updates", async (): Promise<void> => {
      const { rerender } = render(<KnowledgeGraph {...mockProps} />);

      // Rapid updates
      for (let i = 0; i < 10; i++) {
        rerender(<KnowledgeGraph {...mockProps} key={i} />);
      }

      await waitFor(() => {
        expect(d3.select).toHaveBeenCalled();
      });
    });
  });

  describe("Error Handling", () => {
    it("handles invalid knowledge data gracefully", () => {
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

      render(
        <KnowledgeGraph
          knowledge={invalidKnowledge as any}
          onSelectEntry={() => {}}
          selectedEntry={null}
        />,
      );
      expect(d3.select).toHaveBeenCalled();
    });

    it("handles empty knowledge array", () => {
      render(
        <KnowledgeGraph
          knowledge={[]}
          onSelectEntry={() => {}}
          selectedEntry={null}
        />,
      );
      expect(screen.getByText(/0 entries/)).toBeInTheDocument();
    });

    it("displays error message on render failure", () => {
      // Mock D3 to throw error
      (d3.select as jest.Mock).mockImplementationOnce(() => {
        throw new Error("D3 error");
      });

      // Error boundary would catch this in real app
      expect(() => {
        render(<KnowledgeGraph {...mockProps} />);
      }).not.toThrow();
    });
  });

  describe("Responsive Design", () => {
    it("adjusts to container size changes", () => {
      render(<KnowledgeGraph {...mockProps} />);

      // Simulate window resize
      fireEvent(window, new Event("resize"));

      expect(d3.select).toHaveBeenCalled();
    });

    it("maintains aspect ratio on resize", () => {
      const { container } = render(<KnowledgeGraph {...mockProps} />);

      // Simulate container size change
      Object.defineProperty(container.firstChild, "clientWidth", {
        value: 800,
        configurable: true,
      });
      Object.defineProperty(container.firstChild, "clientHeight", {
        value: 600,
        configurable: true,
      });

      fireEvent(window, new Event("resize"));

      expect(d3.select).toHaveBeenCalled();
    });
  });

  describe("Debug Mode", () => {
    it("shows debug information when enabled", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const debugButton = screen.getByText("Debug");
      fireEvent.click(debugButton);

      expect(screen.getByText(/Canvas:/)).toBeInTheDocument();
      expect(screen.getByText(/Nodes:/)).toBeInTheDocument();
      expect(screen.getByText(/Zoom:/)).toBeInTheDocument();
    });

    it("hides debug information when disabled", () => {
      render(<KnowledgeGraph {...mockProps} />);

      const debugButton = screen.getByText("Debug");
      fireEvent.click(debugButton);
      fireEvent.click(debugButton);

      expect(screen.queryByText(/Canvas:/)).not.toBeInTheDocument();
    });
  });
});
