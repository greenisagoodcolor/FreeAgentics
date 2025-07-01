/**
 * Simplified Integration UI Tests
 * Combined interaction patterns without complex timing or touch events
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { jest } from "@jest/globals";

// Mock observers and animation frame
global.IntersectionObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
  root: null,
  rootMargin: "",
  thresholds: [],
}));

global.requestAnimationFrame = jest.fn((callback) => {
  setTimeout(callback, 16);
  return 1;
});

// Simple Responsive Hook
const useResponsive = () => {
  const [width, setWidth] = React.useState(1024);
  
  React.useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return {
    isMobile: width < 768,
    isTablet: width >= 768 && width < 1024,
    isDesktop: width >= 1024,
    width,
  };
};

// Simple Drag Context
interface DragContextValue {
  draggedItem: string | null;
  startDrag: (item: string) => void;
  endDrag: () => void;
}

const DragContext = React.createContext<DragContextValue | null>(null);

const SimpleDragProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [draggedItem, setDraggedItem] = React.useState<string | null>(null);

  const startDrag = (item: string) => setDraggedItem(item);
  const endDrag = () => setDraggedItem(null);

  return (
    <DragContext.Provider value={{ draggedItem, startDrag, endDrag }}>
      {children}
    </DragContext.Provider>
  );
};

// Simple Draggable Component
const SimpleDraggable: React.FC<{
  id: string;
  children: React.ReactNode;
}> = ({ id, children }) => {
  const context = React.useContext(DragContext);

  const handleMouseDown = () => {
    context?.startDrag(id);
  };

  return (
    <div
      data-testid={`draggable-${id}`}
      onMouseDown={handleMouseDown}
      style={{
        cursor: "grab",
        opacity: context?.draggedItem === id ? 0.5 : 1,
      }}
    >
      {children}
    </div>
  );
};

// Simple Drop Zone
const SimpleDropZone: React.FC<{
  id: string;
  children: React.ReactNode;
  onDrop?: (item: string) => void;
}> = ({ id, children, onDrop }) => {
  const context = React.useContext(DragContext);

  const handleMouseUp = () => {
    if (context?.draggedItem) {
      onDrop?.(context.draggedItem);
      context.endDrag();
    }
  };

  return (
    <div
      data-testid={`dropzone-${id}`}
      onMouseUp={handleMouseUp}
      style={{
        minHeight: "100px",
        border: "2px dashed #ccc",
        backgroundColor: context?.draggedItem ? "#f0f0f0" : "transparent",
      }}
    >
      {children}
    </div>
  );
};

// Simple List Component with Virtual Behavior
const SimpleVirtualList: React.FC<{
  items: string[];
  itemHeight?: number;
  containerHeight?: number;
}> = ({ items, itemHeight = 50, containerHeight = 300 }) => {
  const [scrollTop, setScrollTop] = React.useState(0);

  const visibleCount = Math.ceil(containerHeight / itemHeight);
  const startIndex = Math.floor(scrollTop / itemHeight);
  const endIndex = Math.min(startIndex + visibleCount, items.length);

  const visibleItems = items.slice(startIndex, endIndex);

  return (
    <div
      data-testid="virtual-list"
      style={{ height: containerHeight, overflow: "auto" }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
    >
      <div style={{ height: items.length * itemHeight, position: "relative" }}>
        {visibleItems.map((item, index) => (
          <div
            key={startIndex + index}
            data-testid={`list-item-${startIndex + index}`}
            style={{
              position: "absolute",
              top: (startIndex + index) * itemHeight,
              height: itemHeight,
              width: "100%",
            }}
          >
            {item}
          </div>
        ))}
      </div>
    </div>
  );
};

// Comprehensive Test Component
const IntegratedUITest: React.FC = () => {
  const [items, setItems] = React.useState(["Item 1", "Item 2", "Item 3"]);
  const [droppedItems, setDroppedItems] = React.useState<string[]>([]);
  const [virtualItems] = React.useState(
    Array.from({ length: 100 }, (_, i) => `Virtual Item ${i + 1}`)
  );
  const responsive = useResponsive();

  const handleDrop = (item: string) => {
    setDroppedItems((prev) => [...prev, item]);
  };

  const addItem = () => {
    setItems((prev) => [...prev, `Item ${prev.length + 1}`]);
  };

  return (
    <SimpleDragProvider>
      <div data-testid="integrated-ui" style={{ padding: "20px" }}>
        {/* Responsive Info */}
        <div data-testid="responsive-info">
          Width: {responsive.width} |{" "}
          {responsive.isMobile && "Mobile"}
          {responsive.isTablet && "Tablet"}
          {responsive.isDesktop && "Desktop"}
        </div>

        {/* Control Panel */}
        <div style={{ margin: "20px 0" }}>
          <button data-testid="add-item" onClick={addItem}>
            Add Item
          </button>
        </div>

        {/* Draggable Items */}
        <div data-testid="draggable-section" style={{ marginBottom: "20px" }}>
          <h3>Draggable Items</h3>
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
            {items.map((item, index) => (
              <SimpleDraggable key={index} id={`item-${index}`}>
                <div
                  style={{
                    padding: "10px",
                    border: "1px solid #ccc",
                    borderRadius: "4px",
                    backgroundColor: "#f9f9f9",
                  }}
                >
                  {item}
                </div>
              </SimpleDraggable>
            ))}
          </div>
        </div>

        {/* Drop Zone */}
        <div style={{ marginBottom: "20px" }}>
          <h3>Drop Zone</h3>
          <SimpleDropZone id="main" onDrop={handleDrop}>
            <div style={{ padding: "20px", textAlign: "center" }}>
              Drop items here
              {droppedItems.length > 0 && (
                <div data-testid="dropped-items">
                  Dropped: {droppedItems.join(", ")}
                </div>
              )}
            </div>
          </SimpleDropZone>
        </div>

        {/* Virtual List */}
        <div>
          <h3>Virtual List</h3>
          <SimpleVirtualList
            items={virtualItems}
            containerHeight={responsive.isMobile ? 200 : 300}
          />
        </div>
      </div>
    </SimpleDragProvider>
  );
};

// Tests
describe("Integration UI Tests", () => {
  beforeEach(() => {
    // Reset window width
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 1024,
    });
  });

  test("should render all components correctly", () => {
    render(<IntegratedUITest />);

    expect(screen.getByTestId("integrated-ui")).toBeInTheDocument();
    expect(screen.getByTestId("responsive-info")).toBeInTheDocument();
    expect(screen.getByTestId("draggable-section")).toBeInTheDocument();
    expect(screen.getByTestId("dropzone-main")).toBeInTheDocument();
    expect(screen.getByTestId("virtual-list")).toBeInTheDocument();
  });

  test("should show correct responsive state", () => {
    render(<IntegratedUITest />);

    const info = screen.getByTestId("responsive-info");
    expect(info).toHaveTextContent("Desktop");
    expect(info).toHaveTextContent("1024");
  });

  test("should handle responsive changes", () => {
    render(<IntegratedUITest />);

    // Change to mobile
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 600,
    });

    fireEvent(window, new Event("resize"));

    const info = screen.getByTestId("responsive-info");
    expect(info).toHaveTextContent("600");
  });

  test("should add new draggable items", () => {
    render(<IntegratedUITest />);

    const addButton = screen.getByTestId("add-item");
    
    // Should start with 3 items
    expect(screen.getByTestId("draggable-item-0")).toBeInTheDocument();
    expect(screen.getByTestId("draggable-item-1")).toBeInTheDocument();
    expect(screen.getByTestId("draggable-item-2")).toBeInTheDocument();

    // Add a new item
    fireEvent.click(addButton);

    expect(screen.getByTestId("draggable-item-3")).toBeInTheDocument();
  });

  test("should handle drag and drop interaction", () => {
    render(<IntegratedUITest />);

    const draggable = screen.getByTestId("draggable-item-0");
    const dropzone = screen.getByTestId("dropzone-main");

    // Start drag
    fireEvent.mouseDown(draggable);

    // Should show visual feedback
    expect(draggable).toHaveStyle({ opacity: "0.5" });

    // Drop
    fireEvent.mouseUp(dropzone);

    // Should show dropped item
    expect(screen.getByTestId("dropped-items")).toHaveTextContent("item-0");
  });

  test("should handle multiple drag and drop operations", () => {
    render(<IntegratedUITest />);

    const draggable1 = screen.getByTestId("draggable-item-0");
    const draggable2 = screen.getByTestId("draggable-item-1");
    const dropzone = screen.getByTestId("dropzone-main");

    // Drop first item
    fireEvent.mouseDown(draggable1);
    fireEvent.mouseUp(dropzone);

    // Drop second item
    fireEvent.mouseDown(draggable2);
    fireEvent.mouseUp(dropzone);

    const droppedItems = screen.getByTestId("dropped-items");
    expect(droppedItems).toHaveTextContent("item-0, item-1");
  });

  test("should render virtual list items", () => {
    render(<IntegratedUITest />);

    const virtualList = screen.getByTestId("virtual-list");
    expect(virtualList).toBeInTheDocument();

    // Should render some visible items
    expect(screen.getByTestId("list-item-0")).toHaveTextContent("Virtual Item 1");
    expect(screen.getByTestId("list-item-1")).toHaveTextContent("Virtual Item 2");
  });

  test("should handle virtual list scrolling", () => {
    render(<IntegratedUITest />);

    const virtualList = screen.getByTestId("virtual-list");

    // Scroll down
    fireEvent.scroll(virtualList, { target: { scrollTop: 250 } });

    // Should render different items after scrolling
    const items = screen.queryAllByTestId(/^list-item-/);
    expect(items.length).toBeGreaterThan(0);
  });

  test("should combine responsive behavior with other features", () => {
    render(<IntegratedUITest />);

    // Change to mobile
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 500,
    });

    fireEvent(window, new Event("resize"));

    // Should still work with drag and drop
    const draggable = screen.getByTestId("draggable-item-0");
    const dropzone = screen.getByTestId("dropzone-main");

    fireEvent.mouseDown(draggable);
    fireEvent.mouseUp(dropzone);

    expect(screen.getByTestId("dropped-items")).toHaveTextContent("item-0");
  });

  test("should handle rapid interactions", () => {
    render(<IntegratedUITest />);

    const addButton = screen.getByTestId("add-item");
    const dropzone = screen.getByTestId("dropzone-main");

    // Add multiple items quickly
    fireEvent.click(addButton);
    fireEvent.click(addButton);
    fireEvent.click(addButton);

    // Drag multiple items quickly
    fireEvent.mouseDown(screen.getByTestId("draggable-item-0"));
    fireEvent.mouseUp(dropzone);

    fireEvent.mouseDown(screen.getByTestId("draggable-item-1"));
    fireEvent.mouseUp(dropzone);

    const droppedItems = screen.getByTestId("dropped-items");
    expect(droppedItems).toHaveTextContent("item-0, item-1");
  });

  test("should maintain state across interactions", () => {
    render(<IntegratedUITest />);

    const addButton = screen.getByTestId("add-item");

    // Add item
    fireEvent.click(addButton);
    expect(screen.getByTestId("draggable-item-3")).toBeInTheDocument();

    // Drag and drop
    const draggable = screen.getByTestId("draggable-item-3");
    const dropzone = screen.getByTestId("dropzone-main");

    fireEvent.mouseDown(draggable);
    fireEvent.mouseUp(dropzone);

    // Both states should be maintained
    expect(screen.getByTestId("draggable-item-3")).toBeInTheDocument();
    expect(screen.getByTestId("dropped-items")).toHaveTextContent("item-3");
  });

  test("should handle edge cases gracefully", () => {
    render(<IntegratedUITest />);

    const dropzone = screen.getByTestId("dropzone-main");

    // Try to drop without dragging
    fireEvent.mouseUp(dropzone);

    // Should not crash or show dropped items
    expect(screen.queryByTestId("dropped-items")).not.toBeInTheDocument();
  });
});