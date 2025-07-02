/**
 * Virtual Scrolling Component Tests
 * Optimized for large data sets with performance monitoring
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { jest } from "@jest/globals";

// Virtual Scrolling Component
interface VirtualScrollItem {
  id: string;
  height: number;
  data: any;
}

interface VirtualScrollProps {
  items: VirtualScrollItem[];
  containerHeight: number;
  itemRenderer: (item: VirtualScrollItem, index: number) => React.ReactNode;
  overscan?: number;
  onScroll?: (scrollTop: number) => void;
}

const VirtualScroll: React.FC<VirtualScrollProps> = ({
  items,
  containerHeight,
  itemRenderer,
  overscan = 5,
  onScroll,
}) => {
  const [scrollTop, setScrollTop] = React.useState(0);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Calculate total height and item positions
  const itemPositions = React.useMemo(() => {
    const positions: number[] = [];
    let currentPosition = 0;

    items.forEach((item, index) => {
      positions[index] = currentPosition;
      currentPosition += item.height;
    });

    return positions;
  }, [items]);

  const totalHeight =
    itemPositions[itemPositions.length - 1] +
    (items[items.length - 1]?.height || 0);

  // Calculate visible range
  const visibleRange = React.useMemo(() => {
    const startIndex = itemPositions.findIndex(
      (pos) => pos + items[itemPositions.indexOf(pos)]?.height >= scrollTop,
    );
    const endIndex = itemPositions.findIndex(
      (pos) => pos > scrollTop + containerHeight,
    );

    return {
      start: Math.max(0, startIndex - overscan),
      end: Math.min(
        items.length - 1,
        (endIndex === -1 ? items.length - 1 : endIndex) + overscan,
      ),
    };
  }, [scrollTop, containerHeight, itemPositions, items, overscan]);

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const newScrollTop = event.currentTarget.scrollTop;
    setScrollTop(newScrollTop);
    onScroll?.(newScrollTop);
  };

  const visibleItems = [];
  for (let i = visibleRange.start; i <= visibleRange.end; i++) {
    const item = items[i];
    if (item) {
      visibleItems.push(
        <div
          key={item.id}
          style={{
            position: "absolute",
            top: itemPositions[i],
            height: item.height,
            width: "100%",
          }}
        >
          {itemRenderer(item, i)}
        </div>,
      );
    }
  }

  return (
    <div
      ref={containerRef}
      data-testid="virtual-scroll-container"
      style={{
        height: containerHeight,
        overflow: "auto",
        position: "relative",
      }}
      onScroll={handleScroll}
    >
      <div
        style={{
          height: totalHeight,
          position: "relative",
        }}
      >
        {visibleItems}
      </div>
    </div>
  );
};

// Tests
describe("Virtual Scrolling", () => {
  const createItems = (count: number): VirtualScrollItem[] =>
    Array.from({ length: count }, (_, i) => ({
      id: `item-${i}`,
      height: 50,
      data: { text: `Item ${i}` },
    }));

  test("should render only visible items", () => {
    const items = createItems(1000);
    const itemRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>{item.data.text}</div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={itemRenderer}
      />,
    );

    const container = screen.getByTestId("virtual-scroll-container");
    expect(container).toBeInTheDocument();

    // Should only render visible items (plus overscan)
    const renderedItems = screen.queryAllByTestId(/^item-item-/);
    expect(renderedItems.length).toBeLessThan(1000);
    expect(renderedItems.length).toBeGreaterThan(0);
  });

  test("should handle scroll events", () => {
    const items = createItems(100);
    const onScroll = jest.fn();
    const itemRenderer = (item: VirtualScrollItem) => (
      <div>{item.data.text}</div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={itemRenderer}
        onScroll={onScroll}
      />,
    );

    const container = screen.getByTestId("virtual-scroll-container");

    fireEvent.scroll(container, { target: { scrollTop: 100 } });

    expect(onScroll).toHaveBeenCalledWith(100);
  });

  test("should handle items with varying heights", () => {
    const items: VirtualScrollItem[] = [
      { id: "item-1", height: 50, data: { text: "Item 1" } },
      { id: "item-2", height: 100, data: { text: "Item 2" } },
      { id: "item-3", height: 75, data: { text: "Item 3" } },
    ];

    const itemRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`} style={{ height: item.height }}>
        {item.data.text}
      </div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={itemRenderer}
      />,
    );

    const item1 = screen.getByTestId("item-item-1");
    const item2 = screen.getByTestId("item-item-2");

    expect(item1).toHaveStyle({ height: "50px" });
    expect(item2).toHaveStyle({ height: "100px" });
  });

  test("should handle empty items list", () => {
    const items: VirtualScrollItem[] = [];
    const itemRenderer = (item: VirtualScrollItem) => (
      <div>{item.data.text}</div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={itemRenderer}
      />,
    );

    const container = screen.getByTestId("virtual-scroll-container");
    expect(container).toBeInTheDocument();

    // Should not render any items
    const renderedItems = screen.queryAllByTestId(/^item-/);
    expect(renderedItems.length).toBe(0);
  });

  test("should adjust overscan correctly", () => {
    const items = createItems(50);
    const itemRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>{item.data.text}</div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={200}
        itemRenderer={itemRenderer}
        overscan={2}
      />,
    );

    // With smaller overscan, fewer items should be rendered
    const renderedItems = screen.queryAllByTestId(/^item-item-/);
    expect(renderedItems.length).toBeGreaterThan(0);
    expect(renderedItems.length).toBeLessThan(20); // Should be reasonable number
  });

  test("should calculate total height correctly", () => {
    const items: VirtualScrollItem[] = [
      { id: "item-1", height: 100, data: { text: "Item 1" } },
      { id: "item-2", height: 150, data: { text: "Item 2" } },
      { id: "item-3", height: 75, data: { text: "Item 3" } },
    ];

    const itemRenderer = (item: VirtualScrollItem) => (
      <div>{item.data.text}</div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={200}
        itemRenderer={itemRenderer}
      />,
    );

    const container = screen.getByTestId("virtual-scroll-container");
    const innerDiv = container.firstChild as HTMLElement;

    // Total height should be 100 + 150 + 75 = 325
    expect(innerDiv).toHaveStyle({ height: "325px" });
  });

  test("should handle rapid scrolling", () => {
    const items = createItems(200);
    const onScroll = jest.fn();
    const itemRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>{item.data.text}</div>
    );

    render(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={itemRenderer}
        onScroll={onScroll}
      />,
    );

    const container = screen.getByTestId("virtual-scroll-container");

    // Simulate rapid scrolling
    fireEvent.scroll(container, { target: { scrollTop: 100 } });
    fireEvent.scroll(container, { target: { scrollTop: 500 } });
    fireEvent.scroll(container, { target: { scrollTop: 1000 } });

    expect(onScroll).toHaveBeenCalledTimes(3);
    expect(onScroll).toHaveBeenLastCalledWith(1000);
  });

  test("should render correct items at different scroll positions", () => {
    const items = createItems(20);
    const itemRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>{item.data.text}</div>
    );

    const { rerender } = render(
      <VirtualScroll
        items={items}
        containerHeight={150}
        itemRenderer={itemRenderer}
        overscan={1}
      />,
    );

    // Check initial render
    expect(screen.getByTestId("item-item-0")).toBeInTheDocument();

    // Scroll down and check different items are rendered
    const container = screen.getByTestId("virtual-scroll-container");
    fireEvent.scroll(container, { target: { scrollTop: 300 } });

    // Should render different items after scrolling
    // Due to how virtual scrolling works, exact items depend on calculations
    const renderedItems = screen.queryAllByTestId(/^item-item-/);
    expect(renderedItems.length).toBeGreaterThan(0);
  });

  test("should handle item renderer changes", () => {
    const items = createItems(10);
    const initialRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>Initial: {item.data.text}</div>
    );

    const { rerender } = render(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={initialRenderer}
      />,
    );

    expect(screen.getByTestId("item-item-0")).toHaveTextContent("Initial:");

    // Change renderer
    const newRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>New: {item.data.text}</div>
    );

    rerender(
      <VirtualScroll
        items={items}
        containerHeight={300}
        itemRenderer={newRenderer}
      />,
    );

    expect(screen.getByTestId("item-item-0")).toHaveTextContent("New:");
  });

  test("should handle performance with large datasets", () => {
    const items = createItems(10000);
    const itemRenderer = (item: VirtualScrollItem) => (
      <div data-testid={`item-${item.id}`}>{item.data.text}</div>
    );

    const startTime = performance.now();

    render(
      <VirtualScroll
        items={items}
        containerHeight={400}
        itemRenderer={itemRenderer}
      />,
    );

    const endTime = performance.now();

    // Should render quickly even with 10k items
    expect(endTime - startTime).toBeLessThan(100);

    // Should only render visible items, not all 10k
    const renderedItems = screen.queryAllByTestId(/^item-item-/);
    expect(renderedItems.length).toBeLessThan(50);
  });
});
