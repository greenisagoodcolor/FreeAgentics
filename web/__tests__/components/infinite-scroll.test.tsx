/**
 * Infinite Scroll Component Tests
 * Intersection Observer based pagination
 */

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { jest } from "@jest/globals";

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
  root: null,
  rootMargin: "",
  thresholds: [],
}));

// Infinite Scroll Component
interface InfiniteScrollProps {
  children: React.ReactNode;
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  loading?: boolean;
  threshold?: number;
  loader?: React.ReactNode;
}

const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  children,
  onLoadMore,
  hasMore,
  loading = false,
  threshold = 100,
  loader = <div>Loading...</div>,
}) => {
  const [isLoading, setIsLoading] = React.useState(false);
  const sentinelRef = React.useRef<HTMLDivElement>(null);
  const observerRef = React.useRef<IntersectionObserver | null>(null);

  React.useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry.isIntersecting && !isLoading && !loading) {
          setIsLoading(true);
          onLoadMore().finally(() => setIsLoading(false));
        }
      },
      {
        rootMargin: `${threshold}px`,
      },
    );

    observerRef.current.observe(sentinel);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, isLoading, loading, onLoadMore, threshold]);

  return (
    <div data-testid="infinite-scroll">
      {children}
      {hasMore && (
        <div ref={sentinelRef} data-testid="infinite-scroll-sentinel">
          {(isLoading || loading) && loader}
        </div>
      )}
    </div>
  );
};

// Test Components
const TestInfiniteScrollList: React.FC<{
  items: string[];
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  loading?: boolean;
}> = ({ items, onLoadMore, hasMore, loading }) => {
  return (
    <InfiniteScroll 
      onLoadMore={onLoadMore} 
      hasMore={hasMore}
      loading={loading}
    >
      {items.map((item, index) => (
        <div key={index} data-testid={`item-${index}`}>
          {item}
        </div>
      ))}
    </InfiniteScroll>
  );
};

// Tests
describe("Infinite Scroll", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("should render children and sentinel when hasMore is true", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div data-testid="content">Content</div>
      </InfiniteScroll>,
    );

    expect(screen.getByTestId("infinite-scroll")).toBeInTheDocument();
    expect(screen.getByTestId("content")).toBeInTheDocument();
    expect(screen.getByTestId("infinite-scroll-sentinel")).toBeInTheDocument();
  });

  test("should not render sentinel when hasMore is false", () => {
    const onLoadMore = jest.fn();

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={false}>
        <div data-testid="content">Content</div>
      </InfiniteScroll>,
    );

    expect(screen.getByTestId("infinite-scroll")).toBeInTheDocument();
    expect(screen.getByTestId("content")).toBeInTheDocument();
    expect(
      screen.queryByTestId("infinite-scroll-sentinel"),
    ).not.toBeInTheDocument();
  });

  test("should call onLoadMore when sentinel intersects", async () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div style={{ height: "1000px" }}>Content</div>
      </InfiniteScroll>,
    );

    const sentinel = screen.getByTestId("infinite-scroll-sentinel");

    // Mock IntersectionObserver trigger
    const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];

    // Simulate intersection
    callback([{ isIntersecting: true, target: sentinel }]);

    expect(onLoadMore).toHaveBeenCalled();
  });

  test("should show loading state", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <InfiniteScroll
        onLoadMore={onLoadMore}
        hasMore={true}
        loading={true}
        loader={<div data-testid="custom-loader">Loading...</div>}
      >
        <div>Content</div>
      </InfiniteScroll>,
    );

    expect(screen.getByTestId("custom-loader")).toBeInTheDocument();
  });

  test("should show default loader when loading", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <InfiniteScroll
        onLoadMore={onLoadMore}
        hasMore={true}
        loading={true}
      >
        <div>Content</div>
      </InfiniteScroll>,
    );

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  test("should handle load more failure gracefully", async () => {
    const onLoadMore = jest.fn().mockRejectedValue(new Error("Load failed"));

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    const sentinel = screen.getByTestId("infinite-scroll-sentinel");
    const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];

    callback([{ isIntersecting: true, target: sentinel }]);

    await waitFor(() => {
      expect(onLoadMore).toHaveBeenCalled();
    });

    // Should not crash on error
    expect(screen.getByTestId("infinite-scroll")).toBeInTheDocument();
  });

  test("should not load more when already loading", async () => {
    const onLoadMore = jest.fn().mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 100))
    );

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    const sentinel = screen.getByTestId("infinite-scroll-sentinel");
    const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];

    // Trigger multiple times quickly
    callback([{ isIntersecting: true, target: sentinel }]);
    callback([{ isIntersecting: true, target: sentinel }]);
    callback([{ isIntersecting: true, target: sentinel }]);

    // Should only call once
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  test("should use custom threshold", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);
    const customThreshold = 200;

    render(
      <InfiniteScroll 
        onLoadMore={onLoadMore} 
        hasMore={true}
        threshold={customThreshold}
      >
        <div>Content</div>
      </InfiniteScroll>,
    );

    // Verify observer was created with custom threshold
    expect(global.IntersectionObserver).toHaveBeenCalledWith(
      expect.any(Function),
      {
        rootMargin: `${customThreshold}px`,
      }
    );
  });

  test("should cleanup observer on unmount", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);
    const mockDisconnect = jest.fn();

    (global.IntersectionObserver as jest.Mock).mockImplementation(() => ({
      observe: jest.fn(),
      unobserve: jest.fn(),
      disconnect: mockDisconnect,
      root: null,
      rootMargin: "",
      thresholds: [],
    }));

    const { unmount } = render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    unmount();

    expect(mockDisconnect).toHaveBeenCalled();
  });

  test("should handle hasMore changes", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    const { rerender } = render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    expect(screen.getByTestId("infinite-scroll-sentinel")).toBeInTheDocument();

    // Change hasMore to false
    rerender(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={false}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    expect(
      screen.queryByTestId("infinite-scroll-sentinel"),
    ).not.toBeInTheDocument();
  });

  test("should work with dynamic content", () => {
    const items = ["Item 1", "Item 2", "Item 3"];
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <TestInfiniteScrollList
        items={items}
        onLoadMore={onLoadMore}
        hasMore={true}
      />
    );

    // Check all items are rendered
    expect(screen.getByTestId("item-0")).toHaveTextContent("Item 1");
    expect(screen.getByTestId("item-1")).toHaveTextContent("Item 2");
    expect(screen.getByTestId("item-2")).toHaveTextContent("Item 3");
    expect(screen.getByTestId("infinite-scroll-sentinel")).toBeInTheDocument();
  });

  test("should handle no intersection", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];

    // Simulate no intersection
    callback([{ isIntersecting: false, target: null }]);

    expect(onLoadMore).not.toHaveBeenCalled();
  });

  test("should handle external loading state", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    const { rerender } = render(
      <TestInfiniteScrollList
        items={["Item 1"]}
        onLoadMore={onLoadMore}
        hasMore={true}
        loading={false}
      />
    );

    // No loader initially
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();

    // Set external loading
    rerender(
      <TestInfiniteScrollList
        items={["Item 1"]}
        onLoadMore={onLoadMore}
        hasMore={true}
        loading={true}
      />
    );

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  test("should prevent loading when external loading is true", () => {
    const onLoadMore = jest.fn().mockResolvedValue(undefined);

    render(
      <InfiniteScroll onLoadMore={onLoadMore} hasMore={true} loading={true}>
        <div>Content</div>
      </InfiniteScroll>,
    );

    const sentinel = screen.getByTestId("infinite-scroll-sentinel");
    const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];

    // Try to trigger when already loading externally
    callback([{ isIntersecting: true, target: sentinel }]);

    expect(onLoadMore).not.toHaveBeenCalled();
  });

  test("should handle multiple observer instances", () => {
    const onLoadMore1 = jest.fn().mockResolvedValue(undefined);
    const onLoadMore2 = jest.fn().mockResolvedValue(undefined);

    render(
      <div>
        <InfiniteScroll onLoadMore={onLoadMore1} hasMore={true}>
          <div>Content 1</div>
        </InfiniteScroll>
        <InfiniteScroll onLoadMore={onLoadMore2} hasMore={true}>
          <div>Content 2</div>
        </InfiniteScroll>
      </div>
    );

    // Should create separate observers for each instance
    expect(global.IntersectionObserver).toHaveBeenCalledTimes(2);
  });
});