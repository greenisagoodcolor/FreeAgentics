/**
 * React Testing Utilities for Frontend Tests.
 *
 * Expert Committee: Kent C. Dodds (React Testing), Dan Abramov (React)
 * Provides standardized utilities for React component testing.
 */

import React from "react";
import { render, RenderOptions, RenderResult } from "@testing-library/react";
import { act } from "react-dom/test-utils";
import userEvent from "@testing-library/user-event";

// Mock providers and wrappers
export const AllTheProviders = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  return <>{children}</>;
};

// Custom render function with providers
export const customRender = (
  ui: React.ReactElement,
  options?: Omit<RenderOptions, "wrapper">,
): RenderResult => {
  return render(ui, { wrapper: AllTheProviders, ...options });
};

// Act wrapper utility
export const actWrapper = async (callback: () => void | Promise<void>) => {
  await act(async () => {
    await callback();
  });
};

// Mock component factory
export const createMockComponent = (testId: string, displayName?: string) => {
  const MockComponent = (props: any) => (
    <div data-testid={testId} {...props}>
      {displayName || testId}
      {props.children}
    </div>
  );
  MockComponent.displayName = displayName || "MockComponent";
  return MockComponent;
};

// WebSocket mock factory
export const createMockWebSocket = () => {
  const mockWebSocket = {
    send: jest.fn(),
    close: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    readyState: WebSocket.OPEN,
  };

  return mockWebSocket;
};

// Router mock utilities
export const createMockRouter = () => ({
  push: jest.fn(),
  replace: jest.fn(),
  refresh: jest.fn(),
  back: jest.fn(),
  forward: jest.fn(),
  prefetch: jest.fn(),
});

export const createMockSearchParams = (
  params: Record<string, string> = {},
) => ({
  get: jest.fn((key: string) => params[key] || null),
  has: jest.fn((key: string) => key in params),
  toString: jest.fn(() => new URLSearchParams(params).toString()),
});

// User event setup
export const setupUser = () => userEvent.setup();

// Wait utilities
export const waitForAnimation = () =>
  new Promise((resolve) => requestAnimationFrame(resolve));

export const waitForDebounce = (ms: number = 300) =>
  new Promise((resolve) => setTimeout(resolve, ms));

// Mock timers utilities
export const runWithFakeTimers = async (
  callback: () => void | Promise<void>,
) => {
  jest.useFakeTimers();
  try {
    await callback();
    jest.runAllTimers();
  } finally {
    jest.useRealTimers();
  }
};

// Performance mock
export const mockPerformance = () => {
  const originalPerformance = global.performance;

  global.performance = {
    ...originalPerformance,
    now: jest.fn(() => Date.now()),
    mark: jest.fn(),
    measure: jest.fn(),
    clearMarks: jest.fn(),
    clearMeasures: jest.fn(),
    getEntriesByName: jest.fn(() => []),
    getEntriesByType: jest.fn(() => []),
  } as any;

  return () => {
    global.performance = originalPerformance;
  };
};

// LocalStorage mock
export const mockLocalStorage = () => {
  const storage: Record<string, string> = {};

  const localStorageMock = {
    getItem: jest.fn((key: string) => storage[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      storage[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete storage[key];
    }),
    clear: jest.fn(() => {
      Object.keys(storage).forEach((key) => delete storage[key]);
    }),
  };

  Object.defineProperty(window, "localStorage", {
    value: localStorageMock,
    writable: true,
  });

  return localStorageMock;
};

// Intersection Observer mock
export const mockIntersectionObserver = () => {
  const mockIntersectionObserver = jest.fn();
  mockIntersectionObserver.mockReturnValue({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  });

  window.IntersectionObserver = mockIntersectionObserver as any;

  return mockIntersectionObserver;
};

// ResizeObserver mock
export const mockResizeObserver = () => {
  const mockResizeObserver = jest.fn();
  mockResizeObserver.mockReturnValue({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  });

  window.ResizeObserver = mockResizeObserver as any;

  return mockResizeObserver;
};

// Match Media mock
export const mockMatchMedia = (matches: boolean = false) => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: jest.fn().mockImplementation((query) => ({
      matches,
      media: query,
      onchange: null,
      addListener: jest.fn(), // deprecated
      removeListener: jest.fn(), // deprecated
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });
};

// Canvas mock for charts/graphs
export const mockCanvas = () => {
  HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    fillRect: jest.fn(),
    clearRect: jest.fn(),
    getImageData: jest.fn(() => ({
      data: new Uint8ClampedArray(4),
    })),
    putImageData: jest.fn(),
    createImageData: jest.fn(() => ({
      data: new Uint8ClampedArray(4),
    })),
    setTransform: jest.fn(),
    resetTransform: jest.fn(),
    drawImage: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    closePath: jest.fn(),
    stroke: jest.fn(),
    fill: jest.fn(),
    arc: jest.fn(),
    rect: jest.fn(),
    scale: jest.fn(),
    rotate: jest.fn(),
    translate: jest.fn(),
    transform: jest.fn(),
    createLinearGradient: jest.fn(() => ({
      addColorStop: jest.fn(),
    })),
    createRadialGradient: jest.fn(() => ({
      addColorStop: jest.fn(),
    })),
    fillText: jest.fn(),
    strokeText: jest.fn(),
    measureText: jest.fn(() => ({ width: 0 })),
  })) as any;
};

// Error boundary test helper
export const ErrorBoundaryTestHelper = ({
  children,
  onError,
}: {
  children: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}) => {
  class ErrorBoundary extends React.Component<
    { children: React.ReactNode },
    { hasError: boolean }
  > {
    constructor(props: { children: React.ReactNode }) {
      super(props);
      this.state = { hasError: false };
    }

    static getDerivedStateFromError() {
      return { hasError: true };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
      onError?.(error, errorInfo);
    }

    render() {
      if (this.state.hasError) {
        return <div>Something went wrong</div>;
      }
      return this.props.children;
    }
  }

  return <ErrorBoundary>{children}</ErrorBoundary>;
};

// Re-export everything from testing library for convenience
export * from "@testing-library/react";
export { default as userEvent } from "@testing-library/user-event";
