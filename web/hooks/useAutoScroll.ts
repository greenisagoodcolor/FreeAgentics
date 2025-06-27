"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import { debounce } from "lodash-es";

export interface AutoScrollOptions {
  threshold?: number; // Distance from bottom to consider "at bottom" (default: 50px)
  smoothScrollDuration?: number; // Duration for smooth scrolling (default: 300ms)
  enableUserOverride?: boolean; // Allow user scrolling to disable auto-scroll (default: true)
  overrideTimeout?: number; // Time before re-enabling auto-scroll after user scroll (default: 5000ms)
  enableKeyboardShortcuts?: boolean; // Enable keyboard shortcuts for scroll control (default: true)
  onScrollStateChange?: (
    isAutoScrollEnabled: boolean,
    isAtBottom: boolean,
  ) => void;
  onUserOverride?: () => void;
  onAutoScrollResume?: () => void;
}

export interface AutoScrollState {
  isAutoScrollEnabled: boolean;
  isAtBottom: boolean;
  isScrolling: boolean;
  userOverrideActive: boolean;
  lastUserScrollTime: number | null;
  scrollProgress: number; // 0-1 representing scroll position
}

export interface AutoScrollControls {
  scrollToBottom: (smooth?: boolean) => void;
  scrollToTop: (smooth?: boolean) => void;
  scrollToElement: (element: HTMLElement, smooth?: boolean) => void;
  enableAutoScroll: () => void;
  disableAutoScroll: () => void;
  toggleAutoScroll: () => void;
  jumpToLatest: () => void;
  state: AutoScrollState;
}

export function useAutoScroll(
  containerRef: React.RefObject<HTMLElement>,
  dependencies: any[] = [],
  options: AutoScrollOptions = {},
): AutoScrollControls {
  const {
    threshold = 50,
    smoothScrollDuration = 300,
    enableUserOverride = true,
    overrideTimeout = 5000,
    enableKeyboardShortcuts = true,
    onScrollStateChange,
    onUserOverride,
    onAutoScrollResume,
  } = options;

  const [state, setState] = useState<AutoScrollState>({
    isAutoScrollEnabled: true,
    isAtBottom: true,
    isScrolling: false,
    userOverrideActive: false,
    lastUserScrollTime: null,
    scrollProgress: 1,
  });

  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const overrideTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUserScrollingRef = useRef(false);
  const lastScrollTopRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);

  // Calculate if we're at the bottom
  const isAtBottom = useCallback(
    (container: HTMLElement) => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      return scrollHeight - scrollTop - clientHeight <= threshold;
    },
    [threshold],
  );

  // Calculate scroll progress (0 = top, 1 = bottom)
  const getScrollProgress = useCallback((container: HTMLElement) => {
    const { scrollTop, scrollHeight, clientHeight } = container;
    const maxScroll = scrollHeight - clientHeight;
    return maxScroll > 0 ? scrollTop / maxScroll : 1;
  }, []);

  // Smooth scroll to position
  const smoothScrollTo = useCallback(
    (
      container: HTMLElement,
      targetPosition: number,
      duration: number = smoothScrollDuration,
    ) => {
      const startPosition = container.scrollTop;
      const distance = targetPosition - startPosition;
      const startTime = performance.now();

      setState((prev) => ({ ...prev, isScrolling: true }));

      const animateScroll = (currentTime: number) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);

        container.scrollTop = startPosition + distance * easeOut;

        if (progress < 1) {
          animationFrameRef.current = requestAnimationFrame(animateScroll);
        } else {
          setState((prev) => ({ ...prev, isScrolling: false }));
          animationFrameRef.current = null;
        }
      };

      animationFrameRef.current = requestAnimationFrame(animateScroll);
    },
    [smoothScrollDuration],
  );

  // Scroll to bottom
  const scrollToBottom = useCallback(
    (smooth: boolean = true) => {
      const container = containerRef.current;
      if (!container) return;

      const targetPosition = container.scrollHeight - container.clientHeight;

      if (smooth) {
        smoothScrollTo(container, targetPosition);
      } else {
        container.scrollTop = targetPosition;
      }

      setState((prev) => ({
        ...prev,
        isAtBottom: true,
        scrollProgress: 1,
        userOverrideActive: false,
      }));
    },
    [containerRef, smoothScrollTo],
  );

  // Scroll to top
  const scrollToTop = useCallback(
    (smooth: boolean = true) => {
      const container = containerRef.current;
      if (!container) return;

      if (smooth) {
        smoothScrollTo(container, 0);
      } else {
        container.scrollTop = 0;
      }

      setState((prev) => ({
        ...prev,
        isAtBottom: false,
        scrollProgress: 0,
      }));
    },
    [containerRef, smoothScrollTo],
  );

  // Scroll to specific element
  const scrollToElement = useCallback(
    (element: HTMLElement, smooth: boolean = true) => {
      const container = containerRef.current;
      if (!container) return;

      const elementTop = element.offsetTop;
      const containerTop = container.scrollTop;
      const containerHeight = container.clientHeight;

      // Center the element in the viewport
      const targetPosition = elementTop - containerHeight / 2;

      if (smooth) {
        smoothScrollTo(container, Math.max(0, targetPosition));
      } else {
        container.scrollTop = Math.max(0, targetPosition);
      }
    },
    [containerRef, smoothScrollTo],
  );

  // Enable auto-scroll
  const enableAutoScroll = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isAutoScrollEnabled: true,
      userOverrideActive: false,
    }));

    // Clear override timeout
    if (overrideTimeoutRef.current) {
      clearTimeout(overrideTimeoutRef.current);
      overrideTimeoutRef.current = null;
    }

    onAutoScrollResume?.();
  }, [onAutoScrollResume]);

  // Disable auto-scroll
  const disableAutoScroll = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isAutoScrollEnabled: false,
      userOverrideActive: true,
      lastUserScrollTime: Date.now(),
    }));

    onUserOverride?.();
  }, [onUserOverride]);

  // Toggle auto-scroll
  const toggleAutoScroll = useCallback(() => {
    setState((prev) => {
      if (prev.isAutoScrollEnabled) {
        disableAutoScroll();
        return prev;
      } else {
        enableAutoScroll();
        return prev;
      }
    });
  }, [enableAutoScroll, disableAutoScroll]);

  // Jump to latest (force scroll to bottom and enable auto-scroll)
  const jumpToLatest = useCallback(() => {
    enableAutoScroll();
    scrollToBottom(true);
  }, [enableAutoScroll, scrollToBottom]);

  // Handle scroll events
  const handleScroll = useCallback(
    debounce(() => {
      const container = containerRef.current;
      if (!container || state.isScrolling) return;

      const currentScrollTop = container.scrollTop;
      const atBottom = isAtBottom(container);
      const progress = getScrollProgress(container);

      // Detect user scrolling (not programmatic)
      const isUserInitiated =
        !state.isScrolling &&
        Math.abs(currentScrollTop - lastScrollTopRef.current) > 5;

      if (isUserInitiated && enableUserOverride) {
        isUserScrollingRef.current = true;

        // If user scrolled away from bottom, disable auto-scroll
        if (!atBottom && state.isAutoScrollEnabled) {
          setState((prev) => ({
            ...prev,
            isAutoScrollEnabled: false,
            userOverrideActive: true,
            lastUserScrollTime: Date.now(),
          }));

          onUserOverride?.();

          // Set timeout to re-enable auto-scroll
          if (overrideTimeoutRef.current) {
            clearTimeout(overrideTimeoutRef.current);
          }

          overrideTimeoutRef.current = setTimeout(() => {
            setState((prev) => ({
              ...prev,
              isAutoScrollEnabled: true,
              userOverrideActive: false,
            }));
            onAutoScrollResume?.();
          }, overrideTimeout);
        }

        // If user scrolled back to bottom, re-enable auto-scroll
        if (atBottom && !state.isAutoScrollEnabled) {
          enableAutoScroll();
        }
      }

      setState((prev) => ({
        ...prev,
        isAtBottom: atBottom,
        scrollProgress: progress,
      }));

      lastScrollTopRef.current = currentScrollTop;

      // Clear user scrolling flag after a delay
      setTimeout(() => {
        isUserScrollingRef.current = false;
      }, 100);
    }, 50),
    [
      containerRef,
      isAtBottom,
      getScrollProgress,
      enableUserOverride,
      state.isScrolling,
      state.isAutoScrollEnabled,
      overrideTimeout,
      enableAutoScroll,
      onUserOverride,
      onAutoScrollResume,
    ],
  );

  // Auto-scroll when dependencies change
  useEffect(() => {
    if (
      state.isAutoScrollEnabled &&
      state.isAtBottom &&
      !isUserScrollingRef.current
    ) {
      // Small delay to ensure DOM has updated
      setTimeout(() => {
        scrollToBottom(true);
      }, 10);
    }
  }, [...dependencies, state.isAutoScrollEnabled]);

  // Set up scroll listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      container.removeEventListener("scroll", handleScroll);
    };
  }, [handleScroll, containerRef]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!enableKeyboardShortcuts) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Only handle if no input elements are focused
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      switch (event.key) {
        case "End":
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            jumpToLatest();
          }
          break;
        case "Home":
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            scrollToTop(true);
          }
          break;
        case " ":
          if (event.shiftKey) {
            event.preventDefault();
            toggleAutoScroll();
          }
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [enableKeyboardShortcuts, jumpToLatest, scrollToTop, toggleAutoScroll]);

  // Notify state changes
  useEffect(() => {
    onScrollStateChange?.(state.isAutoScrollEnabled, state.isAtBottom);
  }, [state.isAutoScrollEnabled, state.isAtBottom, onScrollStateChange]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      if (overrideTimeoutRef.current) {
        clearTimeout(overrideTimeoutRef.current);
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return {
    scrollToBottom,
    scrollToTop,
    scrollToElement,
    enableAutoScroll,
    disableAutoScroll,
    toggleAutoScroll,
    jumpToLatest,
    state,
  };
}
