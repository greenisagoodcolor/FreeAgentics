/**
 * useDebounce Hook Smoke Tests
 * Tests the debounce hook functionality
 */

import { renderHook, act } from "@testing-library/react";
import { useDebounce } from "@/hooks/useDebounce";

// Mock timers
jest.useFakeTimers();

describe("useDebounce Hook", () => {
  afterEach(() => {
    jest.clearAllTimers();
  });

  it("returns initial value immediately", () => {
    const { result } = renderHook(() => useDebounce("initial", 500));
    expect(result.current).toBe("initial");
  });

  it("debounces value updates", () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: "initial", delay: 500 },
      },
    );

    expect(result.current).toBe("initial");

    // Update the value
    rerender({ value: "updated", delay: 500 });

    // Value should still be initial immediately after update
    expect(result.current).toBe("initial");

    // Fast forward time by 250ms (less than delay)
    act(() => {
      jest.advanceTimersByTime(250);
    });

    // Value should still be initial
    expect(result.current).toBe("initial");

    // Fast forward time by another 250ms (total 500ms, equal to delay)
    act(() => {
      jest.advanceTimersByTime(250);
    });

    // Now value should be updated
    expect(result.current).toBe("updated");
  });

  it("resets timer on rapid value changes", () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: "initial", delay: 500 },
      },
    );

    // Rapid updates
    rerender({ value: "update1", delay: 500 });

    act(() => {
      jest.advanceTimersByTime(250);
    });

    rerender({ value: "update2", delay: 500 });

    act(() => {
      jest.advanceTimersByTime(250);
    });

    // Should still be initial because timer was reset
    expect(result.current).toBe("initial");

    // Complete the delay
    act(() => {
      jest.advanceTimersByTime(250);
    });

    // Should now be the latest value
    expect(result.current).toBe("update2");
  });

  it("handles different delay values", () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: "initial", delay: 100 },
      },
    );

    rerender({ value: "updated", delay: 100 });

    act(() => {
      jest.advanceTimersByTime(100);
    });

    expect(result.current).toBe("updated");
  });

  it("works with different data types", () => {
    // Test with numbers
    const { result: numberResult, rerender: numberRerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: 0, delay: 500 },
      },
    );

    numberRerender({ value: 42, delay: 500 });

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(numberResult.current).toBe(42);

    // Test with objects
    const { result: objectResult, rerender: objectRerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: { count: 0 }, delay: 500 },
      },
    );

    const newObject = { count: 1 };
    objectRerender({ value: newObject, delay: 500 });

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(objectResult.current).toBe(newObject);
  });
});
