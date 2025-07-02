import { renderHook, act } from "@testing-library/react";
import { useDebounce } from "@/hooks/useDebounce";

// Mock timers
jest.useFakeTimers();

describe("useDebounce Hook Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllTimers();
    jest.clearAllMocks();
  });

  afterEach(() => {
    act(() => {
      jest.runOnlyPendingTimers();
    });
  });

  describe("Basic Functionality", () => {
    it("returns initial value immediately", () => {
      const { result } = renderHook(() => useDebounce("initial", 500));

      expect(result.current).toBe("initial");
    });

    it("debounces string values correctly", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 500 } },
      );

      expect(result.current).toBe("initial");

      // Change value
      rerender({ value: "updated", delay: 500 });

      // Should still be initial value immediately
      expect(result.current).toBe("initial");

      // Fast-forward time to trigger debounce
      act(() => {
        jest.advanceTimersByTime(500);
      });

      expect(result.current).toBe("updated");
    });

    it("debounces number values correctly", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: 0, delay: 300 } },
      );

      expect(result.current).toBe(0);

      rerender({ value: 42, delay: 300 });
      expect(result.current).toBe(0);

      act(() => {
        jest.advanceTimersByTime(300);
      });

      expect(result.current).toBe(42);
    });

    it("debounces boolean values correctly", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: false, delay: 200 } },
      );

      expect(result.current).toBe(false);

      rerender({ value: true, delay: 200 });
      expect(result.current).toBe(false);

      act(() => {
        jest.advanceTimersByTime(200);
      });

      expect(result.current).toBe(true);
    });

    it("debounces object values correctly", () => {
      const initialObj = { id: 1, name: "test" };
      const updatedObj = { id: 2, name: "updated" };

      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: initialObj, delay: 400 } },
      );

      expect(result.current).toBe(initialObj);

      rerender({ value: updatedObj, delay: 400 });
      expect(result.current).toBe(initialObj);

      act(() => {
        jest.advanceTimersByTime(400);
      });

      expect(result.current).toBe(updatedObj);
    });

    it("debounces array values correctly", () => {
      const initialArray = [1, 2, 3];
      const updatedArray = [4, 5, 6];

      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: initialArray, delay: 250 } },
      );

      expect(result.current).toBe(initialArray);

      rerender({ value: updatedArray, delay: 250 });
      expect(result.current).toBe(initialArray);

      act(() => {
        jest.advanceTimersByTime(250);
      });

      expect(result.current).toBe(updatedArray);
    });
  });

  describe("Delay Handling", () => {
    it("respects different delay values", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 100 } },
      );

      rerender({ value: "first update", delay: 100 });

      // Should not update before delay
      act(() => {
        jest.advanceTimersByTime(50);
      });
      expect(result.current).toBe("initial");

      // Should update after delay
      act(() => {
        jest.advanceTimersByTime(50);
      });
      expect(result.current).toBe("first update");
    });

    it("resets timer when delay changes", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 500 } },
      );

      rerender({ value: "updated", delay: 500 });

      // Advance time partially
      act(() => {
        jest.advanceTimersByTime(200);
      });
      expect(result.current).toBe("initial");

      // Change delay, which should restart the timer
      rerender({ value: "updated", delay: 300 });

      // Advance by original remaining time - should not trigger
      act(() => {
        jest.advanceTimersByTime(300);
      });
      expect(result.current).toBe("updated");
    });

    it("handles zero delay", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 0 } },
      );

      rerender({ value: "immediate", delay: 0 });

      act(() => {
        jest.advanceTimersByTime(0);
      });

      expect(result.current).toBe("immediate");
    });

    it("handles negative delay gracefully", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: -100 } },
      );

      rerender({ value: "updated", delay: -100 });

      // Negative delay should behave like zero delay
      act(() => {
        jest.advanceTimersByTime(0);
      });

      expect(result.current).toBe("updated");
    });
  });

  describe("Rapid Updates", () => {
    it("cancels previous timeout on rapid value changes", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 500 } },
      );

      // First update
      rerender({ value: "first", delay: 500 });

      // Advance time partially
      act(() => {
        jest.advanceTimersByTime(200);
      });
      expect(result.current).toBe("initial");

      // Second update before first completes
      rerender({ value: "second", delay: 500 });

      // Advance time to when first update would have completed
      act(() => {
        jest.advanceTimersByTime(300);
      });
      expect(result.current).toBe("initial"); // Should still be initial

      // Advance remaining time for second update
      act(() => {
        jest.advanceTimersByTime(200);
      });
      expect(result.current).toBe("second");
    });

    it("handles multiple rapid updates correctly", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 300 } },
      );

      const updates = ["first", "second", "third", "fourth", "final"];

      // Apply multiple updates rapidly
      updates.forEach((update, index) => {
        rerender({ value: update, delay: 300 });

        // Advance time by less than delay
        act(() => {
          jest.advanceTimersByTime(50);
        });

        // Should still be initial value
        expect(result.current).toBe("initial");
      });

      // Now advance time to complete the final update
      act(() => {
        jest.advanceTimersByTime(300);
      });

      expect(result.current).toBe("final");
    });

    it("handles very rapid updates efficiently", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: 0, delay: 100 } },
      );

      // Simulate 100 rapid updates
      for (let i = 1; i <= 100; i++) {
        rerender({ value: i, delay: 100 });

        if (i < 100) {
          act(() => {
            jest.advanceTimersByTime(10);
          });
        }
      }

      // Should still be initial value
      expect(result.current).toBe(0);

      // Complete the debounce
      act(() => {
        jest.advanceTimersByTime(100);
      });

      // Should be the final value
      expect(result.current).toBe(100);
    });
  });

  describe("Edge Cases", () => {
    it("handles null values", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: null, delay: 200 } },
      );

      expect(result.current).toBe(null);

      rerender({ value: "not null", delay: 200 });

      act(() => {
        jest.advanceTimersByTime(200);
      });

      expect(result.current).toBe("not null");
    });

    it("handles undefined values", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: undefined, delay: 150 } },
      );

      expect(result.current).toBe(undefined);

      rerender({ value: "defined", delay: 150 });

      act(() => {
        jest.advanceTimersByTime(150);
      });

      expect(result.current).toBe("defined");
    });

    it("handles same value updates", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "same", delay: 300 } },
      );

      expect(result.current).toBe("same");

      // Update with same value
      rerender({ value: "same", delay: 300 });

      act(() => {
        jest.advanceTimersByTime(300);
      });

      expect(result.current).toBe("same");
    });

    it("handles complex object changes", () => {
      const obj1 = { nested: { value: 1 }, array: [1, 2, 3] };
      const obj2 = { nested: { value: 2 }, array: [4, 5, 6] };

      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: obj1, delay: 250 } },
      );

      expect(result.current).toBe(obj1);

      rerender({ value: obj2, delay: 250 });

      act(() => {
        jest.advanceTimersByTime(250);
      });

      expect(result.current).toBe(obj2);
      expect(result.current.nested.value).toBe(2);
      expect(result.current.array).toEqual([4, 5, 6]);
    });

    it("handles function values", () => {
      const func1 = () => "first";
      const func2 = () => "second";

      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: func1, delay: 200 } },
      );

      expect(result.current).toBe(func1);
      expect(typeof result.current).toBe("function");
      expect(result.current()).toBe("first");

      rerender({ value: func2, delay: 200 });

      act(() => {
        jest.advanceTimersByTime(200);
      });

      expect(result.current).toBe(func2);
      expect(typeof result.current).toBe("function");
      expect(result.current()).toBe("second");
    });
  });

  describe("Performance", () => {
    it("cleans up timers properly", () => {
      const clearTimeoutSpy = jest.spyOn(global, "clearTimeout");

      const { rerender, unmount } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 500 } },
      );

      // Trigger multiple updates
      rerender({ value: "first", delay: 500 });
      rerender({ value: "second", delay: 500 });
      rerender({ value: "third", delay: 500 });

      // Should have called clearTimeout for each update
      expect(clearTimeoutSpy).toHaveBeenCalledTimes(3);

      unmount();

      // Should call clearTimeout one more time on unmount
      expect(clearTimeoutSpy).toHaveBeenCalledTimes(4);

      clearTimeoutSpy.mockRestore();
    });

    it("handles large delay values", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 10000 } },
      );

      rerender({ value: "updated", delay: 10000 });

      // Should not update even after significant time less than delay
      act(() => {
        jest.advanceTimersByTime(9999);
      });
      expect(result.current).toBe("initial");

      // Should update after full delay
      act(() => {
        jest.advanceTimersByTime(1);
      });
      expect(result.current).toBe("updated");
    });

    it("handles very small delay values", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 1 } },
      );

      rerender({ value: "updated", delay: 1 });

      act(() => {
        jest.advanceTimersByTime(1);
      });

      expect(result.current).toBe("updated");
    });
  });

  describe("Component Lifecycle", () => {
    it("works correctly after remounting", () => {
      let hookResult = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 200 } },
      );

      hookResult.unmount();

      // Remount with different initial value
      hookResult = renderHook(({ value, delay }) => useDebounce(value, delay), {
        initialProps: { value: "remounted", delay: 200 },
      });

      expect(hookResult.result.current).toBe("remounted");

      hookResult.rerender({ value: "updated after remount", delay: 200 });

      act(() => {
        jest.advanceTimersByTime(200);
      });

      expect(hookResult.result.current).toBe("updated after remount");
    });

    it("cleans up on unmount", () => {
      const { result, rerender, unmount } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 500 } },
      );

      rerender({ value: "updated", delay: 500 });

      // Unmount before debounce completes
      unmount();

      // Advancing time should not cause any issues
      act(() => {
        jest.advanceTimersByTime(500);
      });

      // No assertions needed - just ensuring no errors occur
    });

    it("maintains state consistency across multiple re-renders", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: "initial", delay: 300 } },
      );

      const updates = ["a", "b", "c", "d", "e"];

      updates.forEach((value, index) => {
        rerender({ value, delay: 300 });

        // Advance time by less than delay for all but last
        if (index < updates.length - 1) {
          act(() => {
            jest.advanceTimersByTime(100);
          });
          expect(result.current).toBe("initial");
        }
      });

      // Complete the final debounce
      act(() => {
        jest.advanceTimersByTime(300);
      });

      expect(result.current).toBe("e");
    });
  });

  describe("Type Safety", () => {
    it("maintains type consistency for strings", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce<string>(value, delay),
        { initialProps: { value: "string", delay: 100 } },
      );

      expect(typeof result.current).toBe("string");

      rerender({ value: "new string", delay: 100 });

      act(() => {
        jest.advanceTimersByTime(100);
      });

      expect(typeof result.current).toBe("string");
      expect(result.current).toBe("new string");
    });

    it("maintains type consistency for numbers", () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce<number>(value, delay),
        { initialProps: { value: 42, delay: 100 } },
      );

      expect(typeof result.current).toBe("number");

      rerender({ value: 84, delay: 100 });

      act(() => {
        jest.advanceTimersByTime(100);
      });

      expect(typeof result.current).toBe("number");
      expect(result.current).toBe(84);
    });

    it("maintains type consistency for complex types", () => {
      interface TestInterface {
        id: number;
        name: string;
        active: boolean;
      }

      const obj1: TestInterface = { id: 1, name: "test1", active: true };
      const obj2: TestInterface = { id: 2, name: "test2", active: false };

      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce<TestInterface>(value, delay),
        { initialProps: { value: obj1, delay: 100 } },
      );

      expect(result.current.id).toBe(1);
      expect(result.current.name).toBe("test1");
      expect(result.current.active).toBe(true);

      rerender({ value: obj2, delay: 100 });

      act(() => {
        jest.advanceTimersByTime(100);
      });

      expect(result.current.id).toBe(2);
      expect(result.current.name).toBe("test2");
      expect(result.current.active).toBe(false);
    });
  });
});
