/**
 * @jest-environment jsdom
 */

import { renderHook, act } from "@testing-library/react";
import { useAsyncOperation } from "@/hooks/useAsyncOperation";

// Set up DOM environment
import "@testing-library/jest-dom";

describe("useAsyncOperation Hook Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Initialization", () => {
    it("initializes with correct default state", () => {
      const mockAsyncFn = jest.fn().mockResolvedValue("test data");
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Before async operation completes
      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(true); // Auto-executes on mount
      expect(result.current.error).toBe(null);
      expect(typeof result.current.reset).toBe("function");
      expect(typeof result.current.execute).toBe("function");
    });

    it("automatically executes async function on mount", async () => {
      const mockAsyncFn = jest.fn().mockResolvedValue("auto-executed");
      renderHook(() => useAsyncOperation(mockAsyncFn));

      expect(mockAsyncFn).toHaveBeenCalledTimes(1);
    });
  });

  describe("Successful Operations", () => {
    it("handles successful async operation", async () => {
      const testData = { message: "success" };
      const mockAsyncFn = jest.fn().mockResolvedValue(testData);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait for async operation to complete
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toEqual(testData);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });

    it("handles string return values", async () => {
      const testString = "test string result";
      const mockAsyncFn = jest.fn().mockResolvedValue(testString);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(testString);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });

    it("handles number return values", async () => {
      const testNumber = 42;
      const mockAsyncFn = jest.fn().mockResolvedValue(testNumber);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(testNumber);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });

    it("handles array return values", async () => {
      const testArray = [1, 2, 3, "test"];
      const mockAsyncFn = jest.fn().mockResolvedValue(testArray);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toEqual(testArray);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });

    it("handles null/undefined return values", async () => {
      const mockAsyncFn = jest.fn().mockResolvedValue(null);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });
  });

  describe("Error Handling", () => {
    it("handles async function that throws Error object", async () => {
      const testError = new Error("Test error message");
      const mockAsyncFn = jest.fn().mockRejectedValue(testError);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(testError);
      expect(result.current.error?.message).toBe("Test error message");
    });

    it("handles async function that throws string", async () => {
      const errorString = "String error";
      const mockAsyncFn = jest.fn().mockRejectedValue(errorString);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeInstanceOf(Error);
      expect(result.current.error?.message).toBe("String error");
    });

    it("handles async function that throws non-Error object", async () => {
      const errorObject = { code: 500, message: "Server error" };
      const mockAsyncFn = jest.fn().mockRejectedValue(errorObject);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeInstanceOf(Error);
      expect(result.current.error?.message).toBe("[object Object]");
    });

    it("handles async function that throws null", async () => {
      const mockAsyncFn = jest.fn().mockRejectedValue(null);
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeInstanceOf(Error);
      expect(result.current.error?.message).toBe("null");
    });

    it("clears previous error on successful retry", async () => {
      let shouldFail = true;
      const mockAsyncFn = jest.fn().mockImplementation(() => {
        if (shouldFail) {
          return Promise.reject(new Error("Initial error"));
        }
        return Promise.resolve("Success after retry");
      });

      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait for initial error
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.error).toBeInstanceOf(Error);
      expect(result.current.data).toBe(null);

      // Retry with success
      shouldFail = false;
      await act(async () => {
        result.current.execute();
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.error).toBe(null);
      expect(result.current.data).toBe("Success after retry");
    });
  });

  describe("Manual Execution", () => {
    it("allows manual execution via execute method", async () => {
      let callCount = 0;
      const mockAsyncFn = jest.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve(`Call ${callCount}`);
      });

      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait for auto-execution
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("Call 1");

      // Manual execution
      await act(async () => {
        result.current.execute();
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("Call 2");
      expect(mockAsyncFn).toHaveBeenCalledTimes(2);
    });

    it("sets loading state during manual execution", async () => {
      const mockAsyncFn = jest
        .fn()
        .mockImplementation(
          () =>
            new Promise((resolve) => setTimeout(() => resolve("delayed"), 100)),
        );

      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait for auto-execution to complete
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 150));
      });

      expect(result.current.loading).toBe(false);

      // Start manual execution
      act(() => {
        result.current.execute();
      });

      expect(result.current.loading).toBe(true);

      // Wait for completion
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 150));
      });

      expect(result.current.loading).toBe(false);
    });

    it("handles overlapping executions (last to complete wins)", async () => {
      let resolveFirst: (value: string) => void;
      let resolveSecond: (value: string) => void;

      const firstPromise = new Promise<string>((resolve) => {
        resolveFirst = resolve;
      });

      const secondPromise = new Promise<string>((resolve) => {
        resolveSecond = resolve;
      });

      let callCount = 0;
      const mockAsyncFn = jest.fn().mockImplementation(() => {
        callCount++;
        return callCount === 1 ? firstPromise : secondPromise;
      });

      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait a bit then start second execution before first completes
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
        result.current.execute();
      });

      // Resolve second execution first
      await act(async () => {
        resolveSecond("Second result");
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("Second result");

      // Resolve first execution after second (this will override due to no race protection)
      await act(async () => {
        resolveFirst("First result");
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      // The hook doesn't have race condition protection, so last to complete wins
      expect(result.current.data).toBe("First result");
    });
  });

  describe("Reset Functionality", () => {
    it("resets all state to initial values", async () => {
      const mockAsyncFn = jest.fn().mockResolvedValue("test data");
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait for auto-execution
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("test data");
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);

      // Reset
      act(() => {
        result.current.reset();
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });

    it("resets error state", async () => {
      const mockAsyncFn = jest.fn().mockRejectedValue(new Error("Test error"));
      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Wait for auto-execution to fail
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.error).toBeInstanceOf(Error);

      // Reset
      act(() => {
        result.current.reset();
      });

      expect(result.current.error).toBe(null);
      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
    });

    it("resets loading state", async () => {
      const mockAsyncFn = jest
        .fn()
        .mockImplementation(
          () =>
            new Promise((resolve) => setTimeout(() => resolve("data"), 100)),
        );

      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Let auto-execution start but don't wait for completion
      act(() => {
        // Loading should be true during auto-execution
      });

      expect(result.current.loading).toBe(true);

      // Reset while loading
      act(() => {
        result.current.reset();
      });

      expect(result.current.loading).toBe(false);
      expect(result.current.data).toBe(null);
      expect(result.current.error).toBe(null);
    });
  });

  describe("Function Reference Changes", () => {
    it("re-executes when async function reference changes", async () => {
      const mockAsyncFn1 = jest.fn().mockResolvedValue("first result");
      const mockAsyncFn2 = jest.fn().mockResolvedValue("second result");

      const { result, rerender } = renderHook(
        ({ asyncFn }) => useAsyncOperation(asyncFn),
        { initialProps: { asyncFn: mockAsyncFn1 } },
      );

      // Wait for first execution
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("first result");
      expect(mockAsyncFn1).toHaveBeenCalledTimes(1);

      // Change function reference
      rerender({ asyncFn: mockAsyncFn2 });

      // Wait for second execution
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("second result");
      expect(mockAsyncFn2).toHaveBeenCalledTimes(1);
    });

    it("maintains stable references for reset and execute methods", () => {
      const mockAsyncFn = jest.fn().mockResolvedValue("data");
      const { result, rerender } = renderHook(() =>
        useAsyncOperation(mockAsyncFn),
      );

      const initialReset = result.current.reset;
      const initialExecute = result.current.execute;

      rerender();

      expect(result.current.reset).toBe(initialReset);
      expect(result.current.execute).toBe(initialExecute);
    });
  });

  describe("Edge Cases", () => {
    it("handles sync function wrapped in Promise.resolve", async () => {
      const syncFunction = () => Promise.resolve("sync result");
      const { result } = renderHook(() => useAsyncOperation(syncFunction));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe("sync result");
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe(null);
    });

    it("handles function that returns immediately rejected promise", async () => {
      const syncErrorFunction = () =>
        Promise.reject(new Error("Immediate error"));
      const { result } = renderHook(() => useAsyncOperation(syncErrorFunction));

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error?.message).toBe("Immediate error");
    });

    it("handles very long-running async operations", async () => {
      const longRunningFn = jest
        .fn()
        .mockImplementation(
          () =>
            new Promise((resolve) =>
              setTimeout(() => resolve("long result"), 1000),
            ),
        );

      const { result } = renderHook(() => useAsyncOperation(longRunningFn));

      expect(result.current.loading).toBe(true);
      expect(result.current.data).toBe(null);

      // Don't wait for completion in this test to verify loading state
      expect(result.current.loading).toBe(true);
    });

    it("handles function that throws synchronously", async () => {
      const throwingFunction = () => {
        throw new Error("Sync error");
      };

      // Wrap in Promise to make it async
      const asyncThrowingFunction = () =>
        Promise.resolve().then(throwingFunction);

      const { result } = renderHook(() =>
        useAsyncOperation(asyncThrowingFunction),
      );

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.data).toBe(null);
      expect(result.current.loading).toBe(false);
      expect(result.current.error?.message).toBe("Sync error");
    });
  });

  describe("Component Lifecycle", () => {
    it("handles unmounting during async operation", async () => {
      const mockAsyncFn = jest
        .fn()
        .mockImplementation(
          () =>
            new Promise((resolve) => setTimeout(() => resolve("data"), 100)),
        );

      const { unmount } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Unmount while operation is in progress
      unmount();

      // Should not throw errors
      expect(() => unmount()).not.toThrow();
    });

    it("handles multiple mount/unmount cycles", () => {
      const mockAsyncFn = jest.fn().mockResolvedValue("data");

      for (let i = 0; i < 5; i++) {
        const { unmount } = renderHook(() => useAsyncOperation(mockAsyncFn));
        unmount();
      }

      expect(mockAsyncFn).toHaveBeenCalledTimes(5);
    });
  });

  describe("Performance", () => {
    it("maintains stable method references across re-renders", () => {
      const mockAsyncFn = jest.fn().mockResolvedValue("data");
      const { result, rerender } = renderHook(() =>
        useAsyncOperation(mockAsyncFn),
      );

      const reset1 = result.current.reset;
      const execute1 = result.current.execute;

      // Multiple re-renders
      for (let i = 0; i < 5; i++) {
        rerender();
      }

      expect(result.current.reset).toBe(reset1);
      expect(result.current.execute).toBe(execute1);
    });

    it("handles rapid successive executions", async () => {
      let counter = 0;
      const mockAsyncFn = jest
        .fn()
        .mockImplementation(() => Promise.resolve(`result-${++counter}`));

      const { result } = renderHook(() => useAsyncOperation(mockAsyncFn));

      // Rapid executions
      await act(async () => {
        for (let i = 0; i < 10; i++) {
          result.current.execute();
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      // Should have been called multiple times
      expect(mockAsyncFn).toHaveBeenCalledTimes(11); // 1 auto + 10 manual
      expect(result.current.data).toMatch(/^result-/);
    });
  });
});
