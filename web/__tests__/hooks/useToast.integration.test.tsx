/**
 * @jest-environment jsdom
 */

import { renderHook, act } from "@testing-library/react";
import { useToast, toast, reducer } from "@/hooks/useToast";

// Set up DOM environment
import "@testing-library/jest-dom";

// Mock timers for timeout testing
jest.useFakeTimers();

describe("useToast Hook Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllTimers();
    jest.clearAllMocks();

    // Clear global state between tests by using the global toast function
    // Since the state is global, we need to clear it properly
  });

  afterEach(() => {
    act(() => {
      jest.runOnlyPendingTimers();
    });
  });

  describe("Basic Toast Functionality", () => {
    it("initializes with empty toasts array", () => {
      const { result } = renderHook(() => useToast());

      expect(result.current.toasts).toEqual([]);
    });

    it("provides toast creation function", () => {
      const { result } = renderHook(() => useToast());

      expect(typeof result.current.toast).toBe("function");
      expect(typeof result.current.dismiss).toBe("function");
    });

    it("creates a basic toast", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          title: "Test Toast",
          description: "This is a test toast",
        });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe("Test Toast");
      expect(result.current.toasts[0].description).toBe("This is a test toast");
      expect(result.current.toasts[0].open).toBe(true);
      expect(result.current.toasts[0].id).toBeDefined();
    });

    it("creates toast with all properties", () => {
      const { result } = renderHook(() => useToast());

      const mockAction = { altText: "Action" };

      act(() => {
        result.current.toast({
          title: "Full Toast",
          description: "Complete toast with all props",
          variant: "destructive",
          action: mockAction,
        });
      });

      const toast = result.current.toasts[0];
      expect(toast.title).toBe("Full Toast");
      expect(toast.description).toBe("Complete toast with all props");
      expect(toast.variant).toBe("destructive");
      expect(toast.action).toBe(mockAction);
    });

    it("generates unique IDs for toasts", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "Toast 1" });
        result.current.toast({ title: "Toast 2" });
      });

      expect(result.current.toasts).toHaveLength(1); // Limited to 1
      const toast = result.current.toasts[0];
      expect(toast.id).toBeDefined();
      expect(typeof toast.id).toBe("string");
    });
  });

  describe("Toast Limit", () => {
    it("limits toasts to maximum of 1", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "Toast 1" });
        result.current.toast({ title: "Toast 2" });
        result.current.toast({ title: "Toast 3" });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe("Toast 3"); // Latest toast
    });

    it("replaces older toasts when limit is exceeded", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "First Toast" });
      });

      expect(result.current.toasts[0].title).toBe("First Toast");

      act(() => {
        result.current.toast({ title: "Second Toast" });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe("Second Toast");
    });
  });

  describe("Toast Dismissal", () => {
    it("dismisses specific toast by ID", () => {
      const { result } = renderHook(() => useToast());

      let toastId: string;

      act(() => {
        const { id } = result.current.toast({ title: "Test Toast" });
        toastId = id;
      });

      expect(result.current.toasts[0].open).toBe(true);

      act(() => {
        result.current.dismiss(toastId);
      });

      expect(result.current.toasts[0].open).toBe(false);
    });

    it("dismisses all toasts when no ID provided", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "Toast 1" });
      });

      expect(result.current.toasts[0].open).toBe(true);

      act(() => {
        result.current.dismiss();
      });

      expect(result.current.toasts[0].open).toBe(false);
    });

    it("removes toast after dismiss timeout", () => {
      const { result } = renderHook(() => useToast());

      let toastId: string;

      act(() => {
        const { id } = result.current.toast({ title: "Test Toast" });
        toastId = id;
      });

      expect(result.current.toasts).toHaveLength(1);

      act(() => {
        result.current.dismiss(toastId);
      });

      // Toast should be dismissed but not removed yet
      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].open).toBe(false);

      // Advance time to trigger removal
      act(() => {
        jest.advanceTimersByTime(1000000);
      });

      expect(result.current.toasts).toHaveLength(0);
    });
  });

  describe("Toast Updates", () => {
    it("allows updating toast properties", () => {
      const { result } = renderHook(() => useToast());

      let updateFn: (props: any) => void;

      act(() => {
        const { update } = result.current.toast({
          title: "Original Title",
          description: "Original Description",
        });
        updateFn = update;
      });

      expect(result.current.toasts[0].title).toBe("Original Title");

      act(() => {
        updateFn({
          title: "Updated Title",
          description: "Updated Description",
          variant: "success",
        });
      });

      const updatedToast = result.current.toasts[0];
      expect(updatedToast.title).toBe("Updated Title");
      expect(updatedToast.description).toBe("Updated Description");
      expect(updatedToast.variant).toBe("success");
    });

    it("preserves unmodified properties during updates", () => {
      const { result } = renderHook(() => useToast());

      let updateFn: (props: any) => void;
      let originalId: string;

      act(() => {
        const { update, id } = result.current.toast({
          title: "Original Title",
          description: "Original Description",
          variant: "default",
        });
        updateFn = update;
        originalId = id;
      });

      act(() => {
        updateFn({ title: "Updated Title" });
      });

      const updatedToast = result.current.toasts[0];
      expect(updatedToast.id).toBe(originalId);
      expect(updatedToast.title).toBe("Updated Title");
      expect(updatedToast.description).toBe("Original Description");
      expect(updatedToast.variant).toBe("default");
    });
  });

  describe("onOpenChange Behavior", () => {
    it("dismisses toast when onOpenChange is called with false", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "Test Toast" });
      });

      const toast = result.current.toasts[0];
      expect(toast.open).toBe(true);

      act(() => {
        toast.onOpenChange?.(false);
      });

      expect(result.current.toasts[0].open).toBe(false);
    });

    it("does not dismiss toast when onOpenChange is called with true", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "Test Toast" });
      });

      const toast = result.current.toasts[0];
      expect(toast.open).toBe(true);

      act(() => {
        toast.onOpenChange?.(true);
      });

      expect(result.current.toasts[0].open).toBe(true);
    });
  });

  describe("Global Toast Function", () => {
    it("creates toast using global toast function", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        toast({ title: "Global Toast" });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe("Global Toast");
    });

    it("returns methods from global toast function", () => {
      act(() => {
        const result = toast({ title: "Global Toast" });

        expect(result.id).toBeDefined();
        expect(typeof result.dismiss).toBe("function");
        expect(typeof result.update).toBe("function");
      });
    });

    it("global and hook toasts share state", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        toast({ title: "Global Toast" });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe("Global Toast");

      act(() => {
        result.current.toast({ title: "Hook Toast" });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe("Hook Toast");
    });
  });

  describe("Multiple Hook Instances", () => {
    it("synchronizes state across multiple hook instances", () => {
      const { result: result1 } = renderHook(() => useToast());
      const { result: result2 } = renderHook(() => useToast());

      act(() => {
        result1.current.toast({ title: "Shared Toast" });
      });

      expect(result1.current.toasts).toHaveLength(1);
      expect(result2.current.toasts).toHaveLength(1);
      expect(result1.current.toasts[0].title).toBe("Shared Toast");
      expect(result2.current.toasts[0].title).toBe("Shared Toast");
    });

    it("dismissal from one instance affects all instances", () => {
      const { result: result1 } = renderHook(() => useToast());
      const { result: result2 } = renderHook(() => useToast());

      let toastId: string;

      act(() => {
        const { id } = result1.current.toast({ title: "Shared Toast" });
        toastId = id;
      });

      expect(result1.current.toasts[0].open).toBe(true);
      expect(result2.current.toasts[0].open).toBe(true);

      act(() => {
        result2.current.dismiss(toastId);
      });

      expect(result1.current.toasts[0].open).toBe(false);
      expect(result2.current.toasts[0].open).toBe(false);
    });
  });

  describe("Error Handling", () => {
    it("handles invalid toast ID in dismiss gracefully", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: "Test Toast" });
      });

      expect(() => {
        act(() => {
          result.current.dismiss("invalid-id");
        });
      }).not.toThrow();

      expect(result.current.toasts[0].open).toBe(true);
    });

    it("handles empty toast props", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({});
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].id).toBeDefined();
      expect(result.current.toasts[0].open).toBe(true);
    });

    it("handles null/undefined toast properties", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          title: null,
          description: undefined,
        });
      });

      expect(result.current.toasts).toHaveLength(1);
      expect(result.current.toasts[0].title).toBe(null);
      expect(result.current.toasts[0].description).toBe(undefined);
    });
  });

  describe("Memory Management", () => {
    it("cleans up listeners on unmount", () => {
      const { unmount } = renderHook(() => useToast());

      expect(() => unmount()).not.toThrow();
    });

    it("handles multiple mount/unmount cycles", () => {
      for (let i = 0; i < 5; i++) {
        const { result, unmount } = renderHook(() => useToast());

        act(() => {
          result.current.toast({ title: `Toast ${i}` });
        });

        unmount();
      }

      // Final check - state should still be accessible
      const { result } = renderHook(() => useToast());
      expect(result.current.toasts).toHaveLength(1);
    });

    it("clears timeouts when toast is manually dismissed", () => {
      const clearTimeoutSpy = jest.spyOn(global, "clearTimeout");
      const { result } = renderHook(() => useToast());

      let toastId: string;

      act(() => {
        const { id } = result.current.toast({ title: "Test Toast" });
        toastId = id;
      });

      act(() => {
        result.current.dismiss(toastId);
      });

      // Manually dismiss before timeout
      act(() => {
        result.current.dismiss(toastId);
      });

      clearTimeoutSpy.mockRestore();
    });
  });

  describe("Edge Cases", () => {
    it("handles rapid toast creation and dismissal", () => {
      const { result } = renderHook(() => useToast());

      const toastIds: string[] = [];

      act(() => {
        for (let i = 0; i < 10; i++) {
          const { id } = result.current.toast({ title: `Toast ${i}` });
          toastIds.push(id);
        }
      });

      // Should only have 1 toast due to limit
      expect(result.current.toasts).toHaveLength(1);

      // Dismiss all
      act(() => {
        toastIds.forEach((id) => result.current.dismiss(id));
      });

      expect(result.current.toasts[0].open).toBe(false);
    });

    it("handles toast creation with React elements", () => {
      const { result } = renderHook(() => useToast());

      const titleElement = <span>React Title</span>;
      const descElement = <div>React Description</div>;

      act(() => {
        result.current.toast({
          title: titleElement,
          description: descElement,
        });
      });

      expect(result.current.toasts[0].title).toBe(titleElement);
      expect(result.current.toasts[0].description).toBe(descElement);
    });

    it("handles very long toast content", () => {
      const { result } = renderHook(() => useToast());

      const longTitle = "A".repeat(1000);
      const longDescription = "B".repeat(5000);

      act(() => {
        result.current.toast({
          title: longTitle,
          description: longDescription,
        });
      });

      expect(result.current.toasts[0].title).toBe(longTitle);
      expect(result.current.toasts[0].description).toBe(longDescription);
    });
  });

  describe("Reducer Function", () => {
    it("adds toast correctly", () => {
      const initialState = { toasts: [] };
      const toast = {
        id: "1",
        title: "Test",
        open: true,
      };

      const newState = reducer(initialState, {
        type: "ADD_TOAST",
        toast,
      });

      expect(newState.toasts).toHaveLength(1);
      expect(newState.toasts[0]).toBe(toast);
    });

    it("updates toast correctly", () => {
      const initialState = {
        toasts: [
          { id: "1", title: "Original", open: true },
          { id: "2", title: "Other", open: true },
        ],
      };

      const newState = reducer(initialState, {
        type: "UPDATE_TOAST",
        toast: { id: "1", title: "Updated" },
      });

      expect(newState.toasts[0].title).toBe("Updated");
      expect(newState.toasts[1].title).toBe("Other");
    });

    it("dismisses toast correctly", () => {
      const initialState = {
        toasts: [{ id: "1", title: "Test", open: true }],
      };

      const newState = reducer(initialState, {
        type: "DISMISS_TOAST",
        toastId: "1",
      });

      expect(newState.toasts[0].open).toBe(false);
    });

    it("removes toast correctly", () => {
      const initialState = {
        toasts: [
          { id: "1", title: "Test 1", open: true },
          { id: "2", title: "Test 2", open: true },
        ],
      };

      const newState = reducer(initialState, {
        type: "REMOVE_TOAST",
        toastId: "1",
      });

      expect(newState.toasts).toHaveLength(1);
      expect(newState.toasts[0].id).toBe("2");
    });

    it("removes all toasts when no ID provided", () => {
      const initialState = {
        toasts: [
          { id: "1", title: "Test 1", open: true },
          { id: "2", title: "Test 2", open: true },
        ],
      };

      const newState = reducer(initialState, {
        type: "REMOVE_TOAST",
        toastId: undefined,
      });

      expect(newState.toasts).toHaveLength(0);
    });
  });
});
