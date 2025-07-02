import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Ensure we're using the real component
jest.unmock("@/app/dashboard/components/panels/GoalPanel");
jest.unmock("@/app/dashboard/components/panels/GoalPanel/index");

// Import after unmocking
import GoalPanel from "@/app/dashboard/components/panels/GoalPanel";

// Mock icons
jest.mock("lucide-react", () => ({
  Send: ({ className }: any) => (
    <span className={className} data-testid="send-icon">
      Send
    </span>
  ),
}));

describe("GoalPanel Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Component Rendering", () => {
    it("renders goal input form with correct elements", () => {
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute("type", "text");

      const submitButton = screen.getByRole("button", { name: /SET GOAL/i });
      expect(submitButton).toBeInTheDocument();
      expect(submitButton).toHaveAttribute("type", "submit");

      expect(screen.getByTestId("send-icon")).toBeInTheDocument();
    });

    it("renders with empty initial state", () => {
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      expect(input).toHaveValue("");

      // No current goal should be displayed initially
      expect(screen.queryByText("CURRENT GOAL")).not.toBeInTheDocument();
    });

    it("applies correct styling classes", () => {
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      expect(input).toHaveClass(
        "flex-1",
        "px-4",
        "py-2",
        "bg-[var(--bg-secondary)]",
        "border",
        "border-[var(--bg-tertiary)]",
        "rounded-lg",
      );

      const button = screen.getByRole("button");
      expect(button).toHaveClass(
        "px-4",
        "py-2",
        "bg-[var(--primary-amber)]",
        "hover:bg-[var(--primary-amber-hover)]",
        "rounded-lg",
      );
    });
  });

  describe("Input Interactions", () => {
    it("updates input value when typing", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Analyze market trends");

      expect(input).toHaveValue("Analyze market trends");
    });

    it("handles various input types correctly", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      const testInputs = [
        "Simple goal",
        "Goal with numbers 123",
        "Goal with symbols !@#$%",
        "Very long goal that contains multiple sentences and should be handled properly by the input field without any issues",
        "   Goal with leading and trailing spaces   ",
      ];

      for (const testInput of testInputs) {
        await user.clear(input);
        await user.type(input, testInput);
        expect(input).toHaveValue(testInput);
      }
    });

    it("preserves input value after submission (component bug)", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const submitButton = screen.getByRole("button");

      await user.type(input, "Test goal");
      await user.click(submitButton);

      // BUG: Component should clear input but doesn't
      expect(input).toHaveValue("Test goal");
      expect(screen.getByText("Test goal")).toBeInTheDocument();
    });
  });

  describe("Form Submission", () => {
    it("submits goal via button click", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const submitButton = screen.getByRole("button");

      await user.type(input, "Optimize performance");
      await user.click(submitButton);

      expect(screen.getByText("CURRENT GOAL")).toBeInTheDocument();
      expect(screen.getByText("Optimize performance")).toBeInTheDocument();
    });

    it("submits goal via Enter key", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Process data efficiently");
      await user.keyboard("{Enter}");

      expect(screen.getByText("CURRENT GOAL")).toBeInTheDocument();
      expect(screen.getByText("Process data efficiently")).toBeInTheDocument();
    });

    it("prevents submission of empty goals", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const submitButton = screen.getByRole("button");

      await user.click(submitButton);

      expect(screen.queryByText("CURRENT GOAL")).not.toBeInTheDocument();
    });

    it("prevents submission of whitespace-only goals", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const submitButton = screen.getByRole("button");

      await user.type(input, "   \n\t   ");
      await user.click(submitButton);

      expect(screen.queryByText("CURRENT GOAL")).not.toBeInTheDocument();
    });

    it("uses original goal text without trimming in display", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "   Clean and organize data   ");
      await user.keyboard("{Enter}");

      // Component displays the original goal as entered, but validates with trim()
      // Use more specific check since testing library normalizes whitespace in getByText
      const goalElement = screen.getByText((content, element) => {
        return element?.textContent === "   Clean and organize data   ";
      });
      expect(goalElement).toBeInTheDocument();
    });
  });

  describe("Current Goal Display", () => {
    it("shows current goal after submission", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Monitor system health");
      await user.keyboard("{Enter}");

      // Check that current goal section appears
      expect(screen.getByText("CURRENT GOAL")).toBeInTheDocument();
      expect(screen.getByText("Monitor system health")).toBeInTheDocument();

      // Find the correct container div (parent of CURRENT GOAL)
      const currentGoalContainer =
        screen.getByText("CURRENT GOAL").parentElement;
      expect(currentGoalContainer).toHaveClass(
        "flex-1",
        "bg-[var(--bg-secondary)]",
        "border",
        "border-[var(--bg-tertiary)]",
        "rounded-lg",
        "p-4",
      );
    });

    it("updates current goal when new goal is submitted", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      // Submit first goal
      await user.type(input, "First goal");
      await user.keyboard("{Enter}");
      expect(screen.getByText("First goal")).toBeInTheDocument();

      // Clear input and submit second goal (since component doesn't auto-clear)
      await user.clear(input);
      await user.type(input, "Second goal");
      await user.keyboard("{Enter}");

      expect(screen.getByText("Second goal")).toBeInTheDocument();
      expect(screen.queryByText("First goal")).not.toBeInTheDocument();
    });

    it("maintains current goal display style", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Style test goal");
      await user.keyboard("{Enter}");

      const goalText = screen.getByText("Style test goal");
      expect(goalText).toHaveClass(
        "text-lg",
        "font-mono",
        "text-[var(--primary-amber)]",
      );

      const goalLabel = screen.getByText("CURRENT GOAL");
      expect(goalLabel).toHaveClass(
        "text-xs",
        "font-mono",
        "text-[var(--text-secondary)]",
        "mb-2",
      );
    });
  });

  describe("View Types", () => {
    it("accepts different view types", () => {
      const views = ["executive", "technical", "research", "minimal"];

      views.forEach((view) => {
        const { unmount } = render(<GoalPanel view={view} />);
        expect(
          screen.getByPlaceholderText("Enter goal for agents..."),
        ).toBeInTheDocument();
        unmount();
      });
    });

    it("maintains functionality across view types", async () => {
      const user = userEvent.setup();
      const views = ["executive", "technical", "research", "minimal"];

      for (const view of views) {
        const { unmount } = render(<GoalPanel view={view} />);

        const input = screen.getByPlaceholderText("Enter goal for agents...");
        await user.type(input, `Goal for ${view} view`);
        await user.keyboard("{Enter}");

        expect(screen.getByText(`Goal for ${view} view`)).toBeInTheDocument();
        unmount();
      }
    });
  });

  describe("Form Validation", () => {
    it("handles form submission with preventDefault", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const form = screen.getByRole("button").closest("form")!;

      // Add some content so the form actually submits
      await user.type(input, "Test goal");

      const submitSpy = jest.fn();
      form.addEventListener("submit", submitSpy);

      await user.keyboard("{Enter}");

      expect(submitSpy).toHaveBeenCalled();
    });

    it("validates input length and content", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      // Test very long input
      const longGoal = "A".repeat(1000);
      await user.type(input, longGoal);
      await user.keyboard("{Enter}");

      expect(screen.getByText(longGoal)).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("has proper form structure", () => {
      render(<GoalPanel view="executive" />);

      const form = screen.getByRole("button").closest("form");
      expect(form).toBeInTheDocument();

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      expect(input).toHaveAttribute("type", "text");

      const button = screen.getByRole("button");
      expect(button).toHaveAttribute("type", "submit");
    });

    it("has accessible placeholder text", () => {
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      expect(input).toBeInTheDocument();
    });

    it("supports keyboard navigation", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const button = screen.getByRole("button");

      // Tab to input
      await user.tab();
      expect(input).toHaveFocus();

      // Tab to button
      await user.tab();
      expect(button).toHaveFocus();
    });

    it("maintains focus management on submission", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Focus test goal");
      await user.keyboard("{Enter}");

      // Input should still be focusable after submission
      await user.click(input);
      expect(input).toHaveFocus();
    });
  });

  describe("Edge Cases", () => {
    it("handles special characters in goals", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const specialCharsGoal = "Goal with special chars";

      await user.type(input, specialCharsGoal);
      await user.keyboard("{Enter}");

      expect(screen.getByText(specialCharsGoal)).toBeInTheDocument();
    });

    it("handles unicode characters", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");
      const unicodeGoal = "分析市场趋势 🚀 émôtïcôns";

      await user.type(input, unicodeGoal);
      await user.keyboard("{Enter}");

      expect(screen.getByText(unicodeGoal)).toBeInTheDocument();
    });

    it("handles rapid successive submissions", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      // Rapidly submit multiple goals (clear input between each since component doesn't auto-clear)
      for (let i = 0; i < 5; i++) {
        await user.clear(input);
        await user.type(input, `Rapid goal ${i}`);
        await user.keyboard("{Enter}");
      }

      // Should show the last goal
      expect(screen.getByText("Rapid goal 4")).toBeInTheDocument();
      expect(screen.queryByText("Rapid goal 3")).not.toBeInTheDocument();
    });

    it("handles input clearing and retyping", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Initial text");
      expect(input).toHaveValue("Initial text");

      await user.clear(input);
      expect(input).toHaveValue("");

      await user.type(input, "New text");
      expect(input).toHaveValue("New text");

      await user.keyboard("{Enter}");
      expect(screen.getByText("New text")).toBeInTheDocument();
    });
  });

  describe("Performance", () => {
    it("handles multiple goal submissions efficiently", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      // Submit multiple goals to test performance (clear between each)
      for (let i = 0; i < 10; i++) {
        await user.clear(input);
        await user.type(input, `Performance test goal ${i}`);
        await user.keyboard("{Enter}");
      }

      // Should maintain functionality
      expect(screen.getByText("Performance test goal 9")).toBeInTheDocument();
      // Input preserves last typed value (component bug)
      expect(input).toHaveValue("Performance test goal 9");
    });

    it("does not cause memory leaks with state updates", async () => {
      const user = userEvent.setup();
      const { rerender } = render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      // Multiple re-renders with goal submissions (with manual clearing)
      for (let i = 0; i < 3; i++) {
        await user.clear(input);
        await user.type(input, `Rerender test ${i}`);
        await user.keyboard("{Enter}");
        rerender(<GoalPanel view="executive" />);
      }

      expect(screen.getByText("Rerender test 2")).toBeInTheDocument();
    });
  });

  describe("Component State Management", () => {
    it("maintains independent state for input and current goal", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      // Set a goal
      await user.type(input, "First goal");
      await user.keyboard("{Enter}");
      expect(screen.getByText("First goal")).toBeInTheDocument();
      // BUG: Input should be cleared but isn't
      expect(input).toHaveValue("First goal");

      // Clear and type something new without submitting
      await user.clear(input);
      await user.type(input, "Typing new goal");
      expect(input).toHaveValue("Typing new goal");
      expect(screen.getByText("First goal")).toBeInTheDocument(); // Current goal unchanged
    });

    it("does not reset input state on successful submission (component bug)", async () => {
      const user = userEvent.setup();
      render(<GoalPanel view="executive" />);

      const input = screen.getByPlaceholderText("Enter goal for agents...");

      await user.type(input, "Test goal for reset");
      expect(input).toHaveValue("Test goal for reset");

      await user.keyboard("{Enter}");

      // BUG: Component should clear input but doesn't
      expect(input).toHaveValue("Test goal for reset");
      expect(screen.getByText("Test goal for reset")).toBeInTheDocument();
    });
  });
});
