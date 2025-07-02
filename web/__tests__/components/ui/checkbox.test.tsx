import "./setup";
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Unmock the checkbox component for this test
jest.unmock("@/components/ui/checkbox");

// Mock Radix UI Checkbox
jest.mock("@radix-ui/react-checkbox", () => {
  const React = require("react");
  return {
    Root: React.forwardRef(
      (
        {
          className,
          checked,
          defaultChecked,
          onCheckedChange,
          disabled,
          required,
          name,
          value,
          children,
          ...props
        }: any,
        ref: any,
      ) => {
        const [isChecked, setIsChecked] = React.useState(
          checked !== undefined ? checked : defaultChecked || false,
        );

        React.useEffect(() => {
          if (checked !== undefined) {
            setIsChecked(checked);
          }
        }, [checked]);

        const handleClick = () => {
          if (!disabled) {
            const newChecked = !isChecked;
            setIsChecked(newChecked);
            if (onCheckedChange) {
              onCheckedChange(newChecked);
            }
          }
        };

        return (
          <button
            ref={ref}
            type="button"
            role="checkbox"
            aria-checked={isChecked}
            aria-disabled={disabled}
            aria-required={required}
            data-state={isChecked ? "checked" : "unchecked"}
            data-disabled={disabled ? "" : undefined}
            disabled={disabled}
            className={className}
            onClick={handleClick}
            {...props}
          >
            {children}
          </button>
        );
      },
    ),
    Indicator: ({ className, children }: any) => {
      // This component is rendered inside the Root component
      // We need to check the parent's state
      return (
        <span
          className={className}
          data-testid="checkbox-indicator"
          style={{ display: "contents" }}
        >
          {children}
        </span>
      );
    },
  };
});

// Mock lucide-react Check icon
jest.mock("lucide-react", () => ({
  Check: () => <span data-testid="check-icon">✓</span>,
}));

import { Checkbox } from "@/components/ui/checkbox";

describe("Checkbox Component", () => {
  describe("Rendering", () => {
    it("renders unchecked by default", () => {
      render(<Checkbox />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toBeInTheDocument();
      expect(checkbox).toHaveAttribute("aria-checked", "false");
      expect(checkbox).toHaveAttribute("data-state", "unchecked");
    });

    it("renders with custom id", () => {
      render(<Checkbox id="terms" />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toHaveAttribute("id", "terms");
    });

    it("renders with default checked state", () => {
      render(<Checkbox defaultChecked />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toHaveAttribute("aria-checked", "true");
      expect(checkbox).toHaveAttribute("data-state", "checked");
    });

    it("renders with controlled checked state", () => {
      const { rerender } = render(<Checkbox checked={false} />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toHaveAttribute("aria-checked", "false");

      rerender(<Checkbox checked={true} />);
      expect(checkbox).toHaveAttribute("aria-checked", "true");
    });
  });

  describe("Interactions", () => {
    it("toggles when clicked", async () => {
      const user = userEvent.setup();
      render(<Checkbox />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox).toHaveAttribute("aria-checked", "false");

      await user.click(checkbox);
      expect(checkbox).toHaveAttribute("aria-checked", "true");

      await user.click(checkbox);
      expect(checkbox).toHaveAttribute("aria-checked", "false");
    });

    it("calls onCheckedChange when toggled", async () => {
      const handleChange = jest.fn();
      const user = userEvent.setup();

      render(<Checkbox onCheckedChange={handleChange} />);
      const checkbox = screen.getByRole("checkbox");

      await user.click(checkbox);
      expect(handleChange).toHaveBeenCalledWith(true);

      await user.click(checkbox);
      expect(handleChange).toHaveBeenCalledWith(false);
    });

    it("supports keyboard interaction", async () => {
      const handleChange = jest.fn();
      const user = userEvent.setup();

      render(<Checkbox onCheckedChange={handleChange} />);
      const checkbox = screen.getByRole("checkbox");

      checkbox.focus();
      expect(checkbox).toHaveFocus();

      await user.keyboard(" ");
      expect(handleChange).toHaveBeenCalledWith(true);
    });
  });

  describe("States", () => {
    it("renders disabled state", () => {
      render(<Checkbox disabled />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox).toBeDisabled();
      expect(checkbox).toHaveAttribute("aria-disabled", "true");
      expect(checkbox).toHaveAttribute("data-disabled", "");
      expect(checkbox.className).toContain("disabled:cursor-not-allowed");
      expect(checkbox.className).toContain("disabled:opacity-50");
    });

    it("prevents interaction when disabled", async () => {
      const handleChange = jest.fn();
      const user = userEvent.setup();

      render(<Checkbox disabled onCheckedChange={handleChange} />);
      const checkbox = screen.getByRole("checkbox");

      await user.click(checkbox);
      expect(handleChange).not.toHaveBeenCalled();
      expect(checkbox).toHaveAttribute("aria-checked", "false");
    });

    it("maintains checked state when disabled", () => {
      render(<Checkbox defaultChecked disabled />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox).toHaveAttribute("aria-checked", "true");
      expect(checkbox).toBeDisabled();
    });

    it("renders required state", () => {
      render(<Checkbox required />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox).toHaveAttribute("aria-required", "true");
    });
  });

  describe("Styling", () => {
    it("applies default styles", () => {
      render(<Checkbox />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox.className).toContain("h-4");
      expect(checkbox.className).toContain("w-4");
      expect(checkbox.className).toContain("rounded-sm");
      expect(checkbox.className).toContain("border");
      expect(checkbox.className).toContain("border-primary");
    });

    it("applies custom className", () => {
      render(<Checkbox className="custom-checkbox-class" />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox.className).toContain("custom-checkbox-class");
      expect(checkbox.className).toContain("h-4"); // Still has default classes
    });

    it("applies focus styles", async () => {
      const user = userEvent.setup();
      render(<Checkbox />);
      const checkbox = screen.getByRole("checkbox");

      await user.tab();
      expect(checkbox).toHaveFocus();
      expect(checkbox.className).toContain("focus-visible:outline-none");
      expect(checkbox.className).toContain("focus-visible:ring-2");
      expect(checkbox.className).toContain("focus-visible:ring-ring");
    });

    it("applies checked state styles", async () => {
      const user = userEvent.setup();
      render(<Checkbox />);
      const checkbox = screen.getByRole("checkbox");

      await user.click(checkbox);
      expect(checkbox.className).toContain("data-[state=checked]:bg-primary");
      expect(checkbox.className).toContain(
        "data-[state=checked]:text-primary-foreground",
      );
    });
  });

  describe("Props and Attributes", () => {
    it("forwards ref correctly", () => {
      const ref = React.createRef<HTMLButtonElement>();
      render(<Checkbox ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLButtonElement);
      expect(ref.current?.getAttribute("role")).toBe("checkbox");
    });

    it("passes through data attributes", () => {
      render(<Checkbox data-testid="my-checkbox" data-custom="value" />);

      const checkbox = screen.getByTestId("my-checkbox");
      expect(checkbox).toHaveAttribute("data-custom", "value");
    });

    it("supports form-related props", () => {
      render(<Checkbox data-value="terms" />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toHaveAttribute("data-value", "terms");
    });
  });

  describe("Indicator", () => {
    it("renders indicator container", () => {
      render(<Checkbox />);

      // The indicator container should always be present
      expect(screen.getByTestId("checkbox-indicator")).toBeInTheDocument();

      // The check icon is inside the indicator
      expect(screen.getByTestId("check-icon")).toBeInTheDocument();
    });

    it("indicator contains check icon", () => {
      render(<Checkbox />);

      const indicator = screen.getByTestId("checkbox-indicator");
      const checkIcon = screen.getByTestId("check-icon");

      // Check icon should be inside the indicator
      expect(indicator).toContainElement(checkIcon);
    });
  });

  describe("Accessibility", () => {
    it("has proper ARIA attributes", () => {
      render(<Checkbox />);
      const checkbox = screen.getByRole("checkbox");

      expect(checkbox).toHaveAttribute("role", "checkbox");
      expect(checkbox).toHaveAttribute("aria-checked");
      expect(checkbox).toHaveAttribute("type", "button");
    });

    it("can be labeled", () => {
      render(
        <div>
          <label htmlFor="terms">
            <Checkbox id="terms" />
            Accept terms and conditions
          </label>
        </div>,
      );

      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toHaveAttribute("id", "terms");

      // Check label association
      const label = screen
        .getByText("Accept terms and conditions")
        .closest("label");
      expect(label).toHaveAttribute("for", "terms");
    });

    it("supports aria-label", () => {
      render(<Checkbox aria-label="Accept terms" />);
      const checkbox = screen.getByRole("checkbox", { name: "Accept terms" });
      expect(checkbox).toBeInTheDocument();
    });

    it("supports aria-describedby", () => {
      render(
        <>
          <Checkbox aria-describedby="terms-description" />
          <span id="terms-description">You must accept the terms</span>
        </>,
      );

      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toHaveAttribute("aria-describedby", "terms-description");
    });
  });

  describe("Form Integration", () => {
    it("works in a form", () => {
      const handleSubmit = jest.fn((e) => e.preventDefault());

      render(
        <form onSubmit={handleSubmit}>
          <Checkbox />
          <button type="submit">Submit</button>
        </form>,
      );

      const checkbox = screen.getByRole("checkbox");
      const submitButton = screen.getByText("Submit");

      expect(checkbox).toBeInTheDocument();
      fireEvent.click(submitButton);
      expect(handleSubmit).toHaveBeenCalled();
    });

    it("maintains state in controlled form", async () => {
      const user = userEvent.setup();
      const ControlledForm = () => {
        const [checked, setChecked] = React.useState(false);

        return (
          <form>
            <Checkbox
              checked={checked}
              onCheckedChange={setChecked}
              data-testid="controlled-checkbox"
            />
            <span data-testid="state">{checked ? "Checked" : "Unchecked"}</span>
          </form>
        );
      };

      render(<ControlledForm />);

      expect(screen.getByTestId("state")).toHaveTextContent("Unchecked");

      await user.click(screen.getByTestId("controlled-checkbox"));
      expect(screen.getByTestId("state")).toHaveTextContent("Checked");
    });
  });
});
