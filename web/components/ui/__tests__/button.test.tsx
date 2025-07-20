import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "../button";

describe("Button Component", () => {
  describe("Rendering", () => {
    it("renders with default props", () => {
      render(<Button>Click me</Button>);
      const button = screen.getByRole("button", { name: /click me/i });
      expect(button).toBeInTheDocument();
      expect(button).toHaveClass("bg-primary");
    });

    it("renders with different variants", () => {
      const variants = ["default", "destructive", "outline", "secondary", "ghost", "link"] as const;

      variants.forEach((variant) => {
        const { unmount } = render(<Button variant={variant}>{variant} button</Button>);
        const button = screen.getByRole("button", { name: new RegExp(variant, "i") });

        switch (variant) {
          case "destructive":
            expect(button).toHaveClass("bg-destructive");
            break;
          case "outline":
            expect(button).toHaveClass("border");
            break;
          case "secondary":
            expect(button).toHaveClass("bg-secondary");
            break;
          case "ghost":
            expect(button).toHaveClass("hover:bg-accent");
            break;
          case "link":
            expect(button).toHaveClass("text-primary");
            break;
        }
        unmount();
      });
    });

    it("renders with different sizes", () => {
      const sizes = ["default", "sm", "lg", "icon"] as const;

      sizes.forEach((size) => {
        const { unmount } = render(
          <Button size={size}>{size === "icon" ? "×" : `${size} button`}</Button>,
        );
        const button = screen.getByRole("button");

        switch (size) {
          case "sm":
            expect(button).toHaveClass("h-9");
            break;
          case "lg":
            expect(button).toHaveClass("h-11");
            break;
          case "icon":
            expect(button).toHaveClass("h-10", "w-10");
            break;
          default:
            expect(button).toHaveClass("h-10");
        }
        unmount();
      });
    });

    it("applies custom className", () => {
      render(<Button className="custom-class">Custom</Button>);
      const button = screen.getByRole("button");
      expect(button).toHaveClass("custom-class");
    });
  });

  describe("Interactions", () => {
    it("handles click events", async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();

      render(<Button onClick={handleClick}>Click me</Button>);
      const button = screen.getByRole("button");

      await user.click(button);
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it("prevents clicks when disabled", async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();

      render(
        <Button onClick={handleClick} disabled>
          Disabled
        </Button>,
      );
      const button = screen.getByRole("button");

      expect(button).toBeDisabled();
      expect(button).toHaveClass("disabled:opacity-50");

      await user.click(button);
      expect(handleClick).not.toHaveBeenCalled();
    });

    it("supports keyboard navigation", async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();

      render(<Button onClick={handleClick}>Keyboard</Button>);
      const button = screen.getByRole("button");

      // Tab to button
      await user.tab();
      expect(button).toHaveFocus();

      // Press Enter
      await user.keyboard("{Enter}");
      expect(handleClick).toHaveBeenCalledTimes(1);

      // Press Space
      await user.keyboard(" ");
      expect(handleClick).toHaveBeenCalledTimes(2);
    });
  });

  describe("Accessibility", () => {
    it("has correct ARIA attributes", () => {
      render(
        <Button aria-label="Save document" aria-pressed="true">
          Save
        </Button>,
      );
      const button = screen.getByRole("button");

      expect(button).toHaveAttribute("aria-label", "Save document");
      expect(button).toHaveAttribute("aria-pressed", "true");
    });

    it("supports focus visible styling", () => {
      render(<Button>Focus me</Button>);
      const button = screen.getByRole("button");

      expect(button).toHaveClass("focus-visible:ring-2");
      expect(button).toHaveClass("focus-visible:ring-offset-2");
    });

    it("announces state changes to screen readers", () => {
      const { rerender } = render(<Button>Loading</Button>);
      const button = screen.getByRole("button");

      expect(button).not.toHaveAttribute("aria-busy");

      rerender(<Button aria-busy="true">Loading</Button>);
      expect(button).toHaveAttribute("aria-busy", "true");
    });
  });

  describe("Ref forwarding", () => {
    it("forwards ref to button element", () => {
      const ref = React.createRef<HTMLButtonElement>();
      render(<Button ref={ref}>Ref button</Button>);

      expect(ref.current).toBeInstanceOf(HTMLButtonElement);
      expect(ref.current?.tagName).toBe("BUTTON");
    });
  });

  describe("Type safety", () => {
    it("accepts all valid HTML button props", () => {
      render(
        <Button type="submit" form="my-form" name="action" value="save" autoFocus>
          Submit
        </Button>,
      );

      const button = screen.getByRole("button");
      expect(button).toHaveAttribute("type", "submit");
      expect(button).toHaveAttribute("form", "my-form");
      expect(button).toHaveAttribute("name", "action");
      expect(button).toHaveAttribute("value", "save");
    });
  });
});
