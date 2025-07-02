import "./setup";
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Unmock the button component for this test
jest.unmock("@/components/ui/button");

import { Button } from "@/components/ui/button";

describe("Button Component", () => {
  describe("Rendering", () => {
    it("renders with default props", () => {
      render(<Button>Click me</Button>);
      const button = screen.getByRole("button", { name: "Click me" });
      expect(button).toBeInTheDocument();
      expect(button.className).toContain("bg-primary");
    });

    it("renders children correctly", () => {
      render(<Button>Test Button</Button>);
      expect(screen.getByText("Test Button")).toBeInTheDocument();
    });

    it("renders with icons", () => {
      render(
        <Button>
          <svg data-testid="icon" />
          Button with Icon
        </Button>,
      );
      expect(screen.getByTestId("icon")).toBeInTheDocument();
      expect(screen.getByText("Button with Icon")).toBeInTheDocument();
    });
  });

  describe("Variants", () => {
    it("renders default variant correctly", () => {
      render(<Button variant="default">Button</Button>);
      const button = screen.getByRole("button");
      expect(button).toBeInTheDocument();
      expect(button.className).toContain("bg-primary");
    });

    it("renders destructive variant correctly", () => {
      render(<Button variant="destructive">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("bg-destructive");
    });

    it("renders outline variant correctly", () => {
      render(<Button variant="outline">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("border");
      expect(button.className).toContain("border-input");
    });

    it("renders secondary variant correctly", () => {
      render(<Button variant="secondary">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("bg-secondary");
    });

    it("renders ghost variant correctly", () => {
      render(<Button variant="ghost">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("hover:bg-accent");
    });

    it("renders link variant correctly", () => {
      render(<Button variant="link">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("text-primary");
      expect(button.className).toContain("underline-offset-4");
    });
  });

  describe("Sizes", () => {
    it("renders default size correctly", () => {
      render(<Button size="default">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("h-10");
      expect(button.className).toContain("px-4");
      expect(button.className).toContain("py-2");
    });

    it("renders sm size correctly", () => {
      render(<Button size="sm">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("h-9");
      expect(button.className).toContain("px-3");
    });

    it("renders lg size correctly", () => {
      render(<Button size="lg">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("h-11");
      expect(button.className).toContain("px-8");
    });

    it("renders icon size correctly", () => {
      render(<Button size="icon">Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("h-10");
      expect(button.className).toContain("w-10");
    });
  });

  describe("Interactions", () => {
    it("handles click events", () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Click me</Button>);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it("handles multiple clicks", () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Click me</Button>);

      const button = screen.getByRole("button");
      fireEvent.click(button);
      fireEvent.click(button);
      fireEvent.click(button);

      expect(handleClick).toHaveBeenCalledTimes(3);
    });

    it("does not trigger click when disabled", () => {
      const handleClick = jest.fn();
      render(
        <Button onClick={handleClick} disabled>
          Click me
        </Button>,
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(handleClick).not.toHaveBeenCalled();
    });

    it("handles keyboard events", async () => {
      const user = userEvent.setup();
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Press me</Button>);

      const button = screen.getByRole("button");
      button.focus();

      await user.keyboard("{Enter}");
      expect(handleClick).toHaveBeenCalledTimes(1);

      await user.keyboard(" ");
      expect(handleClick).toHaveBeenCalledTimes(2);
    });
  });

  describe("States", () => {
    it("renders disabled state correctly", () => {
      render(<Button disabled>Disabled Button</Button>);
      const button = screen.getByRole("button");

      expect(button).toBeDisabled();
      expect(button.className).toContain("disabled:pointer-events-none");
      expect(button.className).toContain("disabled:opacity-50");
    });

    it("maintains disabled visual state with all variants", () => {
      const variants = [
        "default",
        "destructive",
        "outline",
        "secondary",
        "ghost",
        "link",
      ] as const;

      variants.forEach((variant) => {
        const { rerender } = render(
          <Button variant={variant} disabled>
            Button
          </Button>,
        );
        const button = screen.getByRole("button");
        expect(button).toBeDisabled();
        expect(button.className).toContain("disabled:pointer-events-none");
        rerender(<></>);
      });
    });
  });

  describe("Props", () => {
    it("passes through HTML button attributes", () => {
      render(
        <Button
          id="test-button"
          data-testid="custom-button"
          aria-label="Custom label"
          type="submit"
        >
          Button
        </Button>,
      );

      const button = screen.getByRole("button");
      expect(button).toHaveAttribute("id", "test-button");
      expect(button).toHaveAttribute("data-testid", "custom-button");
      expect(button).toHaveAttribute("aria-label", "Custom label");
      expect(button).toHaveAttribute("type", "submit");
    });

    it("merges custom className with variant classes", () => {
      render(<Button className="custom-class another-class">Button</Button>);
      const button = screen.getByRole("button");

      expect(button.className).toContain("custom-class");
      expect(button.className).toContain("another-class");
      expect(button.className).toContain("bg-primary"); // Default variant class
    });

    it("forwards ref correctly", () => {
      const ref = React.createRef<HTMLButtonElement>();
      render(<Button ref={ref}>Button</Button>);

      expect(ref.current).toBeInstanceOf(HTMLButtonElement);
      expect(ref.current?.tagName).toBe("BUTTON");
    });
  });

  describe("asChild prop", () => {
    it("renders as child component when asChild is true", () => {
      render(
        <Button asChild>
          <a href="/test">Link Button</a>
        </Button>,
      );

      const link = screen.getByRole("link", { name: "Link Button" });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute("href", "/test");
      expect(link.className).toContain("bg-primary"); // Should still have button classes
    });

    it("preserves button styling on child elements", () => {
      render(
        <Button asChild variant="destructive" size="lg">
          <span>Span Button</span>
        </Button>,
      );

      const span = screen.getByText("Span Button");
      expect(span.className).toContain("bg-destructive");
      expect(span.className).toContain("h-11");
      expect(span.className).toContain("px-8");
    });
  });

  describe("Accessibility", () => {
    it("has proper focus styles", () => {
      render(<Button>Focusable Button</Button>);
      const button = screen.getByRole("button");

      button.focus();
      expect(button).toHaveFocus();
      expect(button.className).toContain("focus-visible:outline-none");
      expect(button.className).toContain("focus-visible:ring-2");
    });

    it("supports aria attributes", () => {
      render(
        <Button
          aria-pressed="true"
          aria-expanded="false"
          aria-describedby="description"
        >
          Accessible Button
        </Button>,
      );

      const button = screen.getByRole("button");
      expect(button).toHaveAttribute("aria-pressed", "true");
      expect(button).toHaveAttribute("aria-expanded", "false");
      expect(button).toHaveAttribute("aria-describedby", "description");
    });

    it("is keyboard navigable", () => {
      render(
        <>
          <Button>First</Button>
          <Button>Second</Button>
          <Button>Third</Button>
        </>,
      );

      const buttons = screen.getAllByRole("button");

      // Tab navigation
      buttons[0].focus();
      expect(buttons[0]).toHaveFocus();

      // Note: Tab navigation behavior would need to be tested in an integration/e2e test
      // as jsdom doesn't fully support tab navigation
    });
  });

  describe("Visual States", () => {
    it("applies hover styles", async () => {
      const user = userEvent.setup();
      render(<Button>Hover Button</Button>);
      const button = screen.getByRole("button");

      // The hover classes are applied via CSS, checking the class exists
      expect(button.className).toContain("hover:bg-primary/90");
    });

    it("shows focus ring on keyboard focus", () => {
      render(<Button>Focus Button</Button>);
      const button = screen.getByRole("button");

      button.focus();
      expect(button.className).toContain("focus-visible:ring-2");
      expect(button.className).toContain("focus-visible:ring-ring");
    });
  });

  describe("Edge Cases", () => {
    it("handles empty children", () => {
      render(<Button />);
      const button = screen.getByRole("button");
      expect(button).toBeInTheDocument();
      expect(button).toBeEmptyDOMElement();
    });

    it("handles null/undefined variant gracefully", () => {
      render(<Button variant={undefined}>Button</Button>);
      const button = screen.getByRole("button");
      expect(button.className).toContain("bg-primary"); // Should use default
    });

    it("handles complex children structures", () => {
      render(
        <Button>
          <span>Complex</span>
          <div>
            <strong>Children</strong>
          </div>
        </Button>,
      );

      expect(screen.getByText("Complex")).toBeInTheDocument();
      expect(screen.getByText("Children")).toBeInTheDocument();
    });
  });
});
