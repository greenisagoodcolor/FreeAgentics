import "./setup";
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Unmock the input component for this test
jest.unmock("@/components/ui/input");

import { Input } from "@/components/ui/input";

describe("Input Component", () => {
  describe("Rendering", () => {
    it("renders with default props", () => {
      render(<Input />);
      const input = screen.getByRole("textbox");
      expect(input).toBeInTheDocument();
      expect(input).toHaveClass("flex", "h-10", "w-full", "rounded-md");
    });

    it("renders with placeholder", () => {
      render(<Input placeholder="Enter text..." />);
      const input = screen.getByPlaceholderText("Enter text...");
      expect(input).toBeInTheDocument();
    });

    it("renders with default value", () => {
      render(<Input defaultValue="Initial value" />);
      const input = screen.getByRole("textbox");
      expect(input).toHaveValue("Initial value");
    });

    it("renders with controlled value", () => {
      const { rerender } = render(
        <Input value="Controlled value" onChange={() => {}} />,
      );
      const input = screen.getByRole("textbox");
      expect(input).toHaveValue("Controlled value");

      // Update value
      rerender(<Input value="Updated value" onChange={() => {}} />);
      expect(input).toHaveValue("Updated value");
    });
  });

  describe("Input Types", () => {
    const inputTypes = [
      "text",
      "email",
      "password",
      "number",
      "tel",
      "url",
      "search",
    ] as const;

    inputTypes.forEach((type) => {
      it(`renders ${type} input correctly`, () => {
        render(<Input type={type} placeholder={`${type} input`} />);
        const input = screen.getByPlaceholderText(`${type} input`);
        expect(input).toHaveAttribute("type", type);
      });
    });

    it("renders file input correctly", () => {
      render(<Input type="file" data-testid="file-input" />);
      const input = screen.getByTestId("file-input");
      expect(input).toHaveAttribute("type", "file");
      // File inputs have special styling
      expect(input.className).toContain("file:border-0");
      expect(input.className).toContain("file:bg-transparent");
    });

    it("defaults to text type when no type specified", () => {
      render(<Input placeholder="Default type" />);
      const input = screen.getByPlaceholderText("Default type");
      // When no type is specified, the browser doesn't set a type attribute
      // but treats it as text input by default
      expect(input.type).toBe("text");
    });
  });

  describe("User Interactions", () => {
    it("handles text input", async () => {
      const user = userEvent.setup();
      render(<Input placeholder="Type here" />);
      const input = screen.getByPlaceholderText("Type here");

      await user.type(input, "Hello World");
      expect(input).toHaveValue("Hello World");
    });

    it("handles onChange event", async () => {
      const handleChange = jest.fn();
      const user = userEvent.setup();
      render(<Input onChange={handleChange} placeholder="Change me" />);
      const input = screen.getByPlaceholderText("Change me");

      await user.type(input, "Test");
      expect(handleChange).toHaveBeenCalled();
      expect(handleChange).toHaveBeenCalledTimes(4); // Once for each character
    });

    it("handles onFocus and onBlur events", async () => {
      const handleFocus = jest.fn();
      const handleBlur = jest.fn();
      const user = userEvent.setup();

      render(
        <Input
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder="Focus me"
        />,
      );
      const input = screen.getByPlaceholderText("Focus me");

      await user.click(input);
      expect(handleFocus).toHaveBeenCalledTimes(1);

      await user.tab();
      expect(handleBlur).toHaveBeenCalledTimes(1);
    });

    it("handles paste event", async () => {
      const handlePaste = jest.fn();
      const user = userEvent.setup();
      render(<Input onPaste={handlePaste} placeholder="Paste here" />);
      const input = screen.getByPlaceholderText("Paste here");

      // Use userEvent's paste method or fireEvent directly
      fireEvent.paste(input, {
        clipboardData: {
          getData: () => "pasted text",
        },
      });

      expect(handlePaste).toHaveBeenCalledTimes(1);
    });

    it("handles keyboard events", async () => {
      const handleKeyDown = jest.fn();
      const handleKeyUp = jest.fn();
      const user = userEvent.setup();

      render(
        <Input
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          placeholder="Type here"
        />,
      );
      const input = screen.getByPlaceholderText("Type here");

      await user.type(input, "a");
      expect(handleKeyDown).toHaveBeenCalled();
      expect(handleKeyUp).toHaveBeenCalled();
    });

    it("handles form submission on Enter", async () => {
      const handleSubmit = jest.fn((e) => e.preventDefault());
      const user = userEvent.setup();

      render(
        <form onSubmit={handleSubmit}>
          <Input placeholder="Press Enter" />
        </form>,
      );
      const input = screen.getByPlaceholderText("Press Enter");

      await user.type(input, "Test{Enter}");
      expect(handleSubmit).toHaveBeenCalledTimes(1);
    });
  });

  describe("States", () => {
    it("renders disabled state correctly", () => {
      render(<Input disabled placeholder="Disabled input" />);
      const input = screen.getByPlaceholderText("Disabled input");

      expect(input).toBeDisabled();
      expect(input.className).toContain("disabled:cursor-not-allowed");
      expect(input.className).toContain("disabled:opacity-50");
    });

    it("prevents interaction when disabled", async () => {
      const handleChange = jest.fn();
      const user = userEvent.setup();

      render(<Input disabled onChange={handleChange} placeholder="Disabled" />);
      const input = screen.getByPlaceholderText("Disabled");

      await user.type(input, "Test");
      expect(handleChange).not.toHaveBeenCalled();
      expect(input).toHaveValue("");
    });

    it("renders readonly state correctly", () => {
      render(<Input readOnly value="Read only value" />);
      const input = screen.getByRole("textbox");

      expect(input).toHaveAttribute("readonly");
      expect(input).toHaveValue("Read only value");
    });

    it("prevents editing in readonly state", async () => {
      const user = userEvent.setup();
      render(<Input readOnly value="Cannot edit" onChange={() => {}} />);
      const input = screen.getByRole("textbox");

      await user.type(input, "New text");
      expect(input).toHaveValue("Cannot edit"); // Value should not change
    });

    it("shows required state", () => {
      render(<Input required placeholder="Required field" />);
      const input = screen.getByPlaceholderText("Required field");

      expect(input).toHaveAttribute("required");
      expect(input).toBeRequired();
    });
  });

  describe("Props and Attributes", () => {
    it("passes through HTML input attributes", () => {
      render(
        <Input
          id="test-input"
          name="testInput"
          data-testid="custom-input"
          aria-label="Custom input"
          maxLength={50}
          minLength={5}
          pattern="[A-Za-z]+"
          autoComplete="off"
          autoFocus
        />,
      );

      const input = screen.getByRole("textbox");
      expect(input).toHaveAttribute("id", "test-input");
      expect(input).toHaveAttribute("name", "testInput");
      expect(input).toHaveAttribute("data-testid", "custom-input");
      expect(input).toHaveAttribute("aria-label", "Custom input");
      expect(input).toHaveAttribute("maxLength", "50");
      expect(input).toHaveAttribute("minLength", "5");
      expect(input).toHaveAttribute("pattern", "[A-Za-z]+");
      expect(input).toHaveAttribute("autoComplete", "off");
      expect(input).toHaveFocus();
    });

    it("merges custom className with default classes", () => {
      render(<Input className="custom-class another-class" />);
      const input = screen.getByRole("textbox");

      expect(input.className).toContain("custom-class");
      expect(input.className).toContain("another-class");
      expect(input.className).toContain("flex"); // Default class
      expect(input.className).toContain("h-10"); // Default class
    });

    it("forwards ref correctly", () => {
      const ref = React.createRef<HTMLInputElement>();
      render(<Input ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLInputElement);
      expect(ref.current?.tagName).toBe("INPUT");
    });
  });

  describe("Styling and Visual States", () => {
    it("applies focus styles", async () => {
      const user = userEvent.setup();
      render(<Input placeholder="Focus me" />);
      const input = screen.getByPlaceholderText("Focus me");

      await user.click(input);
      expect(input).toHaveFocus();
      expect(input.className).toContain("focus-visible:outline-none");
      expect(input.className).toContain("focus-visible:ring-2");
      expect(input.className).toContain("focus-visible:ring-ring");
    });

    it("applies correct border and background styles", () => {
      render(<Input />);
      const input = screen.getByRole("textbox");

      expect(input.className).toContain("border");
      expect(input.className).toContain("border-input");
      expect(input.className).toContain("bg-background");
    });

    it("applies correct padding and sizing", () => {
      render(<Input />);
      const input = screen.getByRole("textbox");

      expect(input.className).toContain("px-3");
      expect(input.className).toContain("py-2");
      expect(input.className).toContain("h-10");
      expect(input.className).toContain("w-full");
    });

    it("has responsive text size", () => {
      render(<Input />);
      const input = screen.getByRole("textbox");

      expect(input.className).toContain("text-base");
      expect(input.className).toContain("md:text-sm");
    });

    it("styles placeholder text", () => {
      render(<Input placeholder="Placeholder text" />);
      const input = screen.getByPlaceholderText("Placeholder text");

      expect(input.className).toContain("placeholder:text-muted-foreground");
    });
  });

  describe("Validation", () => {
    it("handles HTML5 validation attributes", () => {
      render(
        <Input
          type="email"
          required
          minLength={5}
          maxLength={50}
          pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"
          placeholder="Email"
        />,
      );

      const input = screen.getByPlaceholderText("Email");
      expect(input).toHaveAttribute("type", "email");
      expect(input).toHaveAttribute("required");
      expect(input).toHaveAttribute("minLength", "5");
      expect(input).toHaveAttribute("maxLength", "50");
      expect(input).toHaveAttribute("pattern");
    });

    it("supports number input constraints", () => {
      render(
        <Input type="number" min={0} max={100} step={5} placeholder="Number" />,
      );

      const input = screen.getByPlaceholderText("Number");
      expect(input).toHaveAttribute("type", "number");
      expect(input).toHaveAttribute("min", "0");
      expect(input).toHaveAttribute("max", "100");
      expect(input).toHaveAttribute("step", "5");
    });
  });

  describe("Accessibility", () => {
    it("supports aria attributes", () => {
      render(
        <Input
          aria-label="Accessible input"
          aria-describedby="input-description"
          aria-invalid="true"
          aria-required="true"
          placeholder="Accessible"
        />,
      );

      const input = screen.getByPlaceholderText("Accessible");
      expect(input).toHaveAttribute("aria-label", "Accessible input");
      expect(input).toHaveAttribute("aria-describedby", "input-description");
      expect(input).toHaveAttribute("aria-invalid", "true");
      expect(input).toHaveAttribute("aria-required", "true");
    });

    it("is keyboard navigable", async () => {
      const user = userEvent.setup();
      render(
        <>
          <Input placeholder="First input" />
          <Input placeholder="Second input" />
          <Input placeholder="Third input" />
        </>,
      );

      const firstInput = screen.getByPlaceholderText("First input");
      const secondInput = screen.getByPlaceholderText("Second input");

      await user.click(firstInput);
      expect(firstInput).toHaveFocus();

      await user.tab();
      expect(secondInput).toHaveFocus();
    });
  });

  describe("Edge Cases", () => {
    it("handles very long text input", async () => {
      const user = userEvent.setup();
      const longText = "a".repeat(1000);

      render(<Input placeholder="Long text" />);
      const input = screen.getByPlaceholderText("Long text");

      await user.type(input, longText);
      expect(input).toHaveValue(longText);
    });

    it("handles special characters", async () => {
      const handleChange = jest.fn();
      render(<Input onChange={handleChange} placeholder="Special chars" />);
      const input = screen.getByPlaceholderText(
        "Special chars",
      ) as HTMLInputElement;

      // Test special characters that don't conflict with userEvent's syntax
      const specialChars = "!@#$%^&*()_+-=;':\",./<>?`~";

      // Use fireEvent for direct input simulation to avoid userEvent's parsing
      fireEvent.change(input, { target: { value: specialChars } });

      expect(input.value).toBe(specialChars);
      expect(handleChange).toHaveBeenCalled();
    });

    it("handles empty className prop", () => {
      render(<Input className="" />);
      const input = screen.getByRole("textbox");

      // Should still have default classes
      expect(input.className).toContain("flex");
      expect(input.className).toContain("h-10");
    });

    it("handles rapid value changes", async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();

      render(<Input onChange={handleChange} placeholder="Rapid changes" />);
      const input = screen.getByPlaceholderText("Rapid changes");

      // Type rapidly
      await user.type(input, "abcdefghijklmnop");

      expect(input).toHaveValue("abcdefghijklmnop");
      expect(handleChange).toHaveBeenCalledTimes(16);
    });
  });
});
