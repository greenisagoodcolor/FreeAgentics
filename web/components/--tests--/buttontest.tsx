import "@testing-library/jest-dom";
import { fireEvent, render, screen } from "@testing-library/react";

// Example Button component for testing
const Button = ({
  onClick,
  children,
  disabled = false,
}: {
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
}) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className="px-4 py-2 bg-blue-500 text-white rounded"
  >
    {children}
  </button>
);

describe("Button Component", () => {
  it("renders with children", () => {
    render(<Button onClick={() => {}}>Click me</Button>);

    const button = screen.getByRole("button", { name: /click me/i });
    expect(button).toBeInTheDocument();
  });

  it("calls onClick handler when clicked", () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);

    const button = screen.getByRole("button");
    fireEvent.click(button);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it("does not call onClick when disabled", () => {
    const handleClick = jest.fn();
    render(
      <Button onClick={handleClick} disabled>
        Click me
      </Button>,
    );

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();

    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });

  it("applies the correct CSS classes", () => {
    render(<Button onClick={() => {}}>Styled Button</Button>);

    const button = screen.getByRole("button");
    expect(button).toHaveClass(
      "px-4",
      "py-2",
      "bg-blue-500",
      "text-white",
      "rounded",
    );
  });
});
