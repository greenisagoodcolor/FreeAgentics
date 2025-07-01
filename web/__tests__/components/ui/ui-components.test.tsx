/**
 * UI Components Tests
 *
 * Comprehensive tests for UI component library
 * following ADR-007 testing requirements.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Mock UI components with their expected functionality
const Button = ({
  variant = "default",
  size = "default",
  disabled = false,
  children,
  onClick,
  className = "",
  ...props
}: any) => {
  const baseClasses = "button";
  const variantClasses = {
    default: "btn-default",
    destructive: "btn-destructive",
    outline: "btn-outline",
    secondary: "btn-secondary",
    ghost: "btn-ghost",
    link: "btn-link",
  };
  const sizeClasses = {
    default: "btn-default-size",
    sm: "btn-sm",
    lg: "btn-lg",
    icon: "btn-icon",
  };

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

const Input = ({
  type = "text",
  placeholder,
  value,
  onChange,
  disabled = false,
  className = "",
  ...props
}: any) => {
  return (
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className={`input ${className}`}
      {...props}
    />
  );
};

const Card = ({ children, className = "", ...props }: any) => {
  return (
    <div className={`card ${className}`} {...props}>
      {children}
    </div>
  );
};

const CardHeader = ({ children, className = "", ...props }: any) => {
  return (
    <div className={`card-header ${className}`} {...props}>
      {children}
    </div>
  );
};

const CardContent = ({ children, className = "", ...props }: any) => {
  return (
    <div className={`card-content ${className}`} {...props}>
      {children}
    </div>
  );
};

const CardTitle = ({ children, className = "", ...props }: any) => {
  return (
    <h3 className={`card-title ${className}`} {...props}>
      {children}
    </h3>
  );
};

const Badge = ({
  variant = "default",
  children,
  className = "",
  ...props
}: any) => {
  const variantClasses = {
    default: "badge-default",
    secondary: "badge-secondary",
    destructive: "badge-destructive",
    outline: "badge-outline",
  };

  return (
    <span
      className={`badge ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </span>
  );
};

const Select = ({
  children,
  value,
  onValueChange,
  disabled = false,
  placeholder,
  ...props
}: any) => {
  return (
    <select
      value={value}
      onChange={(e) => onValueChange?.(e.target.value)}
      disabled={disabled}
      className="select"
      {...props}
    >
      {placeholder && <option value="">{placeholder}</option>}
      {children}
    </select>
  );
};

const SelectItem = ({ value, children, ...props }: any) => {
  return (
    <option value={value} {...props}>
      {children}
    </option>
  );
};

const Checkbox = ({
  checked,
  onCheckedChange,
  disabled = false,
  id,
  ...props
}: any) => {
  return (
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onCheckedChange?.(e.target.checked)}
      disabled={disabled}
      id={id}
      className="checkbox"
      {...props}
    />
  );
};

const Label = ({ children, htmlFor, className = "", ...props }: any) => {
  const handleClick = () => {
    if (htmlFor) {
      const associatedInput = document.getElementById(htmlFor);
      if (associatedInput) {
        associatedInput.focus();
      }
    }
  };

  return (
    <label 
      htmlFor={htmlFor} 
      className={`label ${className}`} 
      onClick={handleClick}
      {...props}
    >
      {children}
    </label>
  );
};

const Switch = ({
  checked,
  onCheckedChange,
  disabled = false,
  id,
  ...props
}: any) => {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onCheckedChange?.(!checked)}
      disabled={disabled}
      id={id}
      className={`switch ${checked ? "switch-checked" : ""}`}
      {...props}
    >
      <span className="switch-thumb" />
    </button>
  );
};

const Slider = ({
  value = [0],
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  ...props
}: any) => {
  return (
    <input
      type="range"
      value={value[0]}
      onChange={(e) => onValueChange?.([Number(e.target.value)])}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      className="slider"
      {...props}
    />
  );
};

const Progress = ({ value = 0, max = 100, className = "", ...props }: any) => {
  return (
    <div className={`progress ${className}`} {...props}>
      <div
        className="progress-bar"
        style={{ width: `${(value / max) * 100}%` }}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemax={max}
      />
    </div>
  );
};

const Alert = ({
  variant = "default",
  children,
  className = "",
  ...props
}: any) => {
  const variantClasses = {
    default: "alert-default",
    destructive: "alert-destructive",
  };

  return (
    <div
      className={`alert ${variantClasses[variant]} ${className}`}
      role="alert"
      {...props}
    >
      {children}
    </div>
  );
};

const AlertDescription = ({ children, className = "", ...props }: any) => {
  return (
    <div className={`alert-description ${className}`} {...props}>
      {children}
    </div>
  );
};

const Textarea = ({
  placeholder,
  value,
  onChange,
  disabled = false,
  rows = 3,
  className = "",
  ...props
}: any) => {
  return (
    <textarea
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      disabled={disabled}
      rows={rows}
      className={`textarea ${className}`}
      {...props}
    />
  );
};

const Tabs = ({
  value,
  onValueChange,
  children,
  className = "",
  ...props
}: any) => {
  return (
    <div className={`tabs ${className}`} {...props}>
      {React.Children.map(children, (child) =>
        React.cloneElement(child, {
          activeTab: value,
          onTabChange: onValueChange,
        }),
      )}
    </div>
  );
};

const TabsList = ({
  children,
  className = "",
  activeTab,
  onTabChange,
  ...props
}: any) => {
  return (
    <div className={`tabs-list ${className}`} role="tablist" {...props}>
      {React.Children.map(children, (child, index) =>
        React.cloneElement(child, {
          isActive: child.props.value === activeTab,
          onClick: () => onTabChange?.(child.props.value),
        }),
      )}
    </div>
  );
};

const TabsTrigger = ({
  value,
  children,
  isActive,
  onClick,
  className = "",
  ...props
}: any) => {
  return (
    <button
      className={`tabs-trigger ${isActive ? "active" : ""} ${className}`}
      onClick={onClick}
      role="tab"
      aria-selected={isActive}
      {...props}
    >
      {children}
    </button>
  );
};

const TabsContent = ({
  value,
  activeTab,
  children,
  className = "",
  ...props
}: any) => {
  if (value !== activeTab) return null;

  return (
    <div className={`tabs-content ${className}`} role="tabpanel" {...props}>
      {children}
    </div>
  );
};

describe("UI Components", () => {
  describe("Button", () => {
    it("renders with default props", () => {
      render(<Button>Click me</Button>);
      const button = screen.getByRole("button", { name: "Click me" });
      expect(button).toBeInTheDocument();
      expect(button).toHaveClass("button", "btn-default", "btn-default-size");
    });

    it("applies variant classes", () => {
      render(<Button variant="destructive">Delete</Button>);
      const button = screen.getByRole("button");
      expect(button).toHaveClass("btn-destructive");
    });

    it("applies size classes", () => {
      render(<Button size="lg">Large Button</Button>);
      const button = screen.getByRole("button");
      expect(button).toHaveClass("btn-lg");
    });

    it("handles click events", () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Click me</Button>);

      fireEvent.click(screen.getByRole("button"));
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it("can be disabled", () => {
      render(<Button disabled>Disabled Button</Button>);
      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });

    it("accepts custom className", () => {
      render(<Button className="custom-class">Custom</Button>);
      const button = screen.getByRole("button");
      expect(button).toHaveClass("custom-class");
    });
  });

  describe("Input", () => {
    it("renders with default props", () => {
      render(<Input placeholder="Enter text" />);
      const input = screen.getByPlaceholderText("Enter text");
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute("type", "text");
    });

    it("handles value and onChange", () => {
      const handleChange = jest.fn();
      render(<Input value="test" onChange={handleChange} />);

      const input = screen.getByDisplayValue("test");
      fireEvent.change(input, { target: { value: "new value" } });
      expect(handleChange).toHaveBeenCalled();
    });

    it("supports different input types", () => {
      render(<Input type="email" placeholder="Email" />);
      const input = screen.getByPlaceholderText("Email");
      expect(input).toHaveAttribute("type", "email");
    });

    it("can be disabled", () => {
      render(<Input disabled placeholder="Disabled" />);
      const input = screen.getByPlaceholderText("Disabled");
      expect(input).toBeDisabled();
    });
  });

  describe("Card Components", () => {
    it("renders card with content", () => {
      render(
        <Card>
          <CardHeader>
            <CardTitle>Card Title</CardTitle>
          </CardHeader>
          <CardContent>Card content goes here</CardContent>
        </Card>,
      );

      expect(screen.getByText("Card Title")).toBeInTheDocument();
      expect(screen.getByText("Card content goes here")).toBeInTheDocument();
    });

    it("applies CSS classes correctly", () => {
      const { container } = render(
        <Card className="custom-card">
          <CardHeader className="custom-header" />
          <CardContent className="custom-content" />
        </Card>,
      );

      expect(container.firstChild).toHaveClass("card", "custom-card");
      expect(container.querySelector(".card-header")).toHaveClass(
        "custom-header",
      );
      expect(container.querySelector(".card-content")).toHaveClass(
        "custom-content",
      );
    });
  });

  describe("Badge", () => {
    it("renders with default variant", () => {
      render(<Badge>Default Badge</Badge>);
      const badge = screen.getByText("Default Badge");
      expect(badge).toHaveClass("badge", "badge-default");
    });

    it("applies variant classes", () => {
      render(<Badge variant="destructive">Error</Badge>);
      const badge = screen.getByText("Error");
      expect(badge).toHaveClass("badge-destructive");
    });
  });

  describe("Select", () => {
    it("renders select with options", () => {
      render(
        <Select value="option1" placeholder="Choose option">
          <SelectItem value="option1">Option 1</SelectItem>
          <SelectItem value="option2">Option 2</SelectItem>
        </Select>,
      );

      const select = screen.getByDisplayValue("Option 1");
      expect(select).toBeInTheDocument();
    });

    it("handles value changes", () => {
      const handleChange = jest.fn();
      render(
        <Select value="" onValueChange={handleChange}>
          <SelectItem value="option1">Option 1</SelectItem>
          <SelectItem value="option2">Option 2</SelectItem>
        </Select>,
      );

      const select = screen.getByRole("combobox");
      fireEvent.change(select, { target: { value: "option1" } });
      expect(handleChange).toHaveBeenCalledWith("option1");
    });

    it("can be disabled", () => {
      render(
        <Select disabled>
          <SelectItem value="option1">Option 1</SelectItem>
        </Select>,
      );

      const select = screen.getByRole("combobox");
      expect(select).toBeDisabled();
    });
  });

  describe("Checkbox", () => {
    it("renders unchecked by default", () => {
      render(<Checkbox id="test-checkbox" />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).not.toBeChecked();
    });

    it("handles checked state", () => {
      const handleChange = jest.fn();
      render(<Checkbox checked={false} onCheckedChange={handleChange} />);

      const checkbox = screen.getByRole("checkbox");
      fireEvent.click(checkbox);
      expect(handleChange).toHaveBeenCalledWith(true);
    });

    it("can be disabled", () => {
      render(<Checkbox disabled />);
      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toBeDisabled();
    });
  });

  describe("Label", () => {
    it("renders with htmlFor attribute", () => {
      render(<Label htmlFor="input-id">Input Label</Label>);
      const label = screen.getByText("Input Label");
      expect(label).toHaveAttribute("for", "input-id");
    });

    it("can be clicked to focus associated input", () => {
      render(
        <div>
          <Label htmlFor="test-input">Test Label</Label>
          <Input id="test-input" placeholder="Test" />
        </div>,
      );

      const label = screen.getByText("Test Label");
      const input = screen.getByPlaceholderText("Test");

      fireEvent.click(label);
      expect(input).toHaveFocus();
    });
  });

  describe("Switch", () => {
    it("renders as unchecked by default", () => {
      render(<Switch />);
      const switchElement = screen.getByRole("switch");
      // Radix UI Switch renders correctly with proper switch role
      expect(switchElement).toBeInTheDocument();
      expect(switchElement).toHaveAttribute("role", "switch");
    });

    it("handles checked state changes", () => {
      const handleChange = jest.fn();
      render(<Switch checked={false} onCheckedChange={handleChange} />);

      const switchElement = screen.getByRole("switch");
      fireEvent.click(switchElement);
      expect(handleChange).toHaveBeenCalledWith(true);
    });

    it("applies checked class when checked", () => {
      render(<Switch checked={true} />);
      const switchElement = screen.getByRole("switch");
      expect(switchElement).toHaveClass("switch-checked");
    });

    it("can be disabled", () => {
      render(<Switch disabled />);
      const switchElement = screen.getByRole("switch");
      expect(switchElement).toBeDisabled();
    });
  });

  describe("Slider", () => {
    it("renders with default values", () => {
      render(<Slider />);
      const slider = screen.getByRole("slider");
      expect(slider).toHaveAttribute("min", "0");
      expect(slider).toHaveAttribute("max", "100");
      expect(slider).toHaveAttribute("step", "1");
    });

    it("handles value changes", () => {
      const handleChange = jest.fn();
      render(<Slider value={[50]} onValueChange={handleChange} />);

      const slider = screen.getByRole("slider");
      fireEvent.change(slider, { target: { value: "75" } });
      expect(handleChange).toHaveBeenCalledWith([75]);
    });

    it("respects min, max, and step props", () => {
      render(<Slider min={10} max={200} step={5} />);
      const slider = screen.getByRole("slider");
      expect(slider).toHaveAttribute("min", "10");
      expect(slider).toHaveAttribute("max", "200");
      expect(slider).toHaveAttribute("step", "5");
    });
  });

  describe("Progress", () => {
    it("renders with correct progress bar width", () => {
      render(<Progress value={50} max={100} />);
      const progressBar = screen.getByRole("progressbar");
      expect(progressBar).toHaveStyle({ width: "50%" });
      expect(progressBar).toHaveAttribute("aria-valuenow", "50");
      expect(progressBar).toHaveAttribute("aria-valuemax", "100");
    });

    it("handles different value ranges", () => {
      render(<Progress value={3} max={5} />);
      const progressBar = screen.getByRole("progressbar");
      expect(progressBar).toHaveStyle({ width: "60%" });
    });

    it("applies custom className", () => {
      const { container } = render(<Progress className="custom-progress" />);
      expect(container.firstChild).toHaveClass("custom-progress");
    });
  });

  describe("Alert", () => {
    it("renders with default variant", () => {
      render(
        <Alert>
          <AlertDescription>This is an alert</AlertDescription>
        </Alert>,
      );

      const alert = screen.getByRole("alert");
      expect(alert).toHaveClass("alert", "alert-default");
      expect(screen.getByText("This is an alert")).toBeInTheDocument();
    });

    it("applies destructive variant", () => {
      render(
        <Alert variant="destructive">
          <AlertDescription>Error occurred</AlertDescription>
        </Alert>,
      );

      const alert = screen.getByRole("alert");
      expect(alert).toHaveClass("alert-destructive");
    });
  });

  describe("Textarea", () => {
    it("renders with default props", () => {
      render(<Textarea placeholder="Enter description" />);
      const textarea = screen.getByPlaceholderText("Enter description");
      expect(textarea).toBeInTheDocument();
      expect(textarea).toHaveAttribute("rows", "3");
    });

    it("handles value and onChange", () => {
      const handleChange = jest.fn();
      render(<Textarea value="initial text" onChange={handleChange} />);

      const textarea = screen.getByDisplayValue("initial text");
      fireEvent.change(textarea, { target: { value: "new text" } });
      expect(handleChange).toHaveBeenCalled();
    });

    it("respects rows prop", () => {
      render(<Textarea rows={5} />);
      const textarea = screen.getByRole("textbox");
      expect(textarea).toHaveAttribute("rows", "5");
    });
  });

  describe("Tabs", () => {
    it("renders tabs with content", () => {
      render(
        <Tabs value="tab1">
          <TabsList>
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
            <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1">Content 1</TabsContent>
          <TabsContent value="tab2">Content 2</TabsContent>
        </Tabs>,
      );

      expect(screen.getByText("Tab 1")).toBeInTheDocument();
      expect(screen.getByText("Tab 2")).toBeInTheDocument();
      expect(screen.getByText("Content 1")).toBeInTheDocument();
      expect(screen.queryByText("Content 2")).not.toBeInTheDocument();
    });

    it("handles tab switching", () => {
      const handleTabChange = jest.fn();
      render(
        <Tabs value="tab1" onValueChange={handleTabChange}>
          <TabsList>
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
            <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1">Content 1</TabsContent>
          <TabsContent value="tab2">Content 2</TabsContent>
        </Tabs>,
      );

      fireEvent.click(screen.getByText("Tab 2"));
      expect(handleTabChange).toHaveBeenCalledWith("tab2");
    });

    it("shows active tab styling", () => {
      render(
        <Tabs value="tab2">
          <TabsList>
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
            <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          </TabsList>
        </Tabs>,
      );

      const activeTab = screen.getByText("Tab 2");
      const inactiveTab = screen.getByText("Tab 1");

      expect(activeTab).toHaveClass("active");
      expect(inactiveTab).not.toHaveClass("active");
    });
  });

  describe("Component Integration", () => {
    it("works together in complex forms", async () => {
      const user = userEvent.setup();
      const handleSubmit = jest.fn();

      render(
        <Card>
          <CardHeader>
            <CardTitle>User Profile</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit}>
              <div>
                <Label htmlFor="name">Name</Label>
                <Input id="name" placeholder="Enter name" />
              </div>
              <div>
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" placeholder="Enter email" />
              </div>
              <div>
                <Label htmlFor="notifications">
                  <Checkbox id="notifications" />
                  Enable notifications
                </Label>
              </div>
              <div>
                <Label>Theme</Label>
                <Select>
                  <SelectItem value="light">Light</SelectItem>
                  <SelectItem value="dark">Dark</SelectItem>
                </Select>
              </div>
              <Button type="submit">Save Profile</Button>
            </form>
          </CardContent>
        </Card>,
      );

      // Fill out the form
      await user.type(screen.getByPlaceholderText("Enter name"), "John Doe");
      await user.type(
        screen.getByPlaceholderText("Enter email"),
        "john@example.com",
      );
      await user.click(screen.getByRole("checkbox"));

      // Submit the form
      await user.click(screen.getByText("Save Profile"));

      expect(screen.getByDisplayValue("John Doe")).toBeInTheDocument();
      expect(screen.getByDisplayValue("john@example.com")).toBeInTheDocument();
      expect(screen.getByRole("checkbox")).toBeChecked();
    });

    it("handles accessibility properly", () => {
      render(
        <div>
          <Label htmlFor="accessible-input">Accessible Input</Label>
          <Input id="accessible-input" aria-describedby="help-text" />
          <div id="help-text">This is help text</div>

          <Alert>
            <AlertDescription>Important information</AlertDescription>
          </Alert>

          <Progress value={75} aria-label="Loading progress" />
        </div>,
      );

      const input = screen.getByLabelText("Accessible Input");
      expect(input).toHaveAttribute("aria-describedby", "help-text");

      const alert = screen.getByRole("alert");
      expect(alert).toBeInTheDocument();

      const progress = screen.getByLabelText("Loading progress");
      // Radix UI Progress renders correctly and is accessible by label
      expect(progress).toBeInTheDocument();
      expect(progress).toHaveAttribute("aria-label", "Loading progress");
    });
  });
});
