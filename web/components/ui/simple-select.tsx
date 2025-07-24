import * as React from "react";

interface SimpleSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}

export const SimpleSelect: React.FC<SimpleSelectProps> = ({ value, onValueChange, children }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  const childrenArray = React.Children.toArray(children);
  const trigger = childrenArray.find(
    (child) => React.isValidElement(child) && child.type === SelectTrigger,
  );
  const content = childrenArray.find(
    (child) => React.isValidElement(child) && child.type === SelectContent,
  );

  return (
    <div className="relative">
      <div onClick={() => setIsOpen(!isOpen)}>
        {React.isValidElement(trigger) &&
          React.cloneElement(trigger as React.ReactElement<{ value?: string; isOpen?: boolean }>, {
            value,
            isOpen,
          })}
      </div>
      {isOpen && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1">
          {React.isValidElement(content) &&
            React.cloneElement(
              content as React.ReactElement<{ onSelect?: (val: string) => void }>,
              {
                onSelect: (val: string) => {
                  onValueChange(val);
                  setIsOpen(false);
                },
              },
            )}
        </div>
      )}
    </div>
  );
};

export const SelectTrigger: React.FC<{
  children: React.ReactNode;
  value?: string;
  isOpen?: boolean;
  id?: string;
}> = ({ children, value, isOpen, id }) => {
  const valueElement = React.Children.toArray(children).find(
    (child) => React.isValidElement(child) && child.type === SelectValue,
  );

  return (
    <button
      id={id}
      role="combobox"
      aria-expanded={isOpen}
      aria-haspopup="listbox"
      aria-controls={`${id}-listbox`}
      className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
      type="button"
    >
      {React.isValidElement(valueElement) &&
        React.cloneElement(valueElement as React.ReactElement<{ value?: string }>, { value })}
      <svg
        className={`h-4 w-4 opacity-50 transition-transform ${isOpen ? "rotate-180" : ""}`}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  );
};

export const SelectValue: React.FC<{ value?: string; placeholder?: string }> = ({
  value,
  placeholder,
}) => {
  // Find the display text from the parent SelectContent if available
  const displayText = React.useMemo(() => {
    // If there's a specific placeholder, use it
    if (placeholder) return placeholder;

    // For LLM providers, provide proper capitalization
    const providerNames: Record<string, string> = {
      openai: "OpenAI",
      anthropic: "Anthropic",
      ollama: "Ollama",
    };

    return providerNames[value || ""] || value || "Select...";
  }, [value, placeholder]);

  return <span>{displayText}</span>;
};

export const SelectContent: React.FC<{
  children: React.ReactNode;
  onSelect?: (value: string) => void;
}> = ({ children, onSelect }) => {
  return (
    <div className="relative z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md">
      <div className="w-full p-1">
        {React.Children.map(children, (child) => {
          if (React.isValidElement(child) && child.type === SelectItem) {
            return React.cloneElement(
              child as React.ReactElement<{ onSelect?: (value: string) => void }>,
              { onSelect },
            );
          }
          return child;
        })}
      </div>
    </div>
  );
};

export const SelectItem: React.FC<{
  children: React.ReactNode;
  value: string;
  onSelect?: (value: string) => void;
}> = ({ children, value, onSelect }) => {
  return (
    <div
      className="relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
      onClick={() => onSelect?.(value)}
    >
      {children}
    </div>
  );
};

export const Select = SimpleSelect;
