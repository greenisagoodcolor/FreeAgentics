/**
 * UI Component Mock Factory
 *
 * Centralized mock factory for UI components following established patterns
 * from backend MockFactory and frontend component mocking strategies.
 *
 * This reduces technical debt by:
 * 1. Providing consistent mocking patterns across all frontend tests
 * 2. Centralizing component mock definitions
 * 3. Using established MockFactory patterns from backend
 * 4. Enabling comprehensive component testing without dependency issues
 */

import React from "react";

export interface UIComponentMockOptions {
  includeDataTestIds?: boolean;
  includeAriaLabels?: boolean;
  mockComplexBehavior?: boolean;
}

/**
 * Creates a comprehensive mock component that handles all common props
 * and behaviors expected by UI components
 */
export const createUIComponentMock = (
  componentName: string,
  options: UIComponentMockOptions = {},
) => {
  const {
    includeDataTestIds = true,
    includeAriaLabels = true,
    mockComplexBehavior = false,
  } = options;

  return ({
    children,
    className,
    onClick,
    onChange,
    onValueChange,
    ...props
  }: any) => {
    const testId = includeDataTestIds
      ? `mock-${componentName.toLowerCase()}`
      : undefined;
    const ariaLabel = includeAriaLabels ? `Mock ${componentName}` : undefined;

    // Handle different component types
    if (componentName.includes("Button")) {
      return React.createElement(
        "button",
        {
          className,
          onClick,
          "data-testid": testId,
          "aria-label": ariaLabel,
          ...props,
        },
        children,
      );
    }

    if (componentName.includes("Input")) {
      return React.createElement("input", {
        className,
        onChange,
        "data-testid": testId,
        "aria-label": ariaLabel,
        ...props,
      });
    }

    if (componentName.includes("Select")) {
      return React.createElement(
        "select",
        {
          className,
          onChange: (e: any) => {
            onChange?.(e);
            onValueChange?.(e.target.value);
          },
          "data-testid": testId,
          "aria-label": ariaLabel,
          ...props,
        },
        children,
      );
    }

    // Default to div for container components
    return React.createElement(
      "div",
      {
        className,
        onClick,
        "data-testid": testId,
        "aria-label": ariaLabel,
        ...props,
      },
      children,
    );
  };
};

/**
 * Mock factory for form components with validation behavior
 */
export const createFormComponentMock = (componentName: string) => {
  return ({ children, onSubmit, ...props }: any) => {
    return React.createElement(
      "form",
      {
        onSubmit: (e: any) => {
          e.preventDefault();
          onSubmit?.(e);
        },
        "data-testid": `mock-${componentName.toLowerCase()}`,
        ...props,
      },
      children,
    );
  };
};

/**
 * Mock factory for complex interactive components (modals, dropdowns, etc.)
 */
export const createInteractiveComponentMock = (componentName: string) => {
  return ({ children, open, onOpenChange, isOpen, onClose, ...props }: any) => {
    const isVisible = open ?? isOpen ?? true;

    if (!isVisible) {
      return null;
    }

    return React.createElement(
      "div",
      {
        role: componentName.toLowerCase().includes("dialog")
          ? "dialog"
          : "region",
        "data-testid": `mock-${componentName.toLowerCase()}`,
        "aria-modal": componentName.toLowerCase().includes("dialog")
          ? "true"
          : undefined,
        ...props,
      },
      [
        children,
        // Add close button for dialogs/modals
        (onOpenChange || onClose) &&
          React.createElement(
            "button",
            {
              key: "close-btn",
              onClick: () => {
                onOpenChange?.(false);
                onClose?.();
              },
              "aria-label": "Close",
              "data-testid": "mock-close-button",
            },
            "×",
          ),
      ],
    );
  };
};

/**
 * Comprehensive UI component mock registry
 * Based on shadcn/ui component library structure
 */
export const UI_COMPONENT_MOCKS = {
  // Basic components
  Button: createUIComponentMock("Button"),
  Input: createUIComponentMock("Input"),
  Label: createUIComponentMock("Label"),
  Textarea: createUIComponentMock("Textarea"),

  // Container components
  Card: createUIComponentMock("Card"),
  CardContent: createUIComponentMock("CardContent"),
  CardHeader: createUIComponentMock("CardHeader"),
  CardTitle: createUIComponentMock("CardTitle"),
  CardDescription: createUIComponentMock("CardDescription"),

  // Form components - Custom Select to avoid DOM nesting issues
  Select: ({ children, value, onValueChange, disabled, ...props }: any) => {
    return React.createElement(
      "div",
      {
        "data-testid": "mock-select",
        "data-value": value,
        "data-disabled": disabled,
        ...props,
      },
      children,
    );
  },
  SelectContent: ({ children, ...props }: any) =>
    React.createElement(
      "div",
      {
        "data-testid": "mock-select-content",
        ...props,
      },
      children,
    ),
  SelectItem: ({ value, children, onSelect, ...props }: any) =>
    React.createElement(
      "div",
      {
        "data-testid": "mock-select-item",
        "data-value": value,
        onClick: () => onSelect?.(value),
        ...props,
      },
      children,
    ),
  SelectTrigger: ({ children, ...props }: any) =>
    React.createElement(
      "div",
      {
        "data-testid": "mock-select-trigger",
        ...props,
      },
      children,
    ),
  SelectValue: ({ placeholder, ...props }: any) =>
    React.createElement(
      "span",
      {
        "data-testid": "mock-select-value",
        ...props,
      },
      placeholder,
    ),

  // Layout components
  Tabs: createUIComponentMock("Tabs"),
  TabsContent: createUIComponentMock("TabsContent"),
  TabsList: createUIComponentMock("TabsList"),
  TabsTrigger: createUIComponentMock("TabsTrigger"),

  ScrollArea: createUIComponentMock("ScrollArea"),
  Separator: createUIComponentMock("Separator"),

  // Feedback components
  Alert: createUIComponentMock("Alert"),
  AlertDescription: createUIComponentMock("AlertDescription"),
  AlertTitle: createUIComponentMock("AlertTitle"),
  Badge: createUIComponentMock("Badge"),
  Progress: createUIComponentMock("Progress"),

  // Interactive components
  Dialog: createInteractiveComponentMock("Dialog"),
  DialogContent: createInteractiveComponentMock("DialogContent"),
  DialogDescription: createUIComponentMock("DialogDescription"),
  DialogHeader: createUIComponentMock("DialogHeader"),
  DialogTitle: createUIComponentMock("DialogTitle"),
  DialogTrigger: createUIComponentMock("DialogTrigger"),

  Popover: createInteractiveComponentMock("Popover"),
  PopoverContent: createInteractiveComponentMock("PopoverContent"),
  PopoverTrigger: createUIComponentMock("PopoverTrigger"),

  // Advanced form components
  Slider: ({
    value,
    onValueChange,
    min = 0,
    max = 100,
    step = 1,
    ...props
  }: any) =>
    React.createElement("input", {
      type: "range",
      value: Array.isArray(value) ? value[0] : value,
      onChange: (e: any) => onValueChange?.([Number(e.target.value)]),
      min,
      max,
      step,
      "data-testid": "mock-slider",
      ...props,
    }),

  Switch: ({ checked, onCheckedChange, ...props }: any) =>
    React.createElement("input", {
      type: "checkbox",
      checked,
      onChange: (e: any) => onCheckedChange?.(e.target.checked),
      "data-testid": "mock-switch",
      ...props,
    }),

  Checkbox: ({ checked, onCheckedChange, ...props }: any) =>
    React.createElement("input", {
      type: "checkbox",
      checked,
      onChange: (e: any) => onCheckedChange?.(e.target.checked),
      "data-testid": "mock-checkbox",
      ...props,
    }),

  // Table components
  Table: createUIComponentMock("Table", { includeDataTestIds: true }),
  TableBody: createUIComponentMock("TableBody"),
  TableCell: createUIComponentMock("TableCell"),
  TableHead: createUIComponentMock("TableHead"),
  TableHeader: createUIComponentMock("TableHeader"),
  TableRow: createUIComponentMock("TableRow"),

  // Calendar and date components
  Calendar: ({ selected, onSelect, ...props }: any) =>
    React.createElement(
      "div",
      {
        "data-testid": "mock-calendar",
        ...props,
      },
      [
        React.createElement(
          "button",
          {
            key: "today-btn",
            onClick: () => onSelect?.(new Date()),
            "data-testid": "mock-calendar-today",
          },
          "Today",
        ),
      ],
    ),

  // Accordion components
  Accordion: createUIComponentMock("Accordion"),
  AccordionItem: createUIComponentMock("AccordionItem"),
  AccordionTrigger: createUIComponentMock("AccordionTrigger"),
  AccordionContent: createUIComponentMock("AccordionContent"),

  // Command components
  Command: createUIComponentMock("Command"),
  CommandEmpty: createUIComponentMock("CommandEmpty"),
  CommandGroup: createUIComponentMock("CommandGroup"),
  CommandInput: createUIComponentMock("CommandInput"),
  CommandItem: createUIComponentMock("CommandItem"),
  CommandList: createUIComponentMock("CommandList"),

  // Form components
  Form: createFormComponentMock("Form"),
  FormControl: createUIComponentMock("FormControl"),
  FormDescription: createUIComponentMock("FormDescription"),
  FormField: createUIComponentMock("FormField"),
  FormItem: createUIComponentMock("FormItem"),
  FormLabel: createUIComponentMock("FormLabel"),
  FormMessage: createUIComponentMock("FormMessage"),
};

/**
 * Creates jest mock modules for all UI components
 * This reduces the need to manually mock each component in every test file
 */
export const createUIComponentMocks = () => {
  const mockModules: Record<string, any> = {};

  // Create mock modules for each UI component path
  const componentGroups = [
    "card",
    "button",
    "input",
    "label",
    "textarea",
    "select",
    "tabs",
    "scroll-area",
    "separator",
    "alert",
    "badge",
    "progress",
    "dialog",
    "popover",
    "slider",
    "switch",
    "checkbox",
    "table",
    "calendar",
    "accordion",
    "command",
    "form",
  ];

  componentGroups.forEach((componentName) => {
    const modulePath = `@/components/ui/${componentName}`;
    mockModules[modulePath] = () => {
      const exports: Record<string, any> = {};

      // Add all related component exports for this module
      Object.entries(UI_COMPONENT_MOCKS).forEach(([name, mock]) => {
        if (name.toLowerCase().includes(componentName.replace("-", ""))) {
          exports[name] = mock;
        }
      });

      return exports;
    };
  });

  return mockModules;
};

/**
 * Agent template mock factory following the established interface
 */
export const createAgentTemplateMock = (overrides: Partial<any> = {}) => ({
  id: "mock-explorer",
  name: "Mock Explorer Agent",
  description: "Mock agent template for testing",
  icon: React.createElement("span", { "data-testid": "mock-icon" }, "MockIcon"),
  category: "explorer" as const,
  complexity: "beginner" as const,
  mathematicalFoundation: {
    beliefsStates: 64,
    observationModalities: 8,
    actionSpaces: 4,
    defaultPrecision: {
      sensory: 0.8,
      policy: 0.7,
      state: 0.9,
    },
  },
  capabilities: ["exploration", "discovery", "learning"],
  useCases: ["environment mapping", "resource discovery"],
  expertRecommendation: "Ideal for exploration tasks",
  ...overrides,
});

/**
 * Hook mocks following established patterns
 */
export const HOOK_MOCKS = {
  useToast: () => ({
    toast: jest.fn(),
    dismiss: jest.fn(),
    toasts: [],
  }),
  useDebounce: (value: any, delay?: number) => value,
};

/**
 * Utility mocks following established patterns
 */
export const UTILITY_MOCKS = {
  cn: (...args: any[]) => args.filter(Boolean).join(" "),
  formatTimestamp: (date: Date) => date.toISOString(),
  format: (date: Date, formatStr: string) => date.toISOString().split("T")[0],
};

/**
 * Audit logger mock following established patterns from backend MockFactory
 */
export const AUDIT_LOGGER_MOCK = {
  auditLogger: {
    log: jest.fn(),
    getEntries: jest.fn(() => Promise.resolve([])),
    getStats: jest.fn(() =>
      Promise.resolve({
        totalEntries: 0,
        recentActivity: [],
        complianceMetrics: {
          totalHighRiskOperations: 0,
          totalMediumRiskOperations: 0,
          totalLowRiskOperations: 0,
          totalOperations: 0,
          highRiskPercentage: 0,
          mediumRiskPercentage: 0,
          lowRiskPercentage: 0,
          complianceScore: 100,
          lastAssessment: new Date().toISOString(),
          averageOperationDuration: 125.5,
          successRate: 99.2,
          failureRate: 0.8,
          riskTrends: {
            increasing: 0,
            stable: 0,
            decreasing: 0,
          },
          recentOperations: [],
          auditTrail: [],
        },
        systemMetrics: {
          totalConfigurations: 0,
          activeAgents: 0,
          averagePerformance: 0,
          systemUptime: 0,
          errorRate: 0,
        },
      }),
    ),
    exportLogs: jest.fn(() => []),
    clearLogs: jest.fn(),
  },
  logBoundaryEdit: jest.fn(),
  logTemplateSelection: jest.fn(),
  logThresholdChange: jest.fn(),
  // Add missing AuditLogEntry and related types
  AuditLogEntry: {},
  AuditLogFilter: {},
  AuditLogStats: {},
  ExportOptions: {},
};
