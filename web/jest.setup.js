// Optional: configure or set up a testing framework before each test.
require("@testing-library/jest-dom");
require("fake-indexeddb/auto");
const React = require("react");

// Add structuredClone polyfill for Node < 17
if (typeof structuredClone === "undefined") {
  global.structuredClone = (obj) => {
    return JSON.parse(JSON.stringify(obj));
  };
}

// Mock IndexedDB with proper setup
const FDBFactory = require("fake-indexeddb/lib/FDBFactory");
const FDBKeyRange = require("fake-indexeddb/lib/FDBKeyRange");

global.indexedDB = new FDBFactory();
global.IDBKeyRange = FDBKeyRange;

// Mock Next.js router
jest.mock("next/router", () => ({
  useRouter() {
    return {
      route: "/",
      pathname: "/",
      query: {},
      asPath: "/",
      push: jest.fn(),
      pop: jest.fn(),
      reload: jest.fn(),
      back: jest.fn(),
      prefetch: jest.fn().mockResolvedValue(undefined),
      beforePopState: jest.fn(),
      events: {
        on: jest.fn(),
        off: jest.fn(),
        emit: jest.fn(),
      },
      isFallback: false,
    };
  },
}));

// Mock Next.js navigation
jest.mock("next/navigation", () => ({
  useRouter() {
    return {
      push: jest.fn(),
      replace: jest.fn(),
      refresh: jest.fn(),
      back: jest.fn(),
      forward: jest.fn(),
      prefetch: jest.fn(),
    };
  },
  useSearchParams() {
    return new URLSearchParams();
  },
  usePathname() {
    return "/";
  },
}));

// Mock window.matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock fetch globally to prevent real API calls
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ data: "mocked" }),
    text: () => Promise.resolve("mocked response"),
  }),
);

// Mock all WebSocket connections to prevent hanging
global.WebSocket = jest.fn(() => ({
  close: jest.fn(),
  send: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: 1, // OPEN
}));

// Mock timers and intervals to prevent hanging
jest.spyOn(global, "setTimeout").mockImplementation((fn, delay = 0) => {
  if (delay > 100) {
    // For long delays, don't execute immediately to prevent test hanging
    return 1;
  }
  fn();
  return 1;
});

jest.spyOn(global, "setInterval").mockImplementation((fn, interval = 0) => {
  if (interval > 100) {
    // For long intervals, don't execute to prevent test hanging
    return 1;
  }
  fn();
  return 1;
});

// Mock any async operations that might hang
global.requestAnimationFrame = jest.fn((cb) => {
  cb(0);
  return 1;
});

global.cancelAnimationFrame = jest.fn();

// Mock UI components that might be causing import issues
jest.mock("@/components/ui/button", () => {
  return {
    Button: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("button", props, props.children);
    },
  };
});

jest.mock("@/components/ui/badge", () => {
  return {
    Badge: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("span", props, props.children);
    },
  };
});

jest.mock("@/components/ui/scroll-area", () => {
  return {
    ScrollArea: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
  };
});

jest.mock("@/components/ui/dialog", () => {
  return {
    Dialog: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    DialogContent: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    DialogHeader: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    DialogTitle: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("h2", props, props.children);
    },
    DialogTrigger: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
  };
});

jest.mock("@/components/ui/input", () => {
  return {
    Input: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("input", props);
    },
  };
});

jest.mock("@/components/ui/label", () => {
  return {
    Label: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("label", props, props.children);
    },
  };
});

jest.mock("@/components/ui/card", () => {
  return {
    Card: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    CardHeader: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    CardContent: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    CardTitle: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("h3", props, props.children);
    },
  };
});

jest.mock("@/components/ui/accordion", () => {
  return {
    Accordion: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    AccordionContent: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    AccordionItem: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("div", props, props.children);
    },
    AccordionTrigger: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("button", props, props.children);
    },
  };
});

// Centralized Radix UI Select mock to prevent conflicts across tests
jest.mock("@/components/ui/select", () => {
  return {
    Select: function ({ children, value, onValueChange, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement(
        "div",
        { "data-testid": "select-root", "data-value": value, ...props },
        children,
      );
    },
    SelectContent: function ({ children, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement(
        "div",
        { role: "listbox", ...props },
        children,
      );
    },
    SelectItem: function ({ children, value, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement(
        "div",
        { role: "option", "data-value": value, ...props },
        children,
      );
    },
    SelectTrigger: function ({ children, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement(
        "button",
        { role: "combobox", ...props },
        children,
      );
    },
    SelectValue: function ({ placeholder, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement("span", props, placeholder || "Select...");
    },
    SelectGroup: function ({ children, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement(
        "div",
        { role: "group", ...props },
        children,
      );
    },
    SelectLabel: function ({ children, ...props }) {
      const mockReact = require("react");
      return mockReact.createElement("label", props, children);
    },
    SelectSeparator: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("hr", props);
    },
  };
});

jest.mock("@/components/ui/slider", () => {
  return {
    Slider: function (props) {
      const mockReact = require("react");
      return mockReact.createElement("input", { type: "range", ...props });
    },
  };
});

// Mock AgentTemplateSelector component
jest.mock("@/components/dashboard/AgentTemplateSelector", () => {
  return function AgentTemplateSelector() {
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "agent-template-selector" },
      "Agent Template Selector Mock",
    );
  };
});

// Mock dashboard panel components with relative paths
jest.mock("@/app/dashboard/components/panels/AgentPanel", () => {
  return function AgentPanel({ view }) {
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "agent-panel-mock", view },
      "Agent Panel Mock",
    );
  };
});

jest.mock("@/app/dashboard/components/panels/ConversationPanel", () => {
  return function ConversationPanel({ view }) {
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "conversation-panel-mock", view },
      "Conversation Panel Mock",
    );
  };
});

jest.mock("@/app/dashboard/components/panels/GoalPanel", () => {
  return function GoalPanel({ view }) {
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "goal-panel-mock", view },
      "Goal Panel Mock",
    );
  };
});

jest.mock("@/app/dashboard/components/panels/KnowledgePanel", () => {
  return function KnowledgePanel({ view }) {
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "knowledge-panel-mock", view },
      "Knowledge Panel Mock",
    );
  };
});

jest.mock("@/app/dashboard/components/panels/MetricsPanel", () => {
  return function MetricsPanel({ view }) {
    const mockReact = require("react");
    return mockReact.createElement(
      "div",
      { "data-testid": "metrics-panel-mock", view },
      "Metrics Panel Mock",
    );
  };
});

// Mock Redux store with proper structure matching actual store
jest.mock("@/store/hooks", () => ({
  useAppSelector: jest.fn((selector) => {
    const mockState = {
      agents: {
        agents: {
          "test-agent-1": {
            id: "test-agent-1",
            name: "Test Explorer",
            templateId: "explorer",
            biography: "A test explorer agent",
            knowledgeDomains: ["exploration", "testing"],
            parameters: {
              responseThreshold: 0.6,
              turnTakingProbability: 0.7,
              conversationEngagement: 0.8,
            },
            status: "active",
            avatarUrl: "/avatars/explorer.svg",
            color: "#10B981",
            createdAt: Date.now(),
            lastActive: Date.now(),
            inConversation: true,
            autonomyEnabled: true,
            activityMetrics: {
              messagesCount: 5,
              beliefCount: 3,
              responseTime: [200, 300, 250],
            },
          },
        },
        templates: {
          explorer: {
            id: "explorer",
            name: "Explorer",
            category: "researcher",
            defaultBiography:
              "An adventurous agent that discovers new territories",
            defaultKnowledgeDomains: ["exploration", "mapping", "discovery"],
            defaultParameters: {
              responseThreshold: 0.6,
              turnTakingProbability: 0.7,
              conversationEngagement: 0.8,
            },
            avatarUrl: "/avatars/explorer.svg",
            icon: "Search",
            color: "#10B981",
          },
        },
        selectedAgentId: null,
        typingAgents: [],
        agentOrder: ["test-agent-1"],
      },
      conversations: { all: [], active: null },
      ui: { theme: "dark", sidebarOpen: true },
      connection: { isConnected: false, socket: null },
      knowledge: { nodes: [], edges: [] },
      analytics: { metrics: {} },
    };
    return selector(mockState);
  }),
  useAppDispatch: jest.fn(() => jest.fn()),
}));

// Mock console methods to reduce test noise and prevent hanging
const originalError = console.error;
const originalWarn = console.warn;
const originalLog = console.log;
const originalInfo = console.info;

beforeAll(() => {
  // Suppress all console output to prevent hanging from verbose logging
  console.error = jest.fn();
  console.warn = jest.fn();
  console.log = jest.fn();
  console.info = jest.fn();
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
  console.log = originalLog;
  console.info = originalInfo;
});
