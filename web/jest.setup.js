// Optional: configure or set up a testing framework before each test.
import "@testing-library/jest-dom";
import "fake-indexeddb/auto";
import React from "react";

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

// Mock fetch globally
global.fetch = jest.fn();

// Mock UI components that might be causing import issues
jest.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }) => React.createElement("button", props, children),
}));

jest.mock("@/components/ui/badge", () => ({
  Badge: ({ children, ...props }) => React.createElement("span", props, children),
}));

jest.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children, ...props }) => React.createElement("div", props, children),
}));

jest.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children, ...props }) => React.createElement("div", props, children),
  DialogContent: ({ children, ...props }) => React.createElement("div", props, children),
  DialogHeader: ({ children, ...props }) => React.createElement("div", props, children),
  DialogTitle: ({ children, ...props }) => React.createElement("h2", props, children),
  DialogTrigger: ({ children, ...props }) => React.createElement("div", props, children),
}));

jest.mock("@/components/ui/input", () => ({
  Input: (props) => React.createElement("input", props),
}));

jest.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }) => React.createElement("label", props, children),
}));

jest.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }) => React.createElement("div", props, children),
  CardHeader: ({ children, ...props }) => React.createElement("div", props, children),
  CardContent: ({ children, ...props }) => React.createElement("div", props, children),
  CardTitle: ({ children, ...props }) => React.createElement("h3", props, children),
}));

jest.mock("@/components/ui/accordion", () => ({
  Accordion: ({ children, ...props }) => React.createElement("div", props, children),
  AccordionContent: ({ children, ...props }) => React.createElement("div", props, children),
  AccordionItem: ({ children, ...props }) => React.createElement("div", props, children),
  AccordionTrigger: ({ children, ...props }) => React.createElement("button", props, children),
}));

jest.mock("@/components/ui/slider", () => ({
  Slider: (props) => React.createElement("input", { type: "range", ...props }),
}));

// Mock AgentTemplateSelector component
jest.mock("@/components/dashboard/AgentTemplateSelector", () => {
  return function AgentTemplateSelector() {
    return React.createElement("div", { "data-testid": "agent-template-selector" }, "Agent Template Selector Mock");
  };
});

// Mock dashboard panel components directly to avoid complex dependency chains
// Using absolute paths that match the file structure
jest.mock("/Users/matthewmoroney/builds/FreeAgentics/web/app/dashboard/components/panels/AgentPanel", () => {
  return function AgentPanel({ view }) {
    return React.createElement("div", { "data-testid": "agent-panel-mock", view }, "Agent Panel Mock");
  };
});

jest.mock("/Users/matthewmoroney/builds/FreeAgentics/web/app/dashboard/components/panels/ConversationPanel", () => {
  return function ConversationPanel({ view }) {
    return React.createElement("div", { "data-testid": "conversation-panel-mock", view }, "Conversation Panel Mock");
  };
});

jest.mock("/Users/matthewmoroney/builds/FreeAgentics/web/app/dashboard/components/panels/GoalPanel", () => {
  return function GoalPanel({ view }) {
    return React.createElement("div", { "data-testid": "goal-panel-mock", view }, "Goal Panel Mock");
  };
});

jest.mock("/Users/matthewmoroney/builds/FreeAgentics/web/app/dashboard/components/panels/KnowledgePanel", () => {
  return function KnowledgePanel({ view }) {
    return React.createElement("div", { "data-testid": "knowledge-panel-mock", view }, "Knowledge Panel Mock");
  };
});

jest.mock("/Users/matthewmoroney/builds/FreeAgentics/web/app/dashboard/components/panels/MetricsPanel", () => {
  return function MetricsPanel({ view }) {
    return React.createElement("div", { "data-testid": "metrics-panel-mock", view }, "Metrics Panel Mock");
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
          }
        },
        templates: {
          explorer: {
            id: "explorer",
            name: "Explorer",
            category: "researcher",
            defaultBiography: "An adventurous agent that discovers new territories",
            defaultKnowledgeDomains: ["exploration", "mapping", "discovery"],
            defaultParameters: {
              responseThreshold: 0.6,
              turnTakingProbability: 0.7,
              conversationEngagement: 0.8,
            },
            avatarUrl: "/avatars/explorer.svg",
            icon: "Search",
            color: "#10B981",
          }
        },
        selectedAgentId: null,
        typingAgents: [],
        agentOrder: ["test-agent-1"],
      },
      conversations: { all: [], active: null },
      ui: { theme: 'dark', sidebarOpen: true },
      connection: { isConnected: false, socket: null },
      knowledge: { nodes: [], edges: [] },
      analytics: { metrics: {} }
    };
    return selector(mockState);
  }),
  useAppDispatch: jest.fn(() => jest.fn()),
}));

// Mock console methods to reduce test noise
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  console.error = (...args) => {
    if (
      typeof args[0] === "string" &&
      args[0].includes("Warning: ReactDOM.render is no longer supported")
    ) {
      return;
    }
    originalError.call(console, ...args);
  };

  console.warn = (...args) => {
    if (
      typeof args[0] === "string" &&
      (args[0].includes("componentWillReceiveProps") ||
        args[0].includes("componentWillUpdate"))
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
});
