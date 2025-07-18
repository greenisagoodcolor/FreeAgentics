import React from "react";
import { render, screen } from "@testing-library/react";
import MainPage from "@/app/main/page";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Mock the child components
jest.mock("@/components/main/PromptBar", () => ({
  PromptBar: () => <div data-testid="prompt-bar">Prompt Bar</div>,
}));

jest.mock("@/components/main/AgentCreatorPanel", () => ({
  AgentCreatorPanel: () => <div data-testid="agent-creator-panel">Agent Creator Panel</div>,
}));

jest.mock("@/components/main/ConversationWindow", () => ({
  ConversationWindow: () => <div data-testid="conversation-window">Conversation Window</div>,
}));

jest.mock("@/components/main/KnowledgeGraphView", () => ({
  KnowledgeGraphView: () => <div data-testid="knowledge-graph-view">Knowledge Graph View</div>,
}));

jest.mock("@/components/main/SimulationGrid", () => ({
  SimulationGrid: () => <div data-testid="simulation-grid">Simulation Grid</div>,
}));

jest.mock("@/components/main/MetricsFooter", () => ({
  MetricsFooter: () => <div data-testid="metrics-footer">Metrics Footer</div>,
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("MainPage", () => {
  it("renders all main sections", () => {
    const Wrapper = createWrapper();
    render(<MainPage />, { wrapper: Wrapper });

    // Check all main components are rendered
    expect(screen.getByTestId("prompt-bar")).toBeInTheDocument();
    expect(screen.getByTestId("agent-creator-panel")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-window")).toBeInTheDocument();
    expect(screen.getByTestId("knowledge-graph-view")).toBeInTheDocument();
    expect(screen.getByTestId("simulation-grid")).toBeInTheDocument();
    expect(screen.getByTestId("metrics-footer")).toBeInTheDocument();
  });

  it("has correct layout structure", () => {
    const Wrapper = createWrapper();
    const { container } = render(<MainPage />, { wrapper: Wrapper });

    // Check for main layout container
    const mainContainer = container.querySelector(".main-layout");
    expect(mainContainer).toBeInTheDocument();

    // Check for grid layout in main row
    const mainRow = container.querySelector(".main-row");
    expect(mainRow).toHaveClass("grid", "grid-cols-1", "lg:grid-cols-3");
  });

  it("is responsive - stacks columns on small screens", () => {
    const Wrapper = createWrapper();
    const { container } = render(<MainPage />, { wrapper: Wrapper });

    const mainRow = container.querySelector(".main-row");
    expect(mainRow).toHaveClass("grid-cols-1", "lg:grid-cols-3");
  });

  it("has sticky top bar", () => {
    const Wrapper = createWrapper();
    const { container } = render(<MainPage />, { wrapper: Wrapper });

    const topBar = container.querySelector(".top-bar");
    expect(topBar).toHaveClass("sticky", "top-0");
  });

  it("has fixed height footer", () => {
    const Wrapper = createWrapper();
    const { container } = render(<MainPage />, { wrapper: Wrapper });

    const footer = container.querySelector(".footer-bar");
    expect(footer).toHaveClass("h-12"); // 48px = h-12 in Tailwind
  });
});
