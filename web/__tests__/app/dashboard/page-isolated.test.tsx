/**
 * @jest-environment jsdom
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";

// Mock dashboard panel components before importing HomePage
jest.mock("../../../app/dashboard/components/panels/AgentPanel", () => {
  return function AgentPanel({ view }: { view: string }) {
    return <div data-testid="agent-panel-mock">Agent Panel Mock - {view}</div>;
  };
});

jest.mock("../../../app/dashboard/components/panels/ConversationPanel", () => {
  return function ConversationPanel({ view }: { view: string }) {
    return (
      <div data-testid="conversation-panel-mock">
        Conversation Panel Mock - {view}
      </div>
    );
  };
});

jest.mock("../../../app/dashboard/components/panels/GoalPanel", () => {
  return function GoalPanel({ view }: { view: string }) {
    return <div data-testid="goal-panel-mock">Goal Panel Mock - {view}</div>;
  };
});

jest.mock("../../../app/dashboard/components/panels/KnowledgePanel", () => {
  return function KnowledgePanel({ view }: { view: string }) {
    return (
      <div data-testid="knowledge-panel-mock">
        Knowledge Panel Mock - {view}
      </div>
    );
  };
});

jest.mock("../../../app/dashboard/components/panels/MetricsPanel", () => {
  return function MetricsPanel({ view }: { view: string }) {
    return (
      <div data-testid="metrics-panel-mock">Metrics Panel Mock - {view}</div>
    );
  };
});

// Mock Next.js router
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    refresh: jest.fn(),
  }),
}));

// Now import HomePage after mocks are set up
import HomePage from "../../../app/page";

describe("HomePage (Isolated Test)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders without crashing with mocked components", () => {
    render(<HomePage />);

    // Check that mocked components are rendered
    expect(screen.getByTestId("agent-panel-mock")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-panel-mock")).toBeInTheDocument();
    expect(screen.getByTestId("goal-panel-mock")).toBeInTheDocument();
    expect(screen.getByTestId("knowledge-panel-mock")).toBeInTheDocument();
    expect(screen.getByTestId("metrics-panel-mock")).toBeInTheDocument();
  });

  test("dashboard panels receive correct view prop", () => {
    render(<HomePage />);

    // Check that panels receive the default view
    expect(
      screen.getByText(/Agent Panel Mock - executive/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Conversation Panel Mock - executive/),
    ).toBeInTheDocument();
  });
});
