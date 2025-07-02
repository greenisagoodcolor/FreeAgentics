import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AgentCard from "@/components/agentcard";
import type { Agent } from "@/lib/types";
import {
  AgentStatus,
  AgentResources,
  AgentGoal,
  AgentPersonality,
} from "@/lib/types/agent-api";
import { KnowledgeEntry } from "@/lib/types";

// Mock UI components
jest.mock("@/components/ui/badge", () => ({
  Badge: ({ children, className, variant }: any) => (
    <span className={className} data-testid="badge" data-variant={variant}>
      {children}
    </span>
  ),
}));

jest.mock("@/components/ui/card", () => ({
  Card: ({ children, className, onClick }: any) => (
    <div className={className} onClick={onClick} data-testid="card">
      {children}
    </div>
  ),
  CardContent: ({ children, className }: any) => (
    <div className={className} data-testid="card-content">
      {children}
    </div>
  ),
  CardHeader: ({ children, className }: any) => (
    <div className={className} data-testid="card-header">
      {children}
    </div>
  ),
}));

jest.mock("@/components/ui/progress", () => ({
  Progress: ({ value, className }: any) => (
    <div
      className={className}
      data-testid="progress"
      data-value={value}
      role="progressbar"
      aria-valuenow={value}
    >
      {value}%
    </div>
  ),
}));

jest.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: any) => (
    <div data-testid="tooltip-provider">{children}</div>
  ),
  Tooltip: ({ children }: any) => <div data-testid="tooltip">{children}</div>,
  TooltipTrigger: ({ children }: any) => (
    <div data-testid="tooltip-trigger">{children}</div>
  ),
  TooltipContent: ({ children }: any) => (
    <div data-testid="tooltip-content">{children}</div>
  ),
}));

// Mock icons
jest.mock("lucide-react", () => {
  const createMockIcon =
    (name: string) =>
    ({ size, className }: any) => (
      <span
        className={className}
        data-testid={`${name.toLowerCase()}-icon`}
        style={{ fontSize: size }}
      >
        {name}
      </span>
    );

  return {
    Activity: createMockIcon("Activity"),
    AlertCircle: createMockIcon("AlertCircle"),
    Battery: createMockIcon("Battery"),
    Brain: createMockIcon("Brain"),
    CheckCircle: createMockIcon("CheckCircle"),
    Clock: createMockIcon("Clock"),
    Heart: createMockIcon("Heart"),
    Power: createMockIcon("Power"),
    PowerOff: createMockIcon("PowerOff"),
    Target: createMockIcon("Target"),
    Users: createMockIcon("Users"),
    Zap: createMockIcon("Zap"),
  };
});

// Sample test data
const mockAgent: Agent = {
  id: "agent-1",
  name: "Test Agent",
  color: "#FF5733",
  role: "helper",
  personality: {
    openness: 0.8,
    conscientiousness: 0.7,
    extraversion: 0.9,
    agreeableness: 0.8,
    neuroticism: 0.3,
  } as AgentPersonality,
  status: "interacting" as AgentStatus,
  position: { x: 10, y: 20 },
  autonomyEnabled: true,
  inConversation: false,
  knowledge: [
    { id: "k1", title: "Knowledge 1", content: "Content 1", timestamp: new Date(), tags: [] },
    { id: "k2", title: "Knowledge 2", content: "Content 2", timestamp: new Date(), tags: [] },
    { id: "k3", title: "Knowledge 3", content: "Content 3", timestamp: new Date(), tags: [] },
  ] as KnowledgeEntry[],
};

const mockAgentData = {
  status: "interacting" as AgentStatus,
  resources: {
    energy: 75,
    health: 90,
    memory_used: 250,
    memory_capacity: 500,
  } as AgentResources,
  goals: [
    {
      id: "goal-1",
      description: "Complete important task",
      priority: 0.8,
      deadline: new Date(Date.now() + 86400000).toISOString(),
    },
    {
      id: "goal-2",
      description: "Learn new skills",
      priority: 0.5,
      deadline: new Date(Date.now() + 172800000).toISOString(),
    },
    {
      id: "goal-3",
      description: "Monitor system health",
      priority: 0.3,
      deadline: new Date(Date.now() + 259200000).toISOString(),
    },
  ] as AgentGoal[],
  activity: "Processing user request",
};

describe("AgentCard Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Component Rendering", () => {
    it("renders basic agent information", () => {
      render(<AgentCard agent={mockAgent} />);

      expect(screen.getByTestId("card")).toBeInTheDocument();
      expect(screen.getByText("Test Agent")).toBeInTheDocument();
      expect(screen.getByText("Position: (10, 20)")).toBeInTheDocument();
      expect(screen.getByText("Knowledge entries: 3")).toBeInTheDocument();
    });

    it("renders agent color indicator", () => {
      render(<AgentCard agent={mockAgent} />);

      const colorIndicator = document.querySelector(
        '[style*="background-color: rgb(255, 87, 51)"]',
      );
      expect(colorIndicator).toBeInTheDocument();
    });

    it("renders status badge with correct status", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      const badge = screen.getByTestId("badge");
      expect(badge).toHaveTextContent("interacting");
      expect(screen.getByTestId("users-icon")).toBeInTheDocument();
    });

    it("renders autonomy indicator when enabled", () => {
      render(<AgentCard agent={mockAgent} />);

      expect(screen.getByTestId("power-icon")).toBeInTheDocument();
    });

    it("renders autonomy indicator when disabled", () => {
      const disabledAgent = { ...mockAgent, autonomyEnabled: false };
      render(<AgentCard agent={disabledAgent} />);

      const powerOffIcons = screen.getAllByTestId("poweroff-icon");
      expect(powerOffIcons.length).toBeGreaterThan(0);
    });

    it("applies selected styling when isSelected is true", () => {
      render(<AgentCard agent={mockAgent} isSelected={true} />);

      const card = screen.getByTestId("card");
      expect(card).toHaveClass("ring-2", "ring-primary");
    });

    it("applies custom className", () => {
      render(<AgentCard agent={mockAgent} className="custom-class" />);

      const card = screen.getByTestId("card");
      expect(card).toHaveClass("custom-class");
    });
  });

  describe("Resource Display", () => {
    it("renders energy progress bar", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      const energyIcons = screen.getAllByTestId("battery-icon");
      expect(energyIcons.length).toBeGreaterThan(0);

      const progressBars = screen.getAllByTestId("progress");
      const energyProgress = progressBars.find(
        (bar) => bar.getAttribute("data-value") === "75",
      );
      expect(energyProgress).toBeInTheDocument();

      const percentageTexts = screen.getAllByText("75%");
      expect(percentageTexts.length).toBeGreaterThan(0);
    });

    it("renders health progress bar", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      const healthIcons = screen.getAllByTestId("heart-icon");
      expect(healthIcons.length).toBeGreaterThan(0);

      const progressBars = screen.getAllByTestId("progress");
      const healthProgress = progressBars.find(
        (bar) => bar.getAttribute("data-value") === "90",
      );
      expect(healthProgress).toBeInTheDocument();

      const percentageTexts = screen.getAllByText("90%");
      expect(percentageTexts.length).toBeGreaterThan(0);
    });

    it("renders memory progress bar with calculated percentage", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      const memoryIcons = screen.getAllByTestId("brain-icon");
      expect(memoryIcons.length).toBeGreaterThan(0);

      // Memory usage: 250/500 = 50%
      const progressBars = screen.getAllByTestId("progress");
      const memoryProgress = progressBars.find(
        (bar) => bar.getAttribute("data-value") === "50",
      );
      expect(memoryProgress).toBeInTheDocument();

      const percentageTexts = screen.getAllByText("50%");
      expect(percentageTexts.length).toBeGreaterThan(0);
    });

    it("handles zero memory capacity gracefully", () => {
      const dataWithZeroCapacity = {
        ...mockAgentData,
        resources: {
          ...mockAgentData.resources,
          memory_capacity: 0,
        },
      };

      render(<AgentCard agent={mockAgent} agentData={dataWithZeroCapacity} />);

      const progressBars = screen.getAllByTestId("progress");
      // Should default to 0% when capacity is 0
      const memoryProgress = progressBars.find(
        (bar) => bar.getAttribute("data-value") === "0",
      );
      expect(memoryProgress).toBeInTheDocument();
    });

    it("renders default resources when agentData is not provided", () => {
      render(<AgentCard agent={mockAgent} />);

      // Should show default values of 0 for energy and health
      const progressBars = screen.getAllByTestId("progress");
      expect(progressBars).toHaveLength(3); // energy, health, memory

      // All should show 0%
      const zeroProgress = progressBars.filter(
        (bar) => bar.getAttribute("data-value") === "0",
      );
      expect(zeroProgress).toHaveLength(3);
    });
  });

  describe("Status Handling", () => {
    const statusTestCases: Array<{
      status: AgentStatus;
      icon: string;
      color: string;
    }> = [
      { status: "idle", icon: "clock-icon", color: "bg-gray-500" },
      { status: "moving", icon: "activity-icon", color: "bg-blue-500" },
      { status: "interacting", icon: "users-icon", color: "bg-green-500" },
      { status: "planning", icon: "brain-icon", color: "bg-yellow-500" },
      { status: "executing", icon: "zap-icon", color: "bg-orange-500" },
      { status: "learning", icon: "brain-icon", color: "bg-purple-500" },
      { status: "error", icon: "alertcircle-icon", color: "bg-red-500" },
      { status: "offline", icon: "poweroff-icon", color: "bg-gray-700" },
    ];

    statusTestCases.forEach(({ status, icon, color }) => {
      it(`renders correct icon and color for ${status} status`, () => {
        const statusData = { ...mockAgentData, status };
        render(<AgentCard agent={mockAgent} agentData={statusData} />);

        // Some icons might appear multiple times (e.g., poweroff-icon for autonomy and status)
        const icons = screen.getAllByTestId(icon);
        expect(icons.length).toBeGreaterThan(0);
        expect(screen.getByText(status)).toBeInTheDocument();

        const badge = screen.getByTestId("badge");
        expect(badge).toHaveClass(color);
      });
    });

    it("defaults to offline status when no agentData provided", () => {
      render(<AgentCard agent={mockAgent} />);

      expect(screen.getByTestId("poweroff-icon")).toBeInTheDocument();
      expect(screen.getByText("offline")).toBeInTheDocument();
    });
  });

  describe("Activity Display", () => {
    it("renders current activity when provided", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      expect(screen.getByText("Activity:")).toBeInTheDocument();
      expect(screen.getByText("Processing user request")).toBeInTheDocument();
    });

    it("does not render activity section when not provided", () => {
      const dataWithoutActivity = { ...mockAgentData, activity: undefined };
      render(<AgentCard agent={mockAgent} agentData={dataWithoutActivity} />);

      expect(screen.queryByText("Activity:")).not.toBeInTheDocument();
    });

    it("does not render activity section when empty string", () => {
      const dataWithEmptyActivity = { ...mockAgentData, activity: "" };
      render(<AgentCard agent={mockAgent} agentData={dataWithEmptyActivity} />);

      expect(screen.queryByText("Activity:")).not.toBeInTheDocument();
    });
  });

  describe("Goals Display", () => {
    it("renders goals section with target icon", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      expect(screen.getByTestId("target-icon")).toBeInTheDocument();
      expect(screen.getByText("Active Goals:")).toBeInTheDocument();
    });

    it("renders individual goals with appropriate icons", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      expect(screen.getByText("Complete important task")).toBeInTheDocument();
      expect(screen.getByText("Learn new skills")).toBeInTheDocument();

      // High priority goal should have CheckCircle icon
      const checkIcons = screen.getAllByTestId("checkcircle-icon");
      expect(checkIcons.length).toBeGreaterThan(0);

      // Lower priority goal should have Clock icon
      const clockIcons = screen.getAllByTestId("clock-icon");
      expect(clockIcons.length).toBeGreaterThan(0);
    });

    it("limits displayed goals to 2 and shows remaining count", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      expect(screen.getByText("Complete important task")).toBeInTheDocument();
      expect(screen.getByText("Learn new skills")).toBeInTheDocument();
      expect(
        screen.queryByText("Monitor system health"),
      ).not.toBeInTheDocument();
      expect(screen.getByText("+1 more goals")).toBeInTheDocument();
    });

    it("does not show more goals text when exactly 2 goals", () => {
      const dataWithTwoGoals = {
        ...mockAgentData,
        goals: mockAgentData.goals.slice(0, 2),
      };

      render(<AgentCard agent={mockAgent} agentData={dataWithTwoGoals} />);

      expect(screen.queryByText(/more goals/)).not.toBeInTheDocument();
    });

    it("does not render goals section when no goals provided", () => {
      const dataWithoutGoals = { ...mockAgentData, goals: [] };
      render(<AgentCard agent={mockAgent} agentData={dataWithoutGoals} />);

      expect(screen.queryByText("Active Goals:")).not.toBeInTheDocument();
    });

    it("does not render goals section when goals array is undefined", () => {
      const dataWithUndefinedGoals = { ...mockAgentData, goals: undefined };
      render(
        <AgentCard agent={mockAgent} agentData={dataWithUndefinedGoals} />,
      );

      expect(screen.queryByText("Active Goals:")).not.toBeInTheDocument();
    });
  });

  describe("Interaction Handling", () => {
    it("calls onClick when card is clicked", async () => {
      const mockOnClick = jest.fn();
      const user = userEvent.setup();

      render(<AgentCard agent={mockAgent} onClick={mockOnClick} />);

      const card = screen.getByTestId("card");
      await user.click(card);

      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it("does not crash when onClick is not provided", async () => {
      const user = userEvent.setup();

      render(<AgentCard agent={mockAgent} />);

      const card = screen.getByTestId("card");
      await user.click(card);

      // Should not throw error
    });

    it("has cursor-pointer class for clickable card", () => {
      render(<AgentCard agent={mockAgent} onClick={jest.fn()} />);

      const card = screen.getByTestId("card");
      expect(card).toHaveClass("cursor-pointer");
    });
  });

  describe("Tooltip Integration", () => {
    it("renders tooltip provider and triggers", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      expect(screen.getByTestId("tooltip-provider")).toBeInTheDocument();

      const tooltipTriggers = screen.getAllByTestId("tooltip-trigger");
      expect(tooltipTriggers.length).toBeGreaterThan(0);
    });

    it("provides tooltips for resource indicators", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      const tooltipContents = screen.getAllByTestId("tooltip-content");
      expect(tooltipContents.length).toBeGreaterThan(0);
    });
  });

  describe("Edge Cases", () => {
    it("handles agent with missing position gracefully", () => {
      const agentWithoutPosition = { ...mockAgent, position: { x: 0, y: 0 } };

      expect(() => {
        render(<AgentCard agent={agentWithoutPosition} />);
      }).not.toThrow();

      expect(screen.getByText("Position: (0, 0)")).toBeInTheDocument();
    });

    it("handles agent with empty knowledge array", () => {
      const agentWithEmptyKnowledge = { ...mockAgent, knowledge: [] };

      render(<AgentCard agent={agentWithEmptyKnowledge} />);

      expect(screen.getByText("Knowledge entries: 0")).toBeInTheDocument();
    });

    it("handles undefined agent data gracefully", () => {
      expect(() => {
        render(<AgentCard agent={mockAgent} agentData={undefined} />);
      }).not.toThrow();

      // Should show default offline status
      expect(screen.getByText("offline")).toBeInTheDocument();
    });

    it("handles partial agent data", () => {
      const partialData = {
        status: "idle" as AgentStatus,
        // Missing resources and goals - should use defaults
      };

      expect(() => {
        render(<AgentCard agent={mockAgent} agentData={partialData as any} />);
      }).not.toThrow();

      // Should render with the provided status
      expect(screen.getByText("idle")).toBeInTheDocument();
    });

    it("handles very long agent names", () => {
      const longNameAgent = { ...mockAgent, name: "A".repeat(100) };

      render(<AgentCard agent={longNameAgent} />);

      expect(screen.getByText("A".repeat(100))).toBeInTheDocument();
    });

    it("handles very long goal descriptions", () => {
      const longGoalData = {
        ...mockAgentData,
        goals: [
          {
            id: "long-goal",
            description:
              "This is a very long goal description that should be truncated properly by CSS".repeat(
                10,
              ),
            priority: 0.5,
            deadline: new Date(Date.now() + 86400000).toISOString(),
          },
        ] as AgentGoal[],
      };

      expect(() => {
        render(<AgentCard agent={mockAgent} agentData={longGoalData} />);
      }).not.toThrow();
    });
  });

  describe("Accessibility", () => {
    it("has proper ARIA attributes for progress bars", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      const progressBars = screen.getAllByRole("progressbar");
      expect(progressBars.length).toBe(3);

      progressBars.forEach((bar) => {
        expect(bar).toHaveAttribute("aria-valuenow");
      });
    });

    it("provides meaningful text content", () => {
      render(<AgentCard agent={mockAgent} agentData={mockAgentData} />);

      expect(screen.getByText("Test Agent")).toBeInTheDocument();
      expect(screen.getByText("Position: (10, 20)")).toBeInTheDocument();
      expect(screen.getByText("Knowledge entries: 3")).toBeInTheDocument();
    });

    it("has clickable card element", () => {
      const mockOnClick = jest.fn();
      render(<AgentCard agent={mockAgent} onClick={mockOnClick} />);

      const card = screen.getByTestId("card");
      fireEvent.click(card);

      expect(mockOnClick).toHaveBeenCalled();
    });
  });

  describe("Performance", () => {
    it("renders quickly with complex data", () => {
      const complexData = {
        ...mockAgentData,
        goals: Array.from({ length: 20 }, (_, i) => ({
          id: `goal-${i}`,
          description: `Goal ${i} description`,
          priority: Math.random(),
          deadline: new Date(Date.now() + i * 86400000).toISOString(),
        })) as AgentGoal[],
      };

      const startTime = performance.now();
      render(<AgentCard agent={mockAgent} agentData={complexData} />);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100);
    });

    it("handles rapid prop changes efficiently", () => {
      const { rerender } = render(<AgentCard agent={mockAgent} />);

      for (let i = 0; i < 10; i++) {
        const newAgent = { ...mockAgent, name: `Agent ${i}` };
        rerender(<AgentCard agent={newAgent} />);
      }

      expect(screen.getByText("Agent 9")).toBeInTheDocument();
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts and unmounts without errors", () => {
      const { unmount } = render(<AgentCard agent={mockAgent} />);

      expect(screen.getByText("Test Agent")).toBeInTheDocument();

      expect(() => unmount()).not.toThrow();
    });

    it("handles prop updates correctly", () => {
      const { rerender } = render(
        <AgentCard agent={mockAgent} isSelected={false} />,
      );

      let card = screen.getByTestId("card");
      expect(card).not.toHaveClass("ring-2");

      rerender(<AgentCard agent={mockAgent} isSelected={true} />);

      card = screen.getByTestId("card");
      expect(card).toHaveClass("ring-2");
    });
  });
});
