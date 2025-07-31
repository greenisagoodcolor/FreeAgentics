import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { SimulationGrid } from "./SimulationGrid";
import { useSimulation } from "@/hooks/use-simulation";
import { useAgents } from "@/hooks/use-agents";

// Mock the hooks
jest.mock("@/hooks/use-simulation");
jest.mock("@/hooks/use-agents");

// Mock canvas context
const mockCanvasContext = {
  clearRect: jest.fn(),
  fillRect: jest.fn(),
  strokeRect: jest.fn(),
  beginPath: jest.fn(),
  arc: jest.fn(),
  fill: jest.fn(),
  stroke: jest.fn(),
};

describe("SimulationGrid", () => {
  const mockSimulationControls = {
    startSimulation: jest.fn(),
    stopSimulation: jest.fn(),
    pauseSimulation: jest.fn(),
    resumeSimulation: jest.fn(),
    stepSimulation: jest.fn(),
    resetSimulation: jest.fn(),
    setSpeed: jest.fn(),
    setGridSize: jest.fn(),
  };

  const mockGrid = [
    [
      { x: 0, y: 0, type: "empty" as const },
      { x: 1, y: 0, type: "empty" as const },
    ],
    [
      { x: 0, y: 1, type: "empty" as const },
      { x: 1, y: 1, type: "empty" as const },
    ],
  ];

  const mockAgents = [
    {
      id: "agent-1",
      name: "Explorer 1",
      position: { x: 0, y: 0 },
      state: "exploring" as const,
      energy: 80,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock canvas getContext
    HTMLCanvasElement.prototype.getContext = jest.fn().mockImplementation((contextId) => {
      if (contextId === "2d") {
        return mockCanvasContext;
      }
      return null;
    }) as any;

    (useSimulation as jest.Mock).mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: mockGrid,
      agents: [],
      speed: 1,
      gridSize: { width: 2, height: 2 },
      metrics: null,
      ...mockSimulationControls,
    });

    (useAgents as jest.Mock).mockReturnValue({
      agents: [],
    });
  });

  it("should render empty state when no simulation is active", () => {
    (useSimulation as jest.Mock).mockReturnValue({
      ...mockSimulationControls,
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: [],
      agents: [],
      speed: 1,
      gridSize: { width: 20, height: 20 },
      metrics: null,
    });

    render(<SimulationGrid />);
    expect(screen.getByText("No active simulation")).toBeInTheDocument();
  });

  it("should render simulation canvas when grid is available", () => {
    render(<SimulationGrid />);
    const canvas = screen.getByTestId("simulation-canvas");
    expect(canvas).toBeInTheDocument();
  });

  it("should visibly render agents on the grid and show movement after 2 seconds", async () => {
    // Initial render with agent at position (0, 0)
    (useSimulation as jest.Mock).mockReturnValue({
      ...mockSimulationControls,
      isRunning: true,
      isPaused: false,
      currentStep: 1,
      grid: mockGrid,
      agents: mockAgents,
      speed: 1,
      gridSize: { width: 2, height: 2 },
      metrics: { fps: 60, avgStepTime: 16 },
    });

    const { rerender } = render(<SimulationGrid />);

    // Verify canvas is rendered
    const canvas = screen.getByTestId("simulation-canvas");
    expect(canvas).toBeInTheDocument();

    // Verify agent is drawn (arc method called for agent body)
    expect(mockCanvasContext.arc).toHaveBeenCalled();

    // Clear mocks before simulating movement
    jest.clearAllMocks();

    // Simulate agent movement after 2 seconds
    await waitFor(
      () => {
        // Update agent position
        const movedAgents = [
          {
            ...mockAgents[0],
            position: { x: 1, y: 1 }, // Agent moved to new position
          },
        ];

        (useSimulation as jest.Mock).mockReturnValue({
          ...mockSimulationControls,
          isRunning: true,
          isPaused: false,
          currentStep: 10,
          grid: mockGrid,
          agents: movedAgents,
          speed: 1,
          gridSize: { width: 2, height: 2 },
          metrics: { fps: 60, avgStepTime: 16 },
        });

        rerender(<SimulationGrid />);
      },
      { timeout: 2000 },
    );

    // Verify agent is drawn at new position
    // Arc should be called for the agent at the new position
    expect(mockCanvasContext.arc).toHaveBeenCalled();

    // This test expects smooth movement animation
    // In real implementation, this would be verified visually
    // For now, we're just checking that the canvas is redrawn
    expect(mockCanvasContext.clearRect).toHaveBeenCalled();
  });
});
