import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SimulationGrid } from "@/components/main/SimulationGrid";
import { useSimulation } from "@/hooks/use-simulation";
import { useAgents } from "@/hooks/use-agents";

// Mock the hooks
jest.mock("@/hooks/use-simulation");
jest.mock("@/hooks/use-agents");

const mockUseSimulation = useSimulation as jest.MockedFunction<typeof useSimulation>;
const mockUseAgents = useAgents as jest.MockedFunction<typeof useAgents>;

describe("SimulationGrid", () => {
  const mockStartSimulation = jest.fn();
  const mockStopSimulation = jest.fn();
  const mockResetSimulation = jest.fn();
  const mockStepSimulation = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    // Default simulation state
    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: [],
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 20, height: 20 },
      setGridSize: jest.fn(),
    });

    // Default agents state
    mockUseAgents.mockReturnValue({
      agents: [],
      isLoading: false,
      error: null,
      createAgent: jest.fn(),
      updateAgent: jest.fn(),
      deleteAgent: jest.fn(),
    });
  });

  it("renders simulation grid", () => {
    render(<SimulationGrid />);

    expect(screen.getByRole("heading", { name: /simulation grid/i })).toBeInTheDocument();
    expect(screen.getByText(/step: 0/i)).toBeInTheDocument();
  });

  it("shows empty state when no simulation", () => {
    render(<SimulationGrid />);

    expect(screen.getByText(/no active simulation/i)).toBeInTheDocument();
    expect(screen.getByText(/create agents and start/i)).toBeInTheDocument();
  });

  it("renders grid cells when simulation has data", () => {
    const gridData = Array(10)
      .fill(null)
      .map((_, y) =>
        Array(10)
          .fill(null)
          .map((_, x) => ({
            x,
            y,
            type: "empty" as const,
            occupant: null,
          })),
      );

    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: gridData,
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 10, height: 10 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    expect(screen.getByTestId("simulation-canvas")).toBeInTheDocument();
  });

  it("displays agents on the grid", () => {
    const agents = [
      {
        id: "agent-1",
        name: "Explorer",
        position: { x: 5, y: 5 },
        state: "exploring" as const,
        energy: 80,
      },
      {
        id: "agent-2",
        name: "Collector",
        position: { x: 3, y: 7 },
        state: "collecting" as const,
        energy: 60,
      },
    ];

    const gridData = Array(10)
      .fill(null)
      .map((_, y) =>
        Array(10)
          .fill(null)
          .map((_, x) => ({
            x,
            y,
            type: "empty" as const,
            occupant: agents.find((a) => a.position.x === x && a.position.y === y) || null,
          })),
      );

    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 5,
      grid: gridData,
      agents,
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 10, height: 10 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    // Should show agent count
    expect(screen.getByText(/2 agents/i)).toBeInTheDocument();
  });

  it("provides simulation controls", () => {
    render(<SimulationGrid />);

    expect(screen.getByRole("button", { name: /start/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /step/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reset/i })).toBeInTheDocument();
  });

  it("starts simulation when start button clicked", async () => {
    const user = userEvent.setup();

    // Provide registered agents to enable the start button
    mockUseAgents.mockReturnValue({
      agents: [
        {
          id: "agent-1",
          name: "Test Agent",
          type: "explorer",
          description: "Test",
          status: "idle",
          createdAt: new Date().toISOString(),
        },
      ],
      isLoading: false,
      error: null,
      createAgent: jest.fn(),
      updateAgent: jest.fn(),
      deleteAgent: jest.fn(),
    });

    render(<SimulationGrid />);

    await user.click(screen.getByRole("button", { name: /start/i }));

    expect(mockStartSimulation).toHaveBeenCalled();
  });

  it("shows pause button when simulation is running", () => {
    mockUseSimulation.mockReturnValue({
      isRunning: true,
      isPaused: false,
      currentStep: 10,
      grid: [],
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 20, height: 20 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    expect(screen.getByRole("button", { name: /pause/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start/i })).not.toBeInTheDocument();
  });

  it("provides speed control", () => {
    render(<SimulationGrid />);

    expect(screen.getByLabelText(/speed/i)).toBeInTheDocument();
    expect(screen.getByText(/1x/i)).toBeInTheDocument();
  });

  it.skip("changes simulation speed", async () => {
    const user = userEvent.setup();
    const mockSetSpeed = jest.fn();

    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: [],
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: mockSetSpeed,
      gridSize: { width: 20, height: 20 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    // For Radix UI Slider, we need to find the slider root element
    const speedSlider = screen.getByRole("slider", { name: /speed/i });

    // Simulate changing the value through Radix UI's onValueChange
    // This would typically be done through keyboard or mouse interaction
    await user.keyboard("{ArrowRight}");

    // Or directly trigger the value change (this is more reliable in tests)
    const sliderParent = speedSlider.parentElement;
    if (sliderParent) {
      const onValueChange = (
        sliderParent as unknown as { __reactProps$?: { onValueChange?: (value: number[]) => void } }
      ).__reactProps$?.onValueChange;
      if (onValueChange) {
        onValueChange([2]);
      }
    }

    // Check if setSpeed was called
    await waitFor(() => {
      expect(mockSetSpeed).toHaveBeenCalled();
    });
  });

  it("steps simulation when step button clicked", async () => {
    const user = userEvent.setup();
    render(<SimulationGrid />);

    await user.click(screen.getByRole("button", { name: /step/i }));

    expect(mockStepSimulation).toHaveBeenCalled();
  });

  it("resets simulation when reset button clicked", async () => {
    const user = userEvent.setup();
    render(<SimulationGrid />);

    await user.click(screen.getByRole("button", { name: /reset/i }));

    expect(mockResetSimulation).toHaveBeenCalled();
  });

  it.skip("shows agent details on hover", async () => {
    const agents = [
      {
        id: "agent-1",
        name: "Explorer",
        position: { x: 5, y: 5 },
        state: "exploring" as const,
        energy: 80,
        beliefs: { explored: 10, resources: 3 },
      },
    ];

    const gridData = Array(10)
      .fill(null)
      .map((_, y) =>
        Array(10)
          .fill(null)
          .map((_, x) => ({
            x,
            y,
            type: x === 5 && y === 5 ? ("agent" as const) : ("empty" as const),
            occupant: agents.find((a) => a.position.x === x && a.position.y === y) || null,
          })),
      );

    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: gridData,
      agents,
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 10, height: 10 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    // Mock canvas hover interaction
    const canvas = screen.getByTestId("simulation-canvas");

    // Mock getBoundingClientRect for the canvas
    canvas.getBoundingClientRect = jest.fn(() => ({
      left: 0,
      top: 0,
      right: 300,
      bottom: 300,
      width: 300,
      height: 300,
      x: 0,
      y: 0,
      toJSON: jest.fn(),
    }));

    // Fire mouse move event at the position of the agent (5, 5) on a 10x10 grid
    // Each cell is 30px, so agent at (5,5) is at pixel (150, 150)
    fireEvent.mouseMove(canvas, {
      clientX: 150,
      clientY: 150,
    });

    await waitFor(() => {
      expect(screen.getByTestId("agent-tooltip")).toBeInTheDocument();
      expect(screen.getByText(/Explorer/)).toBeInTheDocument();
      expect(screen.getByText(/Energy: 80/i)).toBeInTheDocument();
    });
  });

  it("shows resource objects on grid", () => {
    const gridData = Array(10)
      .fill(null)
      .map((_, y) =>
        Array(10)
          .fill(null)
          .map((_, x) => ({
            x,
            y,
            type: x === 2 && y === 3 ? ("resource" as const) : ("empty" as const),
            occupant: null,
            resource: x === 2 && y === 3 ? { type: "food", amount: 10 } : undefined,
          })),
      );

    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: gridData,
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 10, height: 10 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    expect(screen.getByTestId("simulation-canvas")).toBeInTheDocument();
  });

  it("allows grid size adjustment", () => {
    render(<SimulationGrid />);

    expect(screen.getByLabelText(/^Grid Width$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^Height$/i)).toBeInTheDocument();
  });

  it.skip("changes grid size", async () => {
    const user = userEvent.setup();
    const mockSetGridSize = jest.fn();

    mockUseSimulation.mockReturnValue({
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      grid: [],
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 20, height: 20 },
      setGridSize: mockSetGridSize,
    });

    render(<SimulationGrid />);

    const widthInput = screen.getByLabelText(/^Grid Width$/i);
    await user.clear(widthInput);
    await user.type(widthInput, "30");

    await waitFor(() => {
      expect(mockSetGridSize).toHaveBeenCalledWith(expect.objectContaining({ width: 30 }));
    });
  });

  it("disables controls during simulation", () => {
    mockUseSimulation.mockReturnValue({
      isRunning: true,
      isPaused: false,
      currentStep: 10,
      grid: [],
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 1,
      setSpeed: jest.fn(),
      gridSize: { width: 20, height: 20 },
      setGridSize: jest.fn(),
    });

    render(<SimulationGrid />);

    expect(screen.getByLabelText(/Grid Width/i)).toBeDisabled();
    expect(screen.getByLabelText(/^Height$/i)).toBeDisabled();
  });

  it("shows performance metrics", () => {
    mockUseSimulation.mockReturnValue({
      isRunning: true,
      isPaused: false,
      currentStep: 100,
      grid: [],
      agents: [],
      startSimulation: mockStartSimulation,
      stopSimulation: mockStopSimulation,
      pauseSimulation: jest.fn(),
      resumeSimulation: jest.fn(),
      stepSimulation: mockStepSimulation,
      resetSimulation: mockResetSimulation,
      speed: 2,
      setSpeed: jest.fn(),
      gridSize: { width: 20, height: 20 },
      setGridSize: jest.fn(),
      metrics: {
        fps: 30,
        avgStepTime: 15,
        totalSteps: 100,
      },
    });

    render(<SimulationGrid />);

    // Performance metrics should appear in the controls section as badges
    const fpsBadge = screen.getAllByText(/30 FPS/i).find(
      (element) => element.classList.contains("inline-flex"), // Badge component class
    );
    const stepTimeBadge = screen.getAllByText(/15ms\/step/i).find(
      (element) => element.classList.contains("inline-flex"), // Badge component class
    );

    expect(fpsBadge).toBeInTheDocument();
    expect(stepTimeBadge).toBeInTheDocument();
  });
});
