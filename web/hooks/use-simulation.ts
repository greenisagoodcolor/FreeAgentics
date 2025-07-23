import { useState, useCallback, useEffect, useRef } from "react";
import { useWebSocket } from "./use-websocket";
import { useAgents } from "./use-agents";

export type CellType = "empty" | "wall" | "resource" | "agent";

export interface GridCell {
  x: number;
  y: number;
  type: CellType;
  occupant: SimulationAgent | null;
  resource?: {
    type: string;
    amount: number;
  };
}

export interface SimulationAgent {
  id: string;
  name: string;
  position: { x: number; y: number };
  state: "idle" | "exploring" | "collecting" | "returning";
  energy: number;
  beliefs?: Record<string, unknown>;
}

export interface SimulationMetrics {
  fps: number;
  avgStepTime: number;
  totalSteps: number;
}

export interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  currentStep: number;
  grid: GridCell[][];
  agents: SimulationAgent[];
  startSimulation: () => void;
  stopSimulation: () => void;
  pauseSimulation: () => void;
  resumeSimulation: () => void;
  stepSimulation: () => void;
  resetSimulation: () => void;
  speed: number;
  setSpeed: (speed: number) => void;
  gridSize: { width: number; height: number };
  setGridSize: (size: { width: number; height: number }) => void;
  metrics?: SimulationMetrics;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useSimulation(): SimulationState {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [grid, setGrid] = useState<GridCell[][]>([]);
  const [agents, setAgents] = useState<SimulationAgent[]>([]);
  const [speed, setSpeed] = useState(1);
  const [gridSize, setGridSize] = useState({ width: 20, height: 20 });
  const [metrics, setMetrics] = useState<SimulationMetrics>();

  const { sendMessage, lastMessage } = useWebSocket();
  const { agents: registeredAgents } = useAgents();

  const animationFrameRef = useRef<number>();
  const lastStepTimeRef = useRef<number>(Date.now());
  const stepTimesRef = useRef<number[]>([]);

  // Initialize empty grid
  useEffect(() => {
    const newGrid: GridCell[][] = Array(gridSize.height)
      .fill(null)
      .map((_, y) =>
        Array(gridSize.width)
          .fill(null)
          .map((_, x) => ({
            x,
            y,
            type: "empty" as CellType,
            occupant: null,
          })),
      );
    setGrid(newGrid);
  }, [gridSize]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "simulation_update") {
      const { step, grid: newGrid, agents: newAgents, metrics: newMetrics } = lastMessage.data;
      setCurrentStep(step);
      if (newGrid) setGrid(newGrid);
      if (newAgents) setAgents(newAgents);
      if (newMetrics) setMetrics(newMetrics);
    } else if (lastMessage.type === "simulation_started") {
      setIsRunning(true);
      setIsPaused(false);
    } else if (lastMessage.type === "simulation_stopped") {
      setIsRunning(false);
      setIsPaused(false);
    } else if (lastMessage.type === "simulation_paused") {
      setIsPaused(true);
    } else if (lastMessage.type === "simulation_resumed") {
      setIsPaused(false);
    }
  }, [lastMessage]);

  // Animation loop for running simulation
  useEffect(() => {
    if (isRunning && !isPaused) {
      const animate = () => {
        const now = Date.now();
        const deltaTime = now - lastStepTimeRef.current;

        // Step based on speed (steps per second)
        if (deltaTime >= 1000 / speed) {
          stepSimulation();
          lastStepTimeRef.current = now;

          // Track step times for FPS calculation
          stepTimesRef.current.push(deltaTime);
          if (stepTimesRef.current.length > 30) {
            stepTimesRef.current.shift();
          }

          // Calculate FPS
          const avgStepTime =
            stepTimesRef.current.reduce((a, b) => a + b, 0) / stepTimesRef.current.length;
          const fps = Math.round(1000 / avgStepTime);

          setMetrics((prev) => ({
            fps,
            avgStepTime: Math.round(avgStepTime),
            totalSteps: currentStep,
            ...prev,
          }));
        }

        animationFrameRef.current = requestAnimationFrame(animate);
      };

      animationFrameRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRunning, isPaused, speed, currentStep]);

  const startSimulation = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/simulation/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gridSize,
          agents: registeredAgents.map((agent) => ({
            id: agent.id,
            name: agent.name,
            type: agent.type,
          })),
        }),
      });

      if (response.ok) {
        setIsRunning(true);
        setIsPaused(false);
        sendMessage({ type: "simulation_started" });
      }
    } catch (error) {
      console.error("Failed to start simulation:", error);
    }
  }, [gridSize, registeredAgents, sendMessage]);

  const stopSimulation = useCallback(() => {
    setIsRunning(false);
    setIsPaused(false);
    sendMessage({ type: "stop_simulation" });
  }, [sendMessage]);

  const pauseSimulation = useCallback(() => {
    setIsPaused(true);
    sendMessage({ type: "pause_simulation" });
  }, [sendMessage]);

  const resumeSimulation = useCallback(() => {
    setIsPaused(false);
    sendMessage({ type: "resume_simulation" });
  }, [sendMessage]);

  const stepSimulation = useCallback(() => {
    sendMessage({ type: "step_simulation" });
  }, [sendMessage]);

  const resetSimulation = useCallback(() => {
    setIsRunning(false);
    setIsPaused(false);
    setCurrentStep(0);
    setAgents([]);
    stepTimesRef.current = [];

    // Reset grid
    const newGrid: GridCell[][] = Array(gridSize.height)
      .fill(null)
      .map((_, y) =>
        Array(gridSize.width)
          .fill(null)
          .map((_, x) => ({
            x,
            y,
            type: "empty" as CellType,
            occupant: null,
          })),
      );
    setGrid(newGrid);

    sendMessage({ type: "reset_simulation" });
  }, [gridSize, sendMessage]);

  const handleSetGridSize = useCallback(
    (size: { width: number; height: number }) => {
      if (!isRunning) {
        setGridSize(size);
      }
    },
    [isRunning],
  );

  return {
    isRunning,
    isPaused,
    currentStep,
    grid,
    agents,
    startSimulation,
    stopSimulation,
    pauseSimulation,
    resumeSimulation,
    stepSimulation,
    resetSimulation,
    speed,
    setSpeed,
    gridSize,
    setGridSize: handleSetGridSize,
    metrics,
  };
}
