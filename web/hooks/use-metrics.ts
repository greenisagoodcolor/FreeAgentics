import { useState, useEffect, useCallback } from "react";

export interface SystemMetrics {
  cpu: number; // CPU usage percentage
  memory: number; // Memory usage percentage
  agents: number; // Active agent count
  messages: number; // Total message count
  uptime: number; // Uptime in seconds
  version: string; // System version
  avgFreeEnergy?: number; // Average free energy across agents
}

export interface MetricsState {
  metrics: SystemMetrics | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const POLLING_INTERVAL = 5000; // Poll every 5 seconds

export function useMetrics(): MetricsState {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/v1/metrics`, {
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.statusText}`);
      }

      const data = await response.json();

      setMetrics({
        cpu: data.cpu_usage || data.cpu || 0,
        memory: data.memory_usage || data.memory || 0,
        agents: data.active_agents || data.agents || 0,
        messages: data.total_inferences || data.messages || 0,
        uptime: data.uptime || 0,
        version: data.version || "1.0.0-alpha",
        avgFreeEnergy: data.avg_free_energy !== undefined ? data.avg_free_energy : undefined,
      });

      setIsLoading(false);
    } catch (err) {
      setError(err as Error);
      setIsLoading(false);
      console.error("Error fetching metrics:", err);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  // Polling
  useEffect(() => {
    const interval = setInterval(fetchMetrics, POLLING_INTERVAL);

    return () => {
      clearInterval(interval);
    };
  }, [fetchMetrics]);

  return {
    metrics,
    isLoading,
    error,
    refetch: fetchMetrics,
  };
}
