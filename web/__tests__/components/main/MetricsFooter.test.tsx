import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { MetricsFooter } from "@/components/main/MetricsFooter";
import { useMetrics } from "@/hooks/use-metrics";

// Mock the hook
jest.mock("@/hooks/use-metrics");

const mockUseMetrics = useMetrics as jest.MockedFunction<typeof useMetrics>;

describe("MetricsFooter", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Default metrics state
    mockUseMetrics.mockReturnValue({
      metrics: {
        cpu: 45.5,
        memory: 67.8,
        agents: 5,
        messages: 1234,
        uptime: 3600,
        version: "1.0.0-alpha",
      },
      isLoading: false,
      error: null,
      refetch: jest.fn(),
    });
  });

  it("renders metrics footer", () => {
    render(<MetricsFooter />);

    expect(screen.getByTestId("metrics-footer")).toBeInTheDocument();
  });

  it("displays CPU usage", () => {
    render(<MetricsFooter />);

    expect(screen.getByText(/CPU/i)).toBeInTheDocument();
    expect(screen.getByText(/45.5%/)).toBeInTheDocument();
  });

  it("displays memory usage", () => {
    render(<MetricsFooter />);

    expect(screen.getByText(/Memory/i)).toBeInTheDocument();
    expect(screen.getByText(/67.8%/)).toBeInTheDocument();
  });

  it("displays agent count", () => {
    render(<MetricsFooter />);

    expect(screen.getByText(/Agents/i)).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("displays message count", () => {
    render(<MetricsFooter />);

    expect(screen.getByText(/Messages/i)).toBeInTheDocument();
    expect(screen.getByText(/1.2K/)).toBeInTheDocument(); // 1234 gets formatted as 1.2K
  });

  it("displays uptime", () => {
    render(<MetricsFooter />);

    expect(screen.getByText(/Uptime/i)).toBeInTheDocument();
    expect(screen.getByText(/1h 0m/)).toBeInTheDocument();
  });

  it("displays version", () => {
    render(<MetricsFooter />);

    expect(screen.getByText(/1.0.0-alpha/)).toBeInTheDocument();
  });

  it("shows loading state", () => {
    mockUseMetrics.mockReturnValue({
      metrics: null,
      isLoading: true,
      error: null,
      refetch: jest.fn(),
    });

    render(<MetricsFooter />);

    expect(screen.getByTestId("metrics-loading")).toBeInTheDocument();
  });

  it("shows error state", () => {
    mockUseMetrics.mockReturnValue({
      metrics: null,
      isLoading: false,
      error: new Error("Failed to fetch metrics"),
      refetch: jest.fn(),
    });

    render(<MetricsFooter />);

    expect(screen.getByText(/Unable to load metrics/i)).toBeInTheDocument();
  });

  it("shows high CPU warning", () => {
    mockUseMetrics.mockReturnValue({
      metrics: {
        cpu: 85.5,
        memory: 50,
        agents: 5,
        messages: 1234,
        uptime: 3600,
        version: "1.0.0-alpha",
      },
      isLoading: false,
      error: null,
      refetch: jest.fn(),
    });

    render(<MetricsFooter />);

    const cpuElement = screen.getByText(/85.5%/);
    expect(cpuElement).toHaveClass("text-destructive");
  });

  it("shows high memory warning", () => {
    mockUseMetrics.mockReturnValue({
      metrics: {
        cpu: 50,
        memory: 92.5,
        agents: 5,
        messages: 1234,
        uptime: 3600,
        version: "1.0.0-alpha",
      },
      isLoading: false,
      error: null,
      refetch: jest.fn(),
    });

    render(<MetricsFooter />);

    const memoryElement = screen.getByText(/92.5%/);
    expect(memoryElement).toHaveClass("text-destructive");
  });

  it("formats large message counts", () => {
    mockUseMetrics.mockReturnValue({
      metrics: {
        cpu: 45.5,
        memory: 67.8,
        agents: 5,
        messages: 1234567,
        uptime: 3600,
        version: "1.0.0-alpha",
      },
      isLoading: false,
      error: null,
      refetch: jest.fn(),
    });

    render(<MetricsFooter />);

    expect(screen.getByText(/1.2M/)).toBeInTheDocument();
  });

  it("formats long uptime", () => {
    mockUseMetrics.mockReturnValue({
      metrics: {
        cpu: 45.5,
        memory: 67.8,
        agents: 5,
        messages: 1234,
        uptime: 93784, // ~26 hours
        version: "1.0.0-alpha",
      },
      isLoading: false,
      error: null,
      refetch: jest.fn(),
    });

    render(<MetricsFooter />);

    expect(screen.getByText(/1d 2h/)).toBeInTheDocument();
  });

  it.skip("updates metrics periodically", async () => {
    const mockRefetch = jest.fn();
    mockUseMetrics.mockReturnValue({
      metrics: {
        cpu: 45.5,
        memory: 67.8,
        agents: 5,
        messages: 1234,
        uptime: 3600,
        version: "1.0.0-alpha",
      },
      isLoading: false,
      error: null,
      refetch: mockRefetch,
    });

    render(<MetricsFooter />);

    // Wait for polling interval
    await waitFor(
      () => {
        expect(mockRefetch).toHaveBeenCalled();
      },
      { timeout: 6000 },
    );
  });
});
