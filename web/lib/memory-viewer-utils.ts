export interface MemoryData {
  timestamp: number;
  value: number;
  label?: string;
}

export interface MemoryMetrics {
  total: number;
  used: number;
  free: number;
  percentage: number;
}

export interface MemoryChartOptions {
  width?: number;
  height?: number;
  padding?: number;
  showGrid?: boolean;
  showLabels?: boolean;
  color?: string;
}

export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

export function calculateMemoryPercentage(used: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((used / total) * 100);
}

export function generateMemoryChartData(data: MemoryData[], maxPoints = 50): MemoryData[] {
  if (data.length <= maxPoints) {
    return data;
  }

  // Sample data points evenly, ensuring we don't exceed maxPoints
  const step = Math.floor(data.length / (maxPoints - 1)); // Reserve space for last point
  const sampled: MemoryData[] = [];

  for (let i = 0; i < data.length && sampled.length < maxPoints - 1; i += step) {
    sampled.push(data[i]);
  }

  // Always include the last point
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }

  return sampled;
}

export function getMemoryStatus(percentage: number): {
  status: "normal" | "warning" | "critical";
  color: string;
} {
  if (percentage < 70) {
    return { status: "normal", color: "#10b981" }; // green
  } else if (percentage <= 85) {
    return { status: "warning", color: "#f59e0b" }; // amber
  } else {
    return { status: "critical", color: "#ef4444" }; // red
  }
}

export function createMemoryChart(
  canvas: HTMLCanvasElement,
  data: MemoryData[],
  options: MemoryChartOptions = {},
): void {
  const {
    width = canvas.width,
    height = canvas.height,
    padding = 20,
    showGrid = true,
    showLabels = true,
    color = "#3b82f6",
  } = options;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  if (data.length === 0) return;

  // Calculate chart dimensions
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  // Find min and max values
  const values = data.map((d) => d.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueRange = maxValue - minValue || 1;

  // Draw grid
  if (showGrid) {
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }
  }

  // Draw chart line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  data.forEach((point, index) => {
    const x = padding + (index / (data.length - 1)) * chartWidth;
    const y = padding + chartHeight - ((point.value - minValue) / valueRange) * chartHeight;

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();

  // Draw labels
  if (showLabels) {
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";

    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const value = minValue + (valueRange / 5) * (5 - i);
      const y = padding + (chartHeight / 5) * i;
      ctx.fillText(formatBytes(value), 5, y + 4);
    }

    // X-axis labels (show first and last)
    if (data.length > 0) {
      const firstLabel = data[0].label || new Date(data[0].timestamp).toLocaleTimeString();
      const lastLabel =
        data[data.length - 1].label ||
        new Date(data[data.length - 1].timestamp).toLocaleTimeString();

      ctx.fillText(firstLabel, padding, height - 5);
      ctx.textAlign = "right";
      ctx.fillText(lastLabel, width - padding, height - 5);
    }
  }
}

export function aggregateMemoryData(data: MemoryData[], intervalMs: number): MemoryData[] {
  if (data.length === 0) return [];

  const aggregated: MemoryData[] = [];
  let currentInterval: MemoryData[] = [];
  let intervalStart = data[0].timestamp;

  data.forEach((point) => {
    if (point.timestamp - intervalStart < intervalMs) {
      currentInterval.push(point);
    } else {
      // Calculate average for the interval
      if (currentInterval.length > 0) {
        const avgValue =
          currentInterval.reduce((sum, p) => sum + p.value, 0) / currentInterval.length;
        aggregated.push({
          timestamp: intervalStart,
          value: avgValue,
          label: new Date(intervalStart).toLocaleTimeString(),
        });
      }

      // Start new interval
      currentInterval = [point];
      intervalStart = point.timestamp;
    }
  });

  // Don't forget the last interval
  if (currentInterval.length > 0) {
    const avgValue = currentInterval.reduce((sum, p) => sum + p.value, 0) / currentInterval.length;
    aggregated.push({
      timestamp: intervalStart,
      value: avgValue,
      label: new Date(intervalStart).toLocaleTimeString(),
    });
  }

  return aggregated;
}
