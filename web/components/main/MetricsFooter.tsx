"use client";

import React from "react";
import { Cpu, MemoryStick, Users, MessageSquare, Clock, AlertCircle, Loader2 } from "lucide-react";
import { useMetrics } from "@/hooks/use-metrics";
import { cn } from "@/lib/utils";

const HIGH_CPU_THRESHOLD = 80;
const HIGH_MEMORY_THRESHOLD = 85;

function formatNumber(num: number): string {
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M`;
  }
  if (num >= 1_000) {
    return `${(num / 1_000).toFixed(1)}K`;
  }
  return num.toString();
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (days > 0) {
    return `${days}d ${hours}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

interface MetricItemProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  warning?: boolean;
  className?: string;
}

function MetricItem({ icon, label, value, warning, className }: MetricItemProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="flex items-center gap-1.5">
        {icon}
        <span className="text-xs text-muted-foreground">{label}:</span>
      </div>
      <span className={cn("text-xs font-medium", warning && "text-destructive")}>{value}</span>
    </div>
  );
}

export function MetricsFooter() {
  const { metrics, isLoading, error } = useMetrics();

  if (isLoading && !metrics) {
    return (
      <footer
        data-testid="metrics-footer"
        className="h-12 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
      >
        <div className="h-full flex items-center justify-center px-4">
          <div
            data-testid="metrics-loading"
            className="flex items-center gap-2 text-xs text-muted-foreground"
          >
            <Loader2 className="h-3 w-3 animate-spin" />
            Loading metrics...
          </div>
        </div>
      </footer>
    );
  }

  if (error || !metrics) {
    return (
      <footer
        data-testid="metrics-footer"
        className="h-12 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
      >
        <div className="h-full flex items-center justify-center px-4">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <AlertCircle className="h-3 w-3" />
            Unable to load metrics
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer
      data-testid="metrics-footer"
      className="h-12 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    >
      <div className="h-full flex items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <MetricItem
            icon={<Cpu className="h-3 w-3" />}
            label="CPU"
            value={`${metrics.cpu.toFixed(1)}%`}
            warning={metrics.cpu > HIGH_CPU_THRESHOLD}
          />

          <MetricItem
            icon={<MemoryStick className="h-3 w-3" />}
            label="Memory"
            value={`${metrics.memory.toFixed(1)}%`}
            warning={metrics.memory > HIGH_MEMORY_THRESHOLD}
          />

          <div className="h-6 w-px bg-border" />

          <MetricItem icon={<Users className="h-3 w-3" />} label="Agents" value={metrics.agents} />

          <MetricItem
            icon={<MessageSquare className="h-3 w-3" />}
            label="Messages"
            value={formatNumber(metrics.messages)}
          />

          <div className="h-6 w-px bg-border" />

          <MetricItem
            icon={<Clock className="h-3 w-3" />}
            label="Uptime"
            value={formatUptime(metrics.uptime)}
          />
        </div>

        <div className="text-xs text-muted-foreground">v{metrics.version}</div>
      </div>
    </footer>
  );
}
