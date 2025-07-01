"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { useAppSelector } from "@/store/hooks";
import { DashboardView } from "../../../../page";
import {
  Activity,
  Users,
  MessageSquare,
  Brain,
  TrendingUp,
  Zap,
  Presentation,
} from "lucide-react";

interface MetricsPanelProps {
  view: DashboardView;
}

export default function MetricsPanel({ view }: MetricsPanelProps) {
  const router = useRouter();

  // Redux state
  const agents = useAppSelector((state) => state.agents?.agents) || {};
  const conversations =
    useAppSelector((state) => state.conversations?.conversations) || {};
  const analytics = useAppSelector((state) => state.analytics) || {
    metrics: {},
  };

  // Calculate metrics
  const totalAgents = Object.keys(agents).length;
  const activeAgents = Object.values(agents).filter(
    (a: any) => a.status === "active",
  ).length;
  const totalMessages = Object.values(conversations).reduce(
    (total: number, conv: any) => {
      return total + (conv.messages?.length || 0);
    },
    0,
  );
  const avgResponseTime = (analytics as any).metrics?.averageResponseTime || 0;

  const metrics = [
    {
      label: "TOTAL AGENTS",
      value: totalAgents.toString(),
      change: "+2",
      trend: "up",
      icon: Users,
      color: "text-blue-400",
    },
    {
      label: "ACTIVE AGENTS",
      value: activeAgents.toString(),
      change: `${((activeAgents / totalAgents) * 100 || 0).toFixed(0)}%`,
      trend: "up",
      icon: Activity,
      color: "text-green-400",
    },
    {
      label: "TOTAL MESSAGES",
      value: totalMessages.toLocaleString(),
      change: "+156",
      trend: "up",
      icon: MessageSquare,
      color: "text-purple-400",
    },
    {
      label: "AVG RESPONSE",
      value: `${avgResponseTime.toFixed(0)}ms`,
      change: "-12ms",
      trend: "down",
      icon: Zap,
      color: "text-yellow-400",
    },
    {
      label: "KNOWLEDGE NODES",
      value: "1,247",
      change: "+89",
      trend: "up",
      icon: Brain,
      color: "text-indigo-400",
    },
    {
      label: "SYSTEM HEALTH",
      value: "99.9%",
      change: "+0.1%",
      trend: "up",
      icon: TrendingUp,
      color: "text-emerald-400",
    },
  ];

  return (
    <div className="h-full bg-[var(--bg-primary)] p-4">
      <div className="grid grid-cols-7 gap-4 h-full">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div
              key={metric.label}
              className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded-lg p-4 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-2">
                <Icon className={`w-5 h-5 ${metric.color}`} />
                <span
                  className={`text-xs font-mono ${
                    metric.trend === "up" ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {metric.change}
                </span>
              </div>

              <div>
                <div className="text-2xl font-bold font-mono text-[var(--text-primary)] mb-1">
                  {metric.value}
                </div>
                <div className="text-xs text-[var(--text-secondary)] font-mono">
                  {metric.label}
                </div>
              </div>
            </div>
          );
        })}

        {/* CEO Demo Button */}
        <div
          className="bg-[var(--bg-secondary)] border border-[var(--primary-amber)] rounded-lg p-4 flex flex-col justify-center items-center cursor-pointer hover:bg-[var(--bg-tertiary)] transition-colors"
          onClick={() => router.push("/dashboard?view=ceo-demo")}
        >
          <Presentation className="w-8 h-8 text-[var(--primary-amber)] mb-2" />
          <div className="text-sm font-bold text-[var(--primary-amber)] font-mono">
            CEO DEMO
          </div>
          <div className="text-xs text-[var(--text-secondary)] font-mono mt-1">
            Launch Demo
          </div>
        </div>
      </div>
    </div>
  );
}
