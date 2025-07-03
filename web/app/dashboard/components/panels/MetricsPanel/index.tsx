"use client";

import React from "react";
import { useAppSelector } from "@/store/hooks";
import {
  Activity,
  Users,
  MessageSquare,
  Brain,
  TrendingUp,
  Zap,
} from "lucide-react";

export default function MetricsPanel() {
  // Redux state
  const agents = useAppSelector((state) => state.agents?.agents) || {};
  const conversations = useAppSelector((state) => state.conversations?.conversations) || {};
  const analytics = useAppSelector((state) => state.analytics) || { metrics: {} };

  // Calculate metrics
  const activeAgents = Object.values(agents).filter(
    (agent: any) => agent.status === "active"
  ).length;
  
  const totalMessages = Object.values(conversations).reduce(
    (sum: number, conv: any) => sum + (conv.messages?.length || 0),
    0
  );

  const metrics = [
    {
      label: "Active Agents",
      value: activeAgents.toString(),
      change: "+2",
      trend: "up",
      icon: Users,
      color: "text-blue-400",
    },
    {
      label: "Messages",
      value: totalMessages.toString(),
      change: "+156",
      trend: "up",
      icon: MessageSquare,
      color: "text-green-400",
    },
    {
      label: "Knowledge Nodes",
      value: "2,341",
      change: "+89",
      trend: "up",
      icon: Brain,
      color: "text-purple-400",
    },
    {
      label: "Processing Rate",
      value: "847/s",
      change: "+12%",
      trend: "up",
      icon: Activity,
      color: "text-yellow-400",
    },
    {
      label: "Free Energy",
      value: "23.4%",
      change: "-2.1%",
      trend: "down",
      icon: Zap,
      color: "text-orange-400",
    },
    {
      label: "System Health",
      value: "99.9%",
      change: "+0.1%",
      trend: "up",
      icon: TrendingUp,
      color: "text-emerald-400",
    },
  ];

  return (
    <div className="h-full flex items-center justify-around px-4">
      {metrics.map((metric) => {
        const Icon = metric.icon;
        return (
          <div key={metric.label} className="flex items-center gap-3">
            <Icon className={`w-5 h-5 ${metric.color}`} />
            <div>
              <div className="text-xs text-gray-400 uppercase">{metric.label}</div>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold">{metric.value}</span>
                <span className={`text-xs ${
                  metric.trend === "up" ? "text-green-400" : "text-red-400"
                }`}>
                  {metric.change}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}