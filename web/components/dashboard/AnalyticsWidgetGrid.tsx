"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3, TrendingUp, Activity, Zap } from "lucide-react";

const metrics = [
  {
    title: "Active Agents",
    value: "4",
    change: "+2",
    icon: Activity,
    color: "text-green-600",
  },
  {
    title: "Conversations",
    value: "23",
    change: "+5",
    icon: BarChart3,
    color: "text-blue-600",
  },
  {
    title: "Performance",
    value: "94%",
    change: "+3%",
    icon: TrendingUp,
    color: "text-purple-600",
  },
  {
    title: "Inference Rate",
    value: "1.2k/s",
    change: "+12%",
    icon: Zap,
    color: "text-orange-600",
  },
];

export function AnalyticsWidgetGrid() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {metrics.map((metric) => (
        <Card key={metric.title}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{metric.title}</p>
                <p className="text-2xl font-bold">{metric.value}</p>
                <p className={`text-sm ${metric.color}`}>{metric.change}</p>
              </div>
              <metric.icon className={`h-8 w-8 ${metric.color}`} />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
