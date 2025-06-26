"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Agent } from "@/lib/types";
import type { AgentDetailed } from "@/lib/types/agent-api";
import { useState } from "react";

interface AgentPerformanceChartProps {
  agents: Agent[];
  agentDetails: Record<string, AgentDetailed>;
}

type MetricType = "resources" | "goals" | "activity" | "efficiency";

export default function AgentPerformanceChart({
  agents,
  agentDetails,
}: AgentPerformanceChartProps) {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>("resources");
  const [selectedAgent, setSelectedAgent] = useState<string>("all");

  // Calculate metrics for visualization
  const getMetricData = () => {
    const filteredAgents =
      selectedAgent === "all"
        ? agents
        : agents.filter((a) => a.id === selectedAgent);

    switch (selectedMetric) {
      case "resources":
        return filteredAgents.map((agent) => {
          const details = agentDetails[agent.id];
          return {
            name: agent.name,
            energy: details?.resources.energy || 0,
            health: details?.resources.health || 0,
            memory: details?.resources.memory_used || 0,
          };
        });

      case "goals":
        return filteredAgents.map((agent) => {
          const details = agentDetails[agent.id];
          const goals = details?.goals || [];
          // Mock data for goal status - in real app this would be tracked
          const completed = Math.floor(goals.length * 0.6);
          const active = goals.length - completed;
          return {
            name: agent.name,
            completed,
            active,
            total: goals.length,
          };
        });

      case "activity":
        return filteredAgents.map((agent) => {
          const details = agentDetails[agent.id];
          // Mock activity scores
          return {
            name: agent.name,
            interactions: Math.floor(Math.random() * 20),
            movements: Math.floor(Math.random() * 50),
            learningEvents: Math.floor(Math.random() * 10),
          };
        });

      case "efficiency":
        return filteredAgents.map((agent) => {
          const details = agentDetails[agent.id];
          const energyEfficiency = details ? 100 - details.resources.energy : 0;
          const memoryEfficiency = details
            ? (details.resources.memory_used /
                details.resources.memory_capacity) *
              100
            : 0;
          return {
            name: agent.name,
            energyEfficiency,
            memoryEfficiency,
            overallScore: (energyEfficiency + memoryEfficiency) / 2,
          };
        });
    }
  };

  const data = getMetricData();

  // Simple bar chart visualization
  const renderBars = (value: number, maxValue: number, color: string) => {
    const percentage = (value / maxValue) * 100;
    return (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Performance Metrics</CardTitle>
          <div className="flex gap-2">
            <Select
              value={selectedMetric}
              onValueChange={(value) => setSelectedMetric(value as MetricType)}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="resources">Resources</SelectItem>
                <SelectItem value="goals">Goals</SelectItem>
                <SelectItem value="activity">Activity</SelectItem>
                <SelectItem value="efficiency">Efficiency</SelectItem>
              </SelectContent>
            </Select>
            <Select value={selectedAgent} onValueChange={setSelectedAgent}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select agent" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Agents</SelectItem>
                {agents.map((agent) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto">
        <div className="space-y-4">
          {selectedMetric === "resources" && (
            <div className="space-y-4">
              {(data as any[]).map((item, index) => (
                <Card key={index} className="p-4">
                  <h4 className="font-medium mb-3">{item.name}</h4>
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Energy</span>
                        <span>{item.energy}%</span>
                      </div>
                      {renderBars(item.energy, 100, "bg-yellow-500")}
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Health</span>
                        <span>{item.health}%</span>
                      </div>
                      {renderBars(item.health, 100, "bg-red-500")}
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Memory</span>
                        <span>{item.memory}MB</span>
                      </div>
                      {renderBars(item.memory, 100, "bg-purple-500")}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {selectedMetric === "goals" && (
            <div className="space-y-4">
              {(data as any[]).map((item, index) => (
                <Card key={index} className="p-4">
                  <h4 className="font-medium mb-3">{item.name}</h4>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-green-500">
                        {item.completed}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Completed
                      </div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-blue-500">
                        {item.active}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Active
                      </div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold">{item.total}</div>
                      <div className="text-sm text-muted-foreground">Total</div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {selectedMetric === "activity" && (
            <div className="space-y-4">
              {(data as any[]).map((item, index) => (
                <Card key={index} className="p-4">
                  <h4 className="font-medium mb-3">{item.name}</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Interactions</span>
                      <span className="font-medium">{item.interactions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Movements</span>
                      <span className="font-medium">{item.movements}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Learning Events</span>
                      <span className="font-medium">{item.learningEvents}</span>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {selectedMetric === "efficiency" && (
            <div className="space-y-4">
              {(data as any[]).map((item, index) => (
                <Card key={index} className="p-4">
                  <h4 className="font-medium mb-3">{item.name}</h4>
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Energy Efficiency</span>
                        <span>{item.energyEfficiency.toFixed(1)}%</span>
                      </div>
                      {renderBars(item.energyEfficiency, 100, "bg-green-500")}
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Memory Efficiency</span>
                        <span>{item.memoryEfficiency.toFixed(1)}%</span>
                      </div>
                      {renderBars(item.memoryEfficiency, 100, "bg-blue-500")}
                    </div>
                    <div className="mt-3 pt-3 border-t">
                      <div className="flex justify-between">
                        <span className="font-medium">Overall Score</span>
                        <span className="font-bold text-lg">
                          {item.overallScore.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </div>
  );
}
