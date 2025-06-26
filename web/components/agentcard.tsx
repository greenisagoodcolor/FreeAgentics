"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Agent } from "@/lib/types";
import type {
  AgentGoal,
  AgentResources,
  AgentStatus,
} from "@/lib/types/agent-api";
import {
  Activity,
  AlertCircle,
  Battery,
  Brain,
  CheckCircle,
  Clock,
  Heart,
  Power,
  PowerOff,
  Target,
  Users,
  Zap,
} from "lucide-react";
import type React from "react";

interface AgentCardProps {
  agent: Agent;
  agentData?: {
    status: AgentStatus;
    resources: AgentResources;
    goals: AgentGoal[];
    activity?: string;
  };
  isSelected?: boolean;
  onClick?: () => void;
  className?: string;
}

const statusColors: Record<AgentStatus, string> = {
  idle: "bg-gray-500",
  moving: "bg-blue-500",
  interacting: "bg-green-500",
  planning: "bg-yellow-500",
  executing: "bg-orange-500",
  learning: "bg-purple-500",
  error: "bg-red-500",
  offline: "bg-gray-700",
};

const statusIcons: Record<AgentStatus, React.ElementType> = {
  idle: Clock,
  moving: Activity,
  interacting: Users,
  planning: Brain,
  executing: Zap,
  learning: Brain,
  error: AlertCircle,
  offline: PowerOff,
};

export default function AgentCard({
  agent,
  agentData,
  isSelected = false,
  onClick,
  className = "",
}: AgentCardProps) {
  const status = agentData?.status || "offline";
  const resources = agentData?.resources || {
    energy: 0,
    health: 0,
    memory_used: 0,
    memory_capacity: 100,
  };
  const StatusIcon = statusIcons[status];

  const memoryUsagePercent =
    resources.memory_capacity > 0
      ? (resources.memory_used / resources.memory_capacity) * 100
      : 0;

  return (
    <TooltipProvider>
      <Card
        className={`cursor-pointer transition-all duration-200 hover:shadow-lg hover:scale-105 ${
          isSelected ? "ring-2 ring-primary" : ""
        } ${className}`}
        onClick={onClick}
      >
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: agent.color }}
              />
              <h3 className="font-semibold text-lg">{agent.name}</h3>
            </div>
            <div className="flex items-center gap-2">
              {/* Autonomy indicator */}
              <Tooltip>
                <TooltipTrigger>
                  {agent.autonomyEnabled ? (
                    <Power className="w-4 h-4 text-green-500" />
                  ) : (
                    <PowerOff className="w-4 h-4 text-gray-400" />
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    Autonomy: {agent.autonomyEnabled ? "Enabled" : "Disabled"}
                  </p>
                </TooltipContent>
              </Tooltip>

              {/* Status badge */}
              <Badge
                variant="secondary"
                className={`${statusColors[status]} text-white`}
              >
                <StatusIcon className="w-3 h-3 mr-1" />
                {status}
              </Badge>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* Position */}
          <div className="text-sm text-muted-foreground">
            Position: ({agent.position.x}, {agent.position.y})
          </div>

          {/* Resources */}
          <div className="space-y-2">
            {/* Energy */}
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger>
                  <Battery className="w-4 h-4 text-yellow-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Energy Level</p>
                </TooltipContent>
              </Tooltip>
              <Progress value={resources.energy} className="flex-1" />
              <span className="text-xs text-muted-foreground w-10 text-right">
                {resources.energy}%
              </span>
            </div>

            {/* Health */}
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger>
                  <Heart className="w-4 h-4 text-red-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Health Status</p>
                </TooltipContent>
              </Tooltip>
              <Progress value={resources.health} className="flex-1" />
              <span className="text-xs text-muted-foreground w-10 text-right">
                {resources.health}%
              </span>
            </div>

            {/* Memory */}
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger>
                  <Brain className="w-4 h-4 text-purple-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    Memory Usage: {resources.memory_used}MB /{" "}
                    {resources.memory_capacity}MB
                  </p>
                </TooltipContent>
              </Tooltip>
              <Progress value={memoryUsagePercent} className="flex-1" />
              <span className="text-xs text-muted-foreground w-10 text-right">
                {memoryUsagePercent.toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Current Activity */}
          {agentData?.activity && (
            <div className="text-sm">
              <span className="text-muted-foreground">Activity:</span>{" "}
              {agentData.activity}
            </div>
          )}

          {/* Goals */}
          {agentData?.goals && agentData.goals.length > 0 && (
            <div className="space-y-1">
              <div className="text-sm font-medium flex items-center gap-1">
                <Target className="w-4 h-4" />
                Active Goals:
              </div>
              <div className="space-y-1">
                {agentData.goals.slice(0, 2).map((goal) => (
                  <div
                    key={goal.id}
                    className="text-xs flex items-center gap-1"
                  >
                    {/* Mock status based on priority */}
                    {goal.priority > 0.7 ? (
                      <CheckCircle className="w-3 h-3 text-green-500" />
                    ) : (
                      <Clock className="w-3 h-3 text-yellow-500" />
                    )}
                    <span className="truncate">{goal.description}</span>
                  </div>
                ))}
                {agentData.goals.length > 2 && (
                  <div className="text-xs text-muted-foreground">
                    +{agentData.goals.length - 2} more goals
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Knowledge count */}
          <div className="text-sm text-muted-foreground">
            Knowledge entries: {agent.knowledge.length}
          </div>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}
