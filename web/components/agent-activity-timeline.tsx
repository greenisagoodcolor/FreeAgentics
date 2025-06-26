"use client";

import { Badge } from "@/components/ui/badge";
import { CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Agent } from "@/lib/types";
import type { AgentDetailed } from "@/lib/types/agent-api";
import {
  Activity,
  Brain,
  Clock,
  MessageSquare,
  Navigation,
  Target,
  Zap,
} from "lucide-react";
import type React from "react";

interface AgentActivityTimelineProps {
  agents: Agent[];
  agentDetails: Record<string, AgentDetailed>;
}

interface IActivityEvent {
  id: string;
  agentId: string;
  agentName: string;
  type:
    | "status_change"
    | "interaction"
    | "goal_update"
    | "learning"
    | "movement"
    | "resource_change";
  description: string;
  timestamp: Date;
  icon: React.ElementType;
  color: string;
}

// Generate mock activity events
function generateMockActivities(
  agents: Agent[],
  agentDetails: Record<string, AgentDetailed>,
): IActivityEvent[] {
  const activities: IActivityEvent[] = [];
  const now = new Date();

  const eventTypes = [
    {
      type: "status_change",
      icon: Activity,
      color: "text-blue-500",
      template: "changed status to",
    },
    {
      type: "interaction",
      icon: MessageSquare,
      color: "text-green-500",
      template: "interacted with",
    },
    {
      type: "goal_update",
      icon: Target,
      color: "text-yellow-500",
      template: "completed goal:",
    },
    {
      type: "learning",
      icon: Brain,
      color: "text-purple-500",
      template: "learned new pattern:",
    },
    {
      type: "movement",
      icon: Navigation,
      color: "text-orange-500",
      template: "moved to position",
    },
    {
      type: "resource_change",
      icon: Zap,
      color: "text-red-500",
      template: "resource update:",
    },
  ] as const;

  // Return empty array if no agents
  if (agents.length === 0) {
    return activities;
  }

  // Generate 20 random activities
  for (let i = 0; i < 20; i++) {
    const agent = agents[Math.floor(Math.random() * agents.length)];
    if (!agent) continue; // Skip if agent is undefined

    const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
    const minutesAgo = Math.floor(Math.random() * 60);

    let description = "";
    switch (eventType.type) {
      case "status_change":
        const details = agentDetails[agent.id];
        description = `${eventType.template} ${details?.status || "idle"}`;
        break;
      case "interaction":
        const otherAgent = agents.find((a) => a.id !== agent.id);
        description = `${eventType.template} ${otherAgent?.name || "unknown"}`;
        break;
      case "goal_update":
        description = `${eventType.template} "Explore sector 7"`;
        break;
      case "learning":
        description = `${eventType.template} "Optimal pathfinding"`;
        break;
      case "movement":
        description = `${eventType.template} (${agent.position?.x || 0}, ${agent.position?.y || 0})`;
        break;
      case "resource_change":
        description = `${eventType.template} Energy +15%`;
        break;
    }

    activities.push({
      id: `activity-${i}`,
      agentId: agent.id,
      agentName: agent.name,
      type: eventType.type,
      description,
      timestamp: new Date(now.getTime() - minutesAgo * 60000),
      icon: eventType.icon,
      color: eventType.color,
    });
  }

  // Sort by timestamp descending
  return activities.sort(
    (a, b) => b.timestamp.getTime() - a.timestamp.getTime(),
  );
}

function formatTimeAgo(date: Date): string {
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function AgentActivityTimeline({
  agents,
  agentDetails,
}: AgentActivityTimelineProps) {
  const activities = generateMockActivities(agents, agentDetails);

  return (
    <div className="h-full flex flex-col">
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden">
        <ScrollArea className="h-full pr-4">
          <div className="space-y-4">
            {activities.map((activity) => {
              const Icon = activity.icon;
              return (
                <div key={activity.id} className="flex items-start gap-3">
                  <div className={`mt-1 ${activity.color}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{activity.agentName}</span>
                      <span className="text-sm text-muted-foreground">
                        {activity.description}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="w-3 h-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        {formatTimeAgo(activity.timestamp)}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {activity.type.replace("_", " ")}
                      </Badge>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </div>
  );
}
