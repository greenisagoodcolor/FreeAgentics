"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Agent } from "@/lib/types";
import type { AgentDetailed, AgentStatus } from "@/lib/types/agent-api";
import { Grid3x3, List, RefreshCw, Search } from "lucide-react";
import { useEffect, useState } from "react";
import AgentActivityTimeline from "./agent-activity-timeline";
import AgentCard from "./agentcard";
import AgentPerformanceChart from "./agent-performance-chart";
import AgentRelationshipNetwork from "./agent-relationship-network";

interface AgentDashboardProps {
  agents: Agent[];
  onSelectAgent: (agent: Agent) => void;
  selectedAgent: Agent | null;
  onRefresh?: () => void;
}

type ViewMode = "grid" | "list";
type FilterStatus = "all" | AgentStatus;

export default function AgentDashboard({
  agents,
  onSelectAgent,
  selectedAgent,
  onRefresh,
}: AgentDashboardProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");
  const [agentDetails, setAgentDetailed] = useState<
    Record<string, AgentDetailed>
  >({});

  // Mock data for agent details - in a real app, this would come from the API
  useEffect(() => {
    const mockDetails: Record<string, AgentDetailed> = {};
    agents.forEach((agent) => {
      mockDetails[agent.id] = {
        id: agent.id,
        name: agent.name,
        status: [
          "idle",
          "moving",
          "interacting",
          "planning",
          "executing",
          "learning",
        ][Math.floor(Math.random() * 6)] as AgentStatus,
        position: agent.position,
        personality: {
          openness: Math.random(),
          conscientiousness: Math.random(),
          extraversion: Math.random(),
          agreeableness: Math.random(),
          neuroticism: Math.random(),
        },
        capabilities: ["movement", "perception", "communication"],
        tags: ["explorer", "active"],
        metadata: {},
        resources: {
          energy: Math.floor(Math.random() * 100),
          health: Math.floor(Math.random() * 100),
          memory_used: Math.floor(Math.random() * 80),
          memory_capacity: 100,
        },
        goals: [
          {
            id: "goal-1",
            description: "Explore the environment",
            priority: 0.9,
            deadline: null,
          },
          {
            id: "goal-2",
            description: "Gather resources",
            priority: 0.6,
            deadline: null,
          },
        ],
        beliefs: [],
        relationships: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
    });
    setAgentDetailed(mockDetails);
  }, [agents]);

  // Filter agents based on search and status
  const filteredAgents = agents.filter((agent) => {
    const matchesSearch = agent.name
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const details = agentDetails[agent.id];
    const matchesStatus =
      filterStatus === "all" || details?.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="flex flex-col h-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-2xl font-bold">Agent Dashboard</CardTitle>
          <Button
            onClick={onRefresh}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-hidden">
        <Tabs defaultValue="overview" className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="relationships">Relationships</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="flex-1 overflow-hidden">
            <div className="space-y-4 h-full flex flex-col">
              {/* Controls */}
              <div className="flex items-center gap-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input
                    placeholder="Search agents..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>
                <Select
                  value={filterStatus}
                  onValueChange={(value) =>
                    setFilterStatus(value as FilterStatus)
                  }
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Filter by status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="idle">Idle</SelectItem>
                    <SelectItem value="moving">Moving</SelectItem>
                    <SelectItem value="interacting">Interacting</SelectItem>
                    <SelectItem value="planning">Planning</SelectItem>
                    <SelectItem value="executing">Executing</SelectItem>
                    <SelectItem value="learning">Learning</SelectItem>
                    <SelectItem value="error">Error</SelectItem>
                    <SelectItem value="offline">Offline</SelectItem>
                  </SelectContent>
                </Select>
                <div className="flex items-center gap-2">
                  <Button
                    variant={viewMode === "grid" ? "default" : "outline"}
                    size="icon"
                    onClick={() => setViewMode("grid")}
                  >
                    <Grid3x3 className="w-4 h-4" />
                  </Button>
                  <Button
                    variant={viewMode === "list" ? "default" : "outline"}
                    size="icon"
                    onClick={() => setViewMode("list")}
                  >
                    <List className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              {/* Agent Grid/List */}
              <ScrollArea className="flex-1">
                {viewMode === "grid" ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {filteredAgents.map((agent) => (
                      <AgentCard
                        key={agent.id}
                        agent={agent}
                        agentData={agentDetails[agent.id]}
                        isSelected={selectedAgent?.id === agent.id}
                        onClick={() => onSelectAgent(agent)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {filteredAgents.map((agent) => (
                      <AgentCard
                        key={agent.id}
                        agent={agent}
                        agentData={agentDetails[agent.id]}
                        isSelected={selectedAgent?.id === agent.id}
                        onClick={() => onSelectAgent(agent)}
                        className="w-full"
                      />
                    ))}
                  </div>
                )}
              </ScrollArea>

              {/* Summary Stats */}
              <div className="grid grid-cols-4 gap-4 pt-4 border-t">
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">
                    Total Agents
                  </div>
                  <div className="text-2xl font-bold">{agents.length}</div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Active</div>
                  <div className="text-2xl font-bold text-green-500">
                    {
                      Object.values(agentDetails).filter(
                        (d) => d.status !== "offline" && d.status !== "error",
                      ).length
                    }
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">
                    In Conversation
                  </div>
                  <div className="text-2xl font-bold text-blue-500">
                    {agents.filter((a) => a.inConversation).length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">
                    Autonomous
                  </div>
                  <div className="text-2xl font-bold text-purple-500">
                    {agents.filter((a) => a.autonomyEnabled).length}
                  </div>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="activity" className="flex-1 overflow-hidden">
            <AgentActivityTimeline
              agents={agents}
              agentDetails={agentDetails}
            />
          </TabsContent>

          <TabsContent value="performance" className="flex-1 overflow-hidden">
            <AgentPerformanceChart
              agents={agents}
              agentDetails={agentDetails}
            />
          </TabsContent>

          <TabsContent value="relationships" className="flex-1 overflow-hidden">
            <AgentRelationshipNetwork agents={agents} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </div>
  );
}
