"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Brain,
  Users,
  Map,
  MessageSquare,
  Network,
  Target,
  Plus,
  Play,
  Pause,
  RefreshCw,
  Maximize2,
  Activity,
} from "lucide-react";
import GridWorld from "@/components/gridworld";
import ChatWindow from "@/components/chat-window";
import AgentList from "@/components/AgentList";
import GlobalKnowledgeGraph from "@/components/GlobalKnowledgeGraph";
import AgentDashboard from "@/components/agentdashboard";
import AutonomousConversationManager from "@/components/autonomous-conversation-manager";
import AgentActivityTimeline from "@/components/agent-activity-timeline";
import { ReadinessPanel } from "@/components/readiness-panel";
import type { Agent, Conversation, Message, Position } from "@/lib/types";
import type { AgentDetailed } from "@/lib/types/agent-api";
import { nanoid } from "nanoid";
import { Separator } from "@/components/ui/separator";
// import {
//   ResizableHandle,
//   ResizablePanel,
//   ResizablePanelGroup,
// } from "@/components/ui/resizable";

export default function DashboardPage() {
  // State management
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [agentDetails, setAgentDetails] = useState<Record<string, AgentDetailed>>({});
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [selectedTab, setSelectedTab] = useState("overview");

  // Initialize with sample agents
  useEffect(() => {
    const sampleAgents: Agent[] = [
      {
        id: "explorer-1",
        name: "Scout Alpha",
        biography: "An explorer agent focused on discovery and reconnaissance",
        position: { x: 3, y: 3 },
        color: "#22c55e",
        knowledge: [],
        inConversation: false,
        autonomyEnabled: true,
      },
      {
        id: "merchant-1",
        name: "Trader Beta",
        biography: "A merchant agent specialized in resource trading and negotiation",
        position: { x: 7, y: 5 },
        color: "#3b82f6",
        knowledge: [],
        inConversation: false,
        autonomyEnabled: true,
      },
      {
        id: "scholar-1",
        name: "Sage Gamma",
        position: { x: 5, y: 8 },
        color: "#a855f7",
        class: "Scholar",
        knowledge: [],
        inConversation: false,
        autonomyEnabled: false,
      },
      {
        id: "guardian-1",
        name: "Sentinel Delta",
        position: { x: 2, y: 6 },
        color: "#ef4444",
        class: "Guardian",
        knowledge: [],
        inConversation: false,
        autonomyEnabled: false,
      },
    ];

    setAgents(sampleAgents);

    // Generate mock detailed data
    const details: Record<string, AgentDetailed> = {};
    sampleAgents.forEach((agent) => {
      details[agent.id] = {
        ...agent,
        status: "idle",
        personality: {
          openness: Math.random(),
          conscientiousness: Math.random(),
          extraversion: Math.random(),
          agreeableness: Math.random(),
          neuroticism: Math.random(),
        },
        capabilities: ["movement", "perception", "communication"],
        tags: [agent.class?.toLowerCase() || "basic", "active"],
        metadata: {},
        resources: {
          energy: Math.floor(Math.random() * 50 + 50),
          health: Math.floor(Math.random() * 50 + 50),
          memory_used: Math.floor(Math.random() * 40),
          memory_capacity: 100,
        },
        goals: [
          {
            id: `goal-${agent.id}-1`,
            description: `${agent.class} primary objective`,
            priority: 0.9,
            deadline: null,
          },
        ],
        beliefs: [],
        relationships: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
    });
    setAgentDetails(details);
  }, []);

  // Agent management functions
  const handleCreateAgent = () => {
    const newAgent: Agent = {
      id: nanoid(),
      name: `Agent ${agents.length + 1}`,
      position: { x: Math.floor(Math.random() * 10), y: Math.floor(Math.random() * 10) },
      color: ["#22c55e", "#3b82f6", "#a855f7", "#ef4444"][Math.floor(Math.random() * 4)],
      class: ["Explorer", "Merchant", "Scholar", "Guardian"][Math.floor(Math.random() * 4)],
      knowledge: [],
      inConversation: false,
      autonomyEnabled: false,
    };
    setAgents([...agents, newAgent]);
  };

  const handleDeleteAgent = (agentId: string) => {
    setAgents(agents.filter((a) => a.id !== agentId));
    if (selectedAgent?.id === agentId) {
      setSelectedAgent(null);
    }
  };

  const handleUpdatePosition = (agentId: string, position: Position) => {
    setAgents(
      agents.map((a) => (a.id === agentId ? { ...a, position } : a))
    );
  };

  const handleAddToConversation = (agentId: string) => {
    setAgents(
      agents.map((a) =>
        a.id === agentId ? { ...a, inConversation: true } : a
      )
    );
  };

  const handleRemoveFromConversation = (agentId: string) => {
    setAgents(
      agents.map((a) =>
        a.id === agentId ? { ...a, inConversation: false } : a
      )
    );
  };

  const handleToggleAutonomy = (agentId: string, enabled: boolean) => {
    setAgents(
      agents.map((a) =>
        a.id === agentId ? { ...a, autonomyEnabled: enabled } : a
      )
    );
  };

  const handleSendMessage = (content: string, senderId: string) => {
    if (!activeConversation) return;

    const newMessage: Message = {
      id: nanoid(),
      senderId,
      content,
      timestamp: new Date(),
    };

    setActiveConversation({
      ...activeConversation,
      messages: [...activeConversation.messages, newMessage],
    });
  };

  const handleStartConversation = () => {
    const participatingAgents = agents.filter((a) => a.inConversation);
    if (participatingAgents.length === 0) return;

    const newConversation: Conversation = {
      id: nanoid(),
      participants: participatingAgents.map((a) => a.id),
      messages: [],
      startTime: new Date(),
      endTime: null,
    };

    setActiveConversation(newConversation);
  };

  const handleEndConversation = () => {
    if (activeConversation) {
      setActiveConversation({
        ...activeConversation,
        endTime: new Date(),
      });
    }
    setActiveConversation(null);
    setAgents(agents.map((a) => ({ ...a, inConversation: false })));
  };

  const toggleSimulation = () => {
    setIsSimulationRunning(!isSimulationRunning);
  };

  // Statistics
  const stats = {
    totalAgents: agents.length,
    activeAgents: agents.filter((a) => a.autonomyEnabled).length,
    inConversation: agents.filter((a) => a.inConversation).length,
    messages: activeConversation?.messages.length || 0,
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-[1920px] mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Brain className="h-8 w-8 text-primary" />
              CogniticNet Control Center
            </h1>
            <p className="text-muted-foreground">
              Real-time multi-agent simulation dashboard
            </p>
          </div>
          <div className="flex items-center gap-4">
            <Button
              onClick={toggleSimulation}
              variant={isSimulationRunning ? "destructive" : "default"}
              className="gap-2"
            >
              {isSimulationRunning ? (
                <>
                  <Pause className="h-4 w-4" />
                  Pause Simulation
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Start Simulation
                </>
              )}
            </Button>
            <Button variant="outline" className="gap-2">
              <RefreshCw className="h-4 w-4" />
              Reset
            </Button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total Agents</p>
                  <p className="text-2xl font-bold">{stats.totalAgents}</p>
                </div>
                <Users className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Active</p>
                  <p className="text-2xl font-bold text-green-500">
                    {stats.activeAgents}
                  </p>
                </div>
                <Activity className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">In Conversation</p>
                  <p className="text-2xl font-bold text-blue-500">
                    {stats.inConversation}
                  </p>
                </div>
                <MessageSquare className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Messages</p>
                  <p className="text-2xl font-bold text-purple-500">
                    {stats.messages}
                  </p>
                </div>
                <MessageSquare className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard */}
        <div className="grid grid-cols-12 gap-4 min-h-[800px]">
          {/* Left Panel - Agent Management */}
          <div className="col-span-3 rounded-lg border bg-card">
            <div className="h-full p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Agents</h2>
                <Button size="sm" onClick={handleCreateAgent} className="gap-1">
                  <Plus className="h-3 w-3" />
                  New
                </Button>
              </div>
              <AgentList
                agents={agents}
                selectedAgent={selectedAgent}
                onSelectAgent={setSelectedAgent}
                onCreateAgent={handleCreateAgent}
                onCreateAgentWithName={(name) => {
                  const newAgent: Agent = {
                    id: nanoid(),
                    name,
                    position: { x: 5, y: 5 },
                    color: "#22c55e",
                    class: "Explorer",
                    knowledge: [],
                    inConversation: false,
                    autonomyEnabled: false,
                  };
                  setAgents([...agents, newAgent]);
                }}
                onDeleteAgent={handleDeleteAgent}
                onAddToConversation={handleAddToConversation}
                onRemoveFromConversation={handleRemoveFromConversation}
                onUpdateAgentColor={(id, color) => {
                  setAgents(agents.map((a) => (a.id === id ? { ...a, color } : a)));
                }}
                onToggleAutonomy={handleToggleAutonomy}
                onExportAgents={() => {}}
                onImportAgents={() => {}}
                activeConversation={!!activeConversation}
              />
            </div>
          </div>

          {/* Center Panel - Main View */}
          <div className="col-span-6 rounded-lg border bg-card">
            <Tabs value={selectedTab} onValueChange={setSelectedTab} className="h-full">
              <div className="p-4 pb-0">
                <TabsList className="grid w-full grid-cols-5">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="world">World Map</TabsTrigger>
                  <TabsTrigger value="conversation">Conversation</TabsTrigger>
                  <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
                  <TabsTrigger value="goals">Goals & Tasks</TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="overview" className="h-full p-4">
                <AgentDashboard
                  agents={agents}
                  onSelectAgent={setSelectedAgent}
                  selectedAgent={selectedAgent}
                  onRefresh={() => {}}
                />
              </TabsContent>

              <TabsContent value="world" className="h-full p-4">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>World Simulation</CardTitle>
                  </CardHeader>
                  <CardContent className="h-[calc(100%-5rem)]">
                    <GridWorld
                      agents={agents}
                      onUpdatePosition={handleUpdatePosition}
                      activeConversation={activeConversation}
                      onSelectKnowledgeNode={() => {}}
                      onShowAbout={() => {}}
                    />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="conversation" className="h-full p-4">
                <div className="h-full space-y-4">
                  {!activeConversation && (
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-muted-foreground">
                              {agents.filter((a) => a.inConversation).length} agents selected
                            </p>
                          </div>
                          <Button
                            onClick={handleStartConversation}
                            disabled={agents.filter((a) => a.inConversation).length === 0}
                          >
                            Start Conversation
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                  {activeConversation && (
                    <ChatWindow
                      conversation={activeConversation}
                      agents={agents.filter((a) => a.inConversation)}
                      onSendMessage={handleSendMessage}
                      onEndConversation={handleEndConversation}
                    />
                  )}
                  <AutonomousConversationManager
                    conversation={activeConversation}
                    agents={agents}
                    onSendMessage={handleSendMessage}
                  />
                </div>
              </TabsContent>

              <TabsContent value="knowledge" className="h-full p-4">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>Global Knowledge Graph</CardTitle>
                  </CardHeader>
                  <CardContent className="h-[calc(100%-5rem)]">
                    <GlobalKnowledgeGraph
                      agents={agents}
                      onSelectNode={() => {}}
                      onShowAbout={() => {}}
                    />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="goals" className="h-full p-4">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>Goals & Tasks</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {selectedAgent && agentDetails[selectedAgent.id]?.goals.map((goal) => (
                        <Card key={goal.id}>
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-medium">{goal.description}</p>
                                <p className="text-sm text-muted-foreground">
                                  Priority: {(goal.priority * 100).toFixed(0)}%
                                </p>
                              </div>
                              <Target className="h-5 w-5 text-muted-foreground" />
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                      {(!selectedAgent || agentDetails[selectedAgent.id]?.goals.length === 0) && (
                        <div className="text-center py-8 text-muted-foreground">
                          {selectedAgent
                            ? "No goals set for this agent"
                            : "Select an agent to view goals"}
                        </div>
                      )}
                      <Button className="w-full" variant="outline">
                        <Plus className="h-4 w-4 mr-2" />
                        Add New Goal
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Right Panel - Activity & Status */}
          <div className="col-span-3 rounded-lg border bg-card">
            <div className="h-full p-4 space-y-4 overflow-y-auto">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">System Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <ReadinessPanel agentId={selectedAgent?.id || agents[0]?.id || ""} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Activity Timeline</CardTitle>
                </CardHeader>
                <CardContent>
                  <AgentActivityTimeline
                    agents={agents}
                    agentDetails={agentDetails}
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
