"use client";

import { useState } from "react";
import { ConversationView } from "@/components/conversation-view";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  MessageSquare,
  Users,
  TrendingUp,
  Clock,
  Filter,
  Play,
  Pause,
} from "lucide-react";

export default function ConversationsPage() {
  const [selectedConversation, setSelectedConversation] = useState<
    string | null
  >(null);
  const [isLive, setIsLive] = useState(true);
  const [filter, setFilter] = useState<"all" | "active" | "completed">("all");

  // Mock conversation data
  const conversations = [
    {
      id: "conv_123",
      participants: ["CuriousExplorer", "WiseScholar"],
      status: "active",
      startTime: "2 minutes ago",
      messageCount: 8,
      intent: "knowledge_sharing",
    },
    {
      id: "conv_456",
      participants: ["BravePioneer", "CautiousGuardian", "CleverMerchant"],
      status: "active",
      startTime: "5 minutes ago",
      messageCount: 15,
      intent: "alliance_formation",
    },
    {
      id: "conv_789",
      participants: ["ResourceHunter", "TerrainMapper"],
      status: "completed",
      startTime: "10 minutes ago",
      messageCount: 23,
      intent: "trade_negotiation",
    },
  ];

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Conversation Monitor</h1>
          <p className="text-muted-foreground mt-1">
            Real-time agent communications and emergent dialogue patterns
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={isLive ? "default" : "outline"}
            size="sm"
            onClick={() => setIsLive(!isLive)}
          >
            {isLive ? (
              <>
                <Pause className="h-4 w-4 mr-2" />
                Pause
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Resume
              </>
            )}
          </Button>
          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
        </div>
      </div>

      {/* Conversation Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Active Conversations
            </CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">3 multi-agent</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Messages/Hour</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">342</div>
            <p className="text-xs text-muted-foreground">+15% from average</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Unique Participants
            </CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">18</div>
            <p className="text-xs text-muted-foreground">
              75% of active agents
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Duration</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">4.5m</div>
            <p className="text-xs text-muted-foreground">12 messages avg</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Conversation List - Takes up 1 column */}
        <div className="lg:col-span-1">
          <Card className="h-[700px]">
            <CardHeader>
              <CardTitle>Conversations</CardTitle>
              <Tabs value={filter} onValueChange={(v) => setFilter(v as any)}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="all">All</TabsTrigger>
                  <TabsTrigger value="active">Active</TabsTrigger>
                  <TabsTrigger value="completed">Completed</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[550px]">
                <div className="space-y-2">
                  {conversations
                    .filter((c) => filter === "all" || c.status === filter)
                    .map((conv) => (
                      <Card
                        key={conv.id}
                        className={`cursor-pointer transition-colors ${
                          selectedConversation === conv.id
                            ? "border-primary"
                            : ""
                        }`}
                        onClick={() => setSelectedConversation(conv.id)}
                      >
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start mb-2">
                            <div className="text-sm font-medium">
                              {conv.participants.join(" ↔ ")}
                            </div>
                            <Badge
                              variant={
                                conv.status === "active"
                                  ? "default"
                                  : "secondary"
                              }
                            >
                              {conv.status}
                            </Badge>
                          </div>
                          <div className="flex justify-between items-center text-xs text-muted-foreground">
                            <span>{conv.startTime}</span>
                            <span>{conv.messageCount} messages</span>
                          </div>
                          <Badge variant="outline" className="mt-2 text-xs">
                            {conv.intent.replace("_", " ")}
                          </Badge>
                        </CardContent>
                      </Card>
                    ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Conversation View - Takes up 2 columns */}
        <div className="lg:col-span-2">
          <Card className="h-[700px]">
            <CardHeader>
              <CardTitle>Conversation Details</CardTitle>
              {selectedConversation && (
                <CardDescription>Live transcript and analysis</CardDescription>
              )}
            </CardHeader>
            <CardContent>
              {selectedConversation ? (
                <ConversationView
                  conversationId={selectedConversation}
                  isLive={isLive}
                />
              ) : (
                <div className="flex items-center justify-center h-[600px] text-muted-foreground">
                  Select a conversation to view details
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Intent Analysis */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Intent Distribution</CardTitle>
            <CardDescription>
              Communication purposes in the last hour
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {[
                { intent: "Knowledge Sharing", count: 45, percentage: 35 },
                { intent: "Trade Negotiation", count: 32, percentage: 25 },
                { intent: "Alliance Formation", count: 26, percentage: 20 },
                { intent: "Casual Greeting", count: 19, percentage: 15 },
                { intent: "Warning/Alert", count: 6, percentage: 5 },
              ].map((item) => (
                <div key={item.intent} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>{item.intent}</span>
                    <span className="text-muted-foreground">{item.count}</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div
                      className="bg-primary rounded-full h-2"
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Emergent Patterns</CardTitle>
            <CardDescription>Notable communication behaviors</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="p-3 bg-secondary rounded-lg">
                <div className="font-medium text-sm mb-1">
                  Trust Networks Forming
                </div>
                <p className="text-xs text-muted-foreground">
                  Explorers and Scholars showing increased cooperation frequency
                </p>
              </div>
              <div className="p-3 bg-secondary rounded-lg">
                <div className="font-medium text-sm mb-1">
                  Resource Information Flow
                </div>
                <p className="text-xs text-muted-foreground">
                  Knowledge about eastern forest resources spreading rapidly
                </p>
              </div>
              <div className="p-3 bg-secondary rounded-lg">
                <div className="font-medium text-sm mb-1">
                  Multi-Agent Coordination
                </div>
                <p className="text-xs text-muted-foreground">
                  Groups of 3+ agents coordinating exploration efforts
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
