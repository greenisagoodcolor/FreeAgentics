"use client";

import { useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Users,
  Map,
  Brain,
  MessageSquare,
  Activity,
  TrendingUp,
  Sparkles,
  ArrowRight,
  Play,
  Settings,
} from "lucide-react";

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      {/* Hero Section */}
      <div className="container mx-auto px-6 py-12">
        <div className="text-center space-y-4 mb-12">
          <Badge variant="secondary" className="mb-4">
            <Sparkles className="h-3 w-3 mr-1" />
            Active Inference Platform
          </Badge>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            FreeAgentics
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Create, simulate, and deploy Active Inference agents in a rich
            hexagonal world
          </p>
          <div className="flex gap-4 justify-center pt-4">
            <Button size="lg" onClick={() => router.push("/agents")}>
              <Users className="h-5 w-5 mr-2" />
              Create Agent
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => router.push("/world")}
            >
              <Play className="h-5 w-5 mr-2" />
              Start Simulation
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid gap-4 md:grid-cols-4 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Active Agents
              </CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">24</div>
              <p className="text-xs text-muted-foreground">
                <TrendingUp className="h-3 w-3 inline mr-1 text-green-500" />
                12% from last hour
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                World Activity
              </CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">High</div>
              <p className="text-xs text-muted-foreground">
                342 actions/minute
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Knowledge Nodes
              </CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">2.8k</div>
              <p className="text-xs text-muted-foreground">
                156 patterns found
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Conversations
              </CardTitle>
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">12</div>
              <p className="text-xs text-muted-foreground">Active dialogues</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Features */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Card
            className="cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => router.push("/agents")}
          >
            <CardHeader>
              <Users className="h-8 w-8 mb-2 text-primary" />
              <CardTitle>Agent Creator</CardTitle>
              <CardDescription>
                Design agents with personality traits and backstories
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-between">
                Open Creator
                <ArrowRight className="h-4 w-4" />
              </Button>
            </CardContent>
          </Card>

          <Card
            className="cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => router.push("/world")}
          >
            <CardHeader>
              <Map className="h-8 w-8 mb-2 text-primary" />
              <CardTitle>World Simulation</CardTitle>
              <CardDescription>
                H3 hexagonal world with resources and terrain
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-between">
                View World
                <ArrowRight className="h-4 w-4" />
              </Button>
            </CardContent>
          </Card>

          <Card
            className="cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => router.push("/knowledge")}
          >
            <CardHeader>
              <Brain className="h-8 w-8 mb-2 text-primary" />
              <CardTitle>Knowledge Graphs</CardTitle>
              <CardDescription>
                Visualize agent knowledge and patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-between">
                Explore Graphs
                <ArrowRight className="h-4 w-4" />
              </Button>
            </CardContent>
          </Card>

          <Card
            className="cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => router.push("/conversations")}
          >
            <CardHeader>
              <MessageSquare className="h-8 w-8 mb-2 text-primary" />
              <CardTitle>Conversations</CardTitle>
              <CardDescription>
                Monitor real-time agent communications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-between">
                View Chats
                <ArrowRight className="h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Recent Activity */}
        <div className="grid gap-6 md:grid-cols-2 mt-8">
          <Card>
            <CardHeader>
              <CardTitle>Recent Events</CardTitle>
              <CardDescription>Latest simulation activity</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="h-2 w-2 bg-green-500 rounded-full mt-1.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Alliance Formed</p>
                    <p className="text-xs text-muted-foreground">
                      CuriousExplorer and WiseScholar agreed to share knowledge
                    </p>
                  </div>
                  <span className="text-xs text-muted-foreground">2m ago</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="h-2 w-2 bg-blue-500 rounded-full mt-1.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Resource Discovered</p>
                    <p className="text-xs text-muted-foreground">
                      BravePioneer found knowledge crystals in the eastern
                      mountains
                    </p>
                  </div>
                  <span className="text-xs text-muted-foreground">5m ago</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="h-2 w-2 bg-yellow-500 rounded-full mt-1.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Pattern Extracted</p>
                    <p className="text-xs text-muted-foreground">
                      TerrainMapper identified optimal paths through desert
                      regions
                    </p>
                  </div>
                  <span className="text-xs text-muted-foreground">8m ago</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>Common tasks and settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push("/agents")}
              >
                <Users className="h-4 w-4 mr-2" />
                Create New Agent
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push("/world")}
              >
                <Play className="h-4 w-4 mr-2" />
                Resume Simulation
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push("/knowledge")}
              >
                <Brain className="h-4 w-4 mr-2" />
                Analyze Knowledge Graphs
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push("/settings")}
              >
                <Settings className="h-4 w-4 mr-2" />
                Platform Settings
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
