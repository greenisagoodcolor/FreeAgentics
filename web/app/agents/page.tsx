"use client";

import { useState } from "react";
import { CharacterCreator } from "@/components/character-creator";
import { BackendAgentList } from "@/components/backend-agent-list";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { PlusCircle, Users, Brain, Activity } from "lucide-react";
import type { Agent } from "@/lib/types";

export default function AgentsPage() {
  const [showCreator, setShowCreator] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Agent Management</h1>
          <p className="text-muted-foreground mt-1">
            Create, manage, and monitor your Active Inference agents
          </p>
        </div>
        <Button
          onClick={() => setShowCreator(true)}
          size="lg"
          className="gap-2"
        >
          <PlusCircle className="h-5 w-5" />
          Create Agent
        </Button>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">24</div>
            <p className="text-xs text-muted-foreground">+3 from last week</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Now</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">18</div>
            <p className="text-xs text-muted-foreground">75% activity rate</p>
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
            <div className="text-2xl font-bold">1,284</div>
            <p className="text-xs text-muted-foreground">+198 today</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Avg. Performance
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87%</div>
            <p className="text-xs text-muted-foreground">+5% improvement</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="all" className="space-y-4">
        <TabsList>
          <TabsTrigger value="all">All Agents</TabsTrigger>
          <TabsTrigger value="explorer">Explorers</TabsTrigger>
          <TabsTrigger value="merchant">Merchants</TabsTrigger>
          <TabsTrigger value="scholar">Scholars</TabsTrigger>
          <TabsTrigger value="guardian">Guardians</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4">
          <BackendAgentList
            onSelectAgent={setSelectedAgent}
            selectedAgent={selectedAgent}
          />
        </TabsContent>

        <TabsContent value="explorer" className="space-y-4">
          <BackendAgentList
            filter={{ class: "Explorer" }}
            onSelectAgent={setSelectedAgent}
            selectedAgent={selectedAgent}
          />
        </TabsContent>

        <TabsContent value="merchant" className="space-y-4">
          <BackendAgentList
            filter={{ class: "Merchant" }}
            onSelectAgent={setSelectedAgent}
            selectedAgent={selectedAgent}
          />
        </TabsContent>

        <TabsContent value="scholar" className="space-y-4">
          <BackendAgentList
            filter={{ class: "Scholar" }}
            onSelectAgent={setSelectedAgent}
            selectedAgent={selectedAgent}
          />
        </TabsContent>

        <TabsContent value="guardian" className="space-y-4">
          <BackendAgentList
            filter={{ class: "Guardian" }}
            onSelectAgent={setSelectedAgent}
            selectedAgent={selectedAgent}
          />
        </TabsContent>
      </Tabs>

      {/* Character Creator Modal */}
      {showCreator && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-background rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Create New Agent</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowCreator(false)}
                >
                  ✕
                </Button>
              </div>
              <CharacterCreator
                onClose={() => setShowCreator(false)}
                onSuccess={() => {
                  setShowCreator(false);
                  // Refresh agent list
                  window.location.reload();
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
