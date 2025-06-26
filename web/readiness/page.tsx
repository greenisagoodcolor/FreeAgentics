"use client";

import { useState } from "react";
import { ReadinessPanel } from "@/components/readiness-panel";
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
import { Brain, Cpu, Rocket, Users } from "lucide-react";

const DEMO_AGENTS = [
  {
    id: "agent_123",
    name: "Explorer Alpha",
    class: "Explorer",
    status: "ready",
    description: "Experienced explorer agent with high readiness scores",
    icon: Rocket,
  },
  {
    id: "agent_456",
    name: "Scholar Beta",
    class: "Scholar",
    status: "training",
    description: "Scholar agent still in training phase",
    icon: Brain,
  },
];

export default function ReadinessPage() {
  const [selectedAgent, setSelectedAgent] = useState(DEMO_AGENTS[0].id);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Agent Readiness Evaluation</h1>
        <p className="text-muted-foreground">
          Evaluate agent readiness for autonomous hardware deployment and export
          deployment packages.
        </p>
      </div>

      {/* Agent Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Agent</CardTitle>
          <CardDescription>
            Choose an agent to view their readiness evaluation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {DEMO_AGENTS.map((agent) => {
              const Icon = agent.icon;
              return (
                <Card
                  key={agent.id}
                  className={`cursor-pointer transition-colors ${
                    selectedAgent === agent.id
                      ? "border-primary bg-primary/5"
                      : "hover:border-primary/50"
                  }`}
                  onClick={() => setSelectedAgent(agent.id)}
                >
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <div className="rounded-lg bg-primary/10 p-2">
                          <Icon className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h3 className="font-semibold">{agent.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            {agent.description}
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <Badge variant="secondary">{agent.class}</Badge>
                            <Badge
                              variant={
                                agent.status === "ready" ? "default" : "outline"
                              }
                            >
                              {agent.status}
                            </Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Readiness Panel */}
      <ReadinessPanel agentId={selectedAgent} />

      {/* Information Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Evaluation Process
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Agents are evaluated across 5 key dimensions: knowledge maturity,
              goal achievement, model stability, collaboration, and resource
              management. An overall score of 85% or higher indicates readiness.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Hardware Targets
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Export packages can be customized for different hardware platforms
              including Raspberry Pi, Mac Mini, and NVIDIA Jetson. Each package
              includes optimized configurations and deployment scripts.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Continuous Improvement
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Agents that aren't ready receive specific recommendations to
              improve their scores. Regular evaluation helps track progress and
              ensures only fully prepared agents are deployed to hardware.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
