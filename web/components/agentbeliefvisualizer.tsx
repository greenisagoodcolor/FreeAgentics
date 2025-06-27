"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { Agent } from "@/lib/types";
import type { AgentBelief, Memory } from "@/lib/types/agent-api";
import { AlertCircle, Brain, Eye, Lightbulb, Target } from "lucide-react";
import type React from "react";

interface AgentBeliefVisualizerProps {
  agent: Agent;
  beliefs?: AgentBelief[];
  memory?: Memory;
}

// Mock belief data generator
function generateMockBeliefs(): AgentBelief[] {
  return [
    {
      id: "belief-1",
      content: "There are resources in the northern sector",
      confidence: 0.85,
    },
    {
      id: "belief-2",
      content: "Agent Alpha is cooperative and trustworthy",
      confidence: 0.92,
    },
    {
      id: "belief-3",
      content: "The optimal path to the goal is through the center",
      confidence: 0.67,
    },
    {
      id: "belief-4",
      content: "Energy conservation is critical for long-term survival",
      confidence: 0.95,
    },
  ];
}

const beliefTypeIcons: Record<string, React.ElementType> = {
  environmental: Eye,
  social: Target,
  strategic: Lightbulb,
  policy: Brain,
};

const beliefTypeColors: Record<string, string> = {
  environmental: "bg-green-500",
  social: "bg-blue-500",
  strategic: "bg-yellow-500",
  policy: "bg-purple-500",
};

export default function AgentBeliefVisualizer({
  agent,
  beliefs = generateMockBeliefs(),
  memory,
}: AgentBeliefVisualizerProps) {
  // Group beliefs by confidence level
  const beliefsByConfidence = beliefs.reduce(
    (acc, belief) => {
      const level =
        belief.confidence >= 0.8
          ? "high"
          : belief.confidence >= 0.5
            ? "medium"
            : "low";
      if (!acc[level]) acc[level] = [];
      acc[level].push(belief);
      return acc;
    },
    {} as Record<string, AgentBelief[]>,
  );

  // Calculate average confidence by level
  const avgConfidenceByLevel = Object.entries(beliefsByConfidence).reduce(
    (acc, [level, beliefs]) => {
      const avg =
        beliefs.reduce((sum, b) => sum + b.confidence, 0) / beliefs.length;
      acc[level] = avg;
      return acc;
    },
    {} as Record<string, number>,
  );

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            <CardTitle>Belief State: {agent.name}</CardTitle>
          </div>
          <Badge variant="secondary">{beliefs.length} beliefs</Badge>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto">
        <div className="space-y-6">
          {/* Confidence Overview */}
          <div>
            <h3 className="text-sm font-medium mb-3">Confidence Levels</h3>
            <div className="space-y-2">
              {Object.entries(avgConfidenceByLevel).map(
                ([level, confidence]) => {
                  const Icon =
                    level === "high"
                      ? Lightbulb
                      : level === "medium"
                        ? Eye
                        : AlertCircle;
                  return (
                    <div key={level} className="flex items-center gap-3">
                      <Icon className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm capitalize w-24">{level}</span>
                      <Progress value={confidence * 100} className="flex-1" />
                      <span className="text-sm text-muted-foreground w-12 text-right">
                        {(confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                },
              )}
            </div>
          </div>

          {/* Belief Details */}
          <div>
            <h3 className="text-sm font-medium mb-3">Belief Details</h3>
            <Accordion type="single" collapsible className="w-full">
              {Object.entries(beliefsByConfidence).map(
                ([level, levelBeliefs]) => (
                  <AccordionItem key={level} value={level}>
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="secondary"
                          className={
                            level === "high"
                              ? "bg-green-500"
                              : level === "medium"
                                ? "bg-yellow-500"
                                : "bg-red-500"
                          }
                        >
                          {levelBeliefs.length}
                        </Badge>
                        <span className="capitalize">
                          {level} Confidence Beliefs
                        </span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-3 pt-2">
                        {levelBeliefs.map((belief) => (
                          <div
                            key={belief.id}
                            className="border rounded-lg p-3"
                          >
                            <p className="text-sm mb-2">{belief.content}</p>
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground">
                                  Confidence:{" "}
                                  {(belief.confidence * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                ),
              )}
            </Accordion>
          </div>

          {/* Memory Summary */}
          {memory && (
            <div>
              <h3 className="text-sm font-medium mb-3">Memory Overview</h3>
              <Card className="p-3">
                <div className="text-xs text-muted-foreground">Memory ID</div>
                <div className="text-sm font-medium">{memory.id}</div>
                <div className="text-xs text-muted-foreground mt-2">Type</div>
                <div className="text-sm">{memory.type}</div>
                <div className="text-xs text-muted-foreground mt-2">
                  Content
                </div>
                <div className="text-sm">{memory.content}</div>
              </Card>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
