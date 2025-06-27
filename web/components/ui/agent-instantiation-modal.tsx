"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import {
  Brain,
  Compass,
  Shield,
  BookOpen,
  User,
  Settings,
  Zap,
  Activity,
  CheckCircle2,
  AlertCircle,
  Loader2,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentTemplate } from "./horizontal-template-selector";
import { HorizontalTemplateSelector } from "./horizontal-template-selector";
import {
  useAgentCreation,
  type AgentCreationConfig,
  type CreatedAgentInfo,
} from "../../lib/services/agent-creation-service";

// Mock data for activity sparklines
const generateSparklineData = () => {
  return Array.from({ length: 20 }, () => Math.random() * 100);
};

interface AgentConfig {
  name: string;
  description: string;
  position: { x: number; y: number };
}

interface AgentCard {
  id: string;
  name: string;
  template: AgentTemplate;
  status: "creating" | "initializing" | "ready" | "error";
  progress: number;
  activityData: number[];
  energy: number;
  health: number;
  lastActivity: string;
}

interface AgentInstantiationModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAgentCreated?: (agent: any) => void;
  quickStartMode?: boolean;
}

// Simple sparkline component
function Sparkline({
  data,
  className,
}: {
  data: number[];
  className?: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = svgRef.current;
    const width = 100;
    const height = 24;

    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;

    // Create path
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    });

    const path = `M ${points.join(" L ")}`;

    // Clear and add path
    svg.innerHTML = "";
    const pathElement = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    pathElement.setAttribute("d", path);
    pathElement.setAttribute("stroke", "currentColor");
    pathElement.setAttribute("stroke-width", "1.5");
    pathElement.setAttribute("fill", "none");
    pathElement.setAttribute("opacity", "0.7");

    svg.appendChild(pathElement);
  }, [data]);

  return (
    <svg
      ref={svgRef}
      className={cn("w-full h-6", className)}
      viewBox="0 0 100 24"
    />
  );
}

// Agent status indicator
function AgentStatusIndicator({
  status,
}: {
  status: CreatedAgentInfo["status"];
}) {
  const statusConfig = {
    creating: {
      icon: Loader2,
      color: "text-blue-500",
      label: "Creating",
      animate: true,
    },
    initializing: {
      icon: Settings,
      color: "text-yellow-500",
      label: "Initializing",
      animate: true,
    },
    ready: {
      icon: CheckCircle2,
      color: "text-green-500",
      label: "Ready",
      animate: false,
    },
    error: {
      icon: AlertCircle,
      color: "text-red-500",
      label: "Error",
      animate: false,
    },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div className="flex items-center space-x-2">
      <Icon
        className={cn(
          "h-4 w-4",
          config.color,
          config.animate && "animate-spin",
        )}
      />
      <span className="text-sm text-muted-foreground">{config.label}</span>
    </div>
  );
}

// Individual agent card component
function AgentInstanceCard({
  agent,
  template,
}: {
  agent: CreatedAgentInfo;
  template?: AgentTemplate;
}) {
  const activityData = generateSparklineData();
  const energy = Math.floor(Math.random() * 40) + 60;
  const health = Math.floor(Math.random() * 30) + 70;

  return (
    <Card className="relative overflow-hidden transition-all duration-300 hover:shadow-lg">
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary/20 to-primary/60" />

      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              {template?.icon || <Brain className="h-6 w-6" />}
            </div>
            <div>
              <CardTitle className="text-base">{agent.name}</CardTitle>
              <CardDescription className="text-sm">
                {template?.name || agent.templateId}
              </CardDescription>
            </div>
          </div>
          <AgentStatusIndicator status={agent.status} />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress bar for creating/initializing agents */}
        {(agent.status === "creating" || agent.status === "initializing") && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{agent.progress}%</span>
            </div>
            <Progress value={agent.progress} className="h-2" />
          </div>
        )}

        {/* Error message */}
        {agent.status === "error" && agent.error && (
          <div className="p-2 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-xs text-red-800">{agent.error}</p>
          </div>
        )}

        {/* Resource indicators */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <div className="flex items-center space-x-1">
              <Zap className="h-3 w-3 text-yellow-500" />
              <span className="text-xs text-muted-foreground">Energy</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-yellow-400 to-yellow-600 transition-all duration-500"
                  style={{ width: `${energy}%` }}
                />
              </div>
              <span className="text-xs font-medium">{energy}%</span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center space-x-1">
              <Activity className="h-3 w-3 text-green-500" />
              <span className="text-xs text-muted-foreground">Health</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-500"
                  style={{ width: `${health}%` }}
                />
              </div>
              <span className="text-xs font-medium">{health}%</span>
            </div>
          </div>
        </div>

        {/* Activity sparkline */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1">
              <TrendingUp className="h-3 w-3 text-blue-500" />
              <span className="text-xs text-muted-foreground">Activity</span>
            </div>
            <span className="text-xs text-muted-foreground">Live</span>
          </div>
          <div className="h-6 text-blue-500">
            <Sparkline data={activityData} />
          </div>
        </div>

        {/* Agent capabilities */}
        {template && (
          <div className="space-y-2">
            <span className="text-xs font-medium text-muted-foreground">
              Capabilities
            </span>
            <div className="flex flex-wrap gap-1">
              {template.capabilities.slice(0, 2).map((capability) => (
                <Badge key={capability} variant="secondary" className="text-xs">
                  {capability}
                </Badge>
              ))}
              {template.capabilities.length > 2 && (
                <Badge variant="outline" className="text-xs">
                  +{template.capabilities.length - 2}
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function AgentInstantiationModal({
  open,
  onOpenChange,
  onAgentCreated,
  quickStartMode = false,
}: AgentInstantiationModalProps) {
  const [selectedTemplate, setSelectedTemplate] =
    useState<AgentTemplate | null>(null);
  const [agentConfig, setAgentConfig] = useState<Partial<AgentCreationConfig>>({
    name: "",
    description: "",
    position: { x: 0, y: 0 },
  });
  const [createdAgents, setCreatedAgents] = useState<CreatedAgentInfo[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [currentStep, setCurrentStep] = useState<
    "template" | "configure" | "creating" | "complete"
  >("template");

  const { createAgent, createQuickStartAgents } = useAgentCreation();
  const { toast } = useToast();

  const handleQuickStart = useCallback(async () => {
    setIsCreating(true);
    setCurrentStep("creating");

    try {
      const results = await createQuickStartAgents();

      // Extract successful agents
      const successfulAgents = results
        .filter((result) => result.success && result.agent)
        .map((result) => result.agent!);

      setCreatedAgents(successfulAgents);

      // Show notifications
      const successCount = results.filter((r) => r.success).length;
      const errorCount = results.filter((r) => !r.success).length;

      if (successCount > 0) {
        toast({
          title: "Quick Start Complete",
          description: `Successfully created ${successCount} agents${errorCount > 0 ? ` (${errorCount} failed)` : ""}`,
        });
      }

      if (errorCount > 0) {
        toast({
          title: "Some agents failed to create",
          description: "Check the error messages for details",
          variant: "destructive",
        });
      }

      setCurrentStep("complete");
    } catch (error) {
      console.error("Quick start failed:", error);
      toast({
        title: "Quick Start Failed",
        description:
          error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsCreating(false);
    }
  }, [toast, createQuickStartAgents]);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (!open) {
      setCurrentStep("template");
      setSelectedTemplate(null);
      setAgentConfig({
        name: "",
        description: "",
        position: { x: 0, y: 0 },
      });
      setCreatedAgents([]);
      setIsCreating(false);
    }
  }, [open]);

  // Quick start mode
  useEffect(() => {
    if (quickStartMode && open && currentStep === "template") {
      handleQuickStart();
    }
  }, [quickStartMode, open, currentStep, handleQuickStart]);

  const handleCreateAgent = async () => {
    if (!selectedTemplate) return;

    setIsCreating(true);
    setCurrentStep("creating");

    try {
      const result = await createAgent(selectedTemplate, agentConfig);

      if (result.success && result.agent) {
        setCreatedAgents([result.agent]);

        toast({
          title: "Agent Created",
          description: `${result.agent.name} has been created successfully`,
        });

        if (onAgentCreated) {
          onAgentCreated(result.agent);
        }

        setCurrentStep("complete");
      } else {
        toast({
          title: "Agent Creation Failed",
          description: result.error || "Unknown error occurred",
          variant: "destructive",
        });
        setCurrentStep("configure");
      }
    } catch (error) {
      console.error("Agent creation failed:", error);
      toast({
        title: "Agent Creation Failed",
        description:
          error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
      setCurrentStep("configure");
    } finally {
      setIsCreating(false);
    }
  };

  const getStepTitle = () => {
    switch (currentStep) {
      case "template":
        return "Select Agent Template";
      case "configure":
        return "Configure Agent";
      case "creating":
        return "Creating Agent...";
      case "complete":
        return "Agent Created Successfully";
      default:
        return "Create Agent";
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <User className="h-5 w-5" />
            <span>{getStepTitle()}</span>
          </DialogTitle>
          <DialogDescription>
            {currentStep === "template" &&
              "Choose a template to create your Active Inference agent"}
            {currentStep === "configure" &&
              "Customize your agent's configuration"}
            {currentStep === "creating" &&
              "Your agent is being created with mathematical rigor"}
            {currentStep === "complete" && "Your agent is ready for deployment"}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Step 1: Template Selection */}
          {currentStep === "template" && (
            <div className="space-y-4">
              <HorizontalTemplateSelector
                selectedTemplate={selectedTemplate}
                onTemplateSelect={setSelectedTemplate}
                showMathematicalDetails={true}
              />

              <div className="flex justify-between">
                <Button variant="outline" onClick={handleQuickStart}>
                  Quick Start (3 Default Agents)
                </Button>
                <Button
                  onClick={() => setCurrentStep("configure")}
                  disabled={!selectedTemplate}
                >
                  Configure Agent
                </Button>
              </div>
            </div>
          )}

          {/* Step 2: Configuration */}
          {currentStep === "configure" && selectedTemplate && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="agent-name">Agent Name</Label>
                    <Input
                      id="agent-name"
                      placeholder={`${selectedTemplate.name} Instance`}
                      value={agentConfig.name}
                      onChange={(e) =>
                        setAgentConfig((prev) => ({
                          ...prev,
                          name: e.target.value,
                        }))
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="agent-description">
                      Description (Optional)
                    </Label>
                    <Textarea
                      id="agent-description"
                      placeholder="Describe this agent's purpose..."
                      value={agentConfig.description}
                      onChange={(e) =>
                        setAgentConfig((prev) => ({
                          ...prev,
                          description: e.target.value,
                        }))
                      }
                      rows={3}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="pos-x">Position X</Label>
                      <Input
                        id="pos-x"
                        type="number"
                        value={agentConfig.position?.x || 0}
                        onChange={(e) =>
                          setAgentConfig((prev) => ({
                            ...prev,
                            position: {
                              ...prev.position,
                              x: Number(e.target.value),
                              y: prev.position?.y || 0,
                            },
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="pos-y">Position Y</Label>
                      <Input
                        id="pos-y"
                        type="number"
                        value={agentConfig.position?.y || 0}
                        onChange={(e) =>
                          setAgentConfig((prev) => ({
                            ...prev,
                            position: {
                              ...prev.position,
                              y: Number(e.target.value),
                              x: prev.position?.x || 0,
                            },
                          }))
                        }
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">
                        Template Preview
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 rounded-lg bg-primary/10 text-primary">
                          {selectedTemplate.icon}
                        </div>
                        <div>
                          <div className="font-medium">
                            {selectedTemplate.name}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {selectedTemplate.complexity}
                          </div>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {selectedTemplate.description}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {selectedTemplate.capabilities
                          .slice(0, 3)
                          .map((capability) => (
                            <Badge
                              key={capability}
                              variant="secondary"
                              className="text-xs"
                            >
                              {capability}
                            </Badge>
                          ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="flex justify-between">
                <Button
                  variant="outline"
                  onClick={() => setCurrentStep("template")}
                >
                  Back
                </Button>
                <Button onClick={handleCreateAgent} disabled={isCreating}>
                  {isCreating ? "Creating..." : "Create Agent"}
                </Button>
              </div>
            </div>
          )}

          {/* Step 3 & 4: Creating and Complete */}
          {(currentStep === "creating" || currentStep === "complete") && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {createdAgents.map((agent) => (
                  <AgentInstanceCard
                    key={agent.id}
                    agent={agent}
                    template={selectedTemplate || undefined}
                  />
                ))}
              </div>

              {currentStep === "complete" && (
                <div className="flex justify-center space-x-4">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentStep("template")}
                  >
                    Create Another
                  </Button>
                  <Button onClick={() => onOpenChange(false)}>Close</Button>
                </div>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
