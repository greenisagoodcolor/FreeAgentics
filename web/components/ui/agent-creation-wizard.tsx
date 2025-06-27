"use client";

import React, { useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  CheckCircle2,
  ArrowLeft,
  ArrowRight,
  Rocket,
  Info,
  AlertTriangle,
  Brain,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  AgentTemplateSelector,
  AgentTemplate,
  AGENT_TEMPLATES,
} from "./agent-template-selector";
import {
  AgentConfigurationForm,
  AgentConfigurationData,
} from "./agent-configuration-form";

// Wizard step definitions
const WIZARD_STEPS = [
  {
    id: "template",
    title: "Select Template",
    description: "Choose an Active Inference agent template",
    icon: Brain,
  },
  {
    id: "configure",
    title: "Configure Parameters",
    description: "Set mathematical and behavioral parameters",
    icon: CheckCircle2,
  },
  {
    id: "review",
    title: "Review & Deploy",
    description: "Review configuration and create agent",
    icon: Rocket,
  },
] as const;

type WizardStep = (typeof WIZARD_STEPS)[number]["id"];

interface AgentCreationWizardProps {
  onAgentCreate: (
    template: AgentTemplate,
    configuration: AgentConfigurationData,
  ) => Promise<void>;
  onCancel: () => void;
  className?: string;
  initialTemplate?: AgentTemplate;
}

export function AgentCreationWizard({
  onAgentCreate,
  onCancel,
  className,
  initialTemplate,
}: AgentCreationWizardProps) {
  const [currentStep, setCurrentStep] = useState<WizardStep>("template");
  const [selectedTemplate, setSelectedTemplate] =
    useState<AgentTemplate | null>(initialTemplate || null);
  const [configurationData, setConfigurationData] =
    useState<AgentConfigurationData | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Calculate progress percentage
  const stepIndex = WIZARD_STEPS.findIndex((step) => step.id === currentStep);
  const progress = ((stepIndex + 1) / WIZARD_STEPS.length) * 100;

  // Navigation helpers
  const canGoNext = () => {
    switch (currentStep) {
      case "template":
        return selectedTemplate !== null;
      case "configure":
        return configurationData !== null;
      case "review":
        return true;
      default:
        return false;
    }
  };

  const canGoPrevious = () => {
    return stepIndex > 0;
  };

  const goNext = useCallback(() => {
    const nextIndex = stepIndex + 1;
    if (nextIndex < WIZARD_STEPS.length) {
      setCurrentStep(WIZARD_STEPS[nextIndex].id);
      setError(null);
    }
  }, [stepIndex]);

  const goPrevious = () => {
    const prevIndex = stepIndex - 1;
    if (prevIndex >= 0) {
      setCurrentStep(WIZARD_STEPS[prevIndex].id);
      setError(null);
    }
  };

  // Event handlers
  const handleTemplateSelect = useCallback((template: AgentTemplate) => {
    setSelectedTemplate(template);
    setConfigurationData(null); // Reset configuration when template changes
    setError(null);
  }, []);

  const handleConfiguration = useCallback(
    (data: AgentConfigurationData) => {
      setConfigurationData(data);
      setError(null);
      goNext(); // Auto-advance to review step
    },
    [goNext],
  );

  const handleCreateAgent = async () => {
    if (!selectedTemplate || !configurationData) {
      setError("Missing template or configuration data");
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      await onAgentCreate(selectedTemplate, configurationData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setIsCreating(false);
    }
  };

  const getCurrentStepInfo = () => {
    return WIZARD_STEPS.find((step) => step.id === currentStep);
  };

  const stepInfo = getCurrentStepInfo();

  return (
    <div className={cn("max-w-6xl mx-auto space-y-6", className)}>
      {/* Header */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">
              Create Active Inference Agent
            </h1>
            <p className="text-muted-foreground">
              Design and deploy a mathematically rigorous AI agent
            </p>
          </div>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        </div>

        {/* Progress and Steps */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Step {stepIndex + 1} of {WIZARD_STEPS.length}
            </div>
            <div className="text-sm text-muted-foreground">
              {Math.round(progress)}% Complete
            </div>
          </div>
          <Progress value={progress} className="w-full" />

          {/* Step Navigation */}
          <div className="flex items-center justify-center space-x-4">
            {WIZARD_STEPS.map((step, index) => {
              const isActive = step.id === currentStep;
              const isCompleted = index < stepIndex;
              const Icon = step.icon;

              return (
                <div
                  key={step.id}
                  className={cn(
                    "flex items-center space-x-2",
                    isActive && "text-primary",
                    isCompleted && "text-green-600",
                    !isActive && !isCompleted && "text-muted-foreground",
                  )}
                >
                  <div
                    className={cn(
                      "flex items-center justify-center w-8 h-8 rounded-full border-2",
                      isActive &&
                        "border-primary bg-primary text-primary-foreground",
                      isCompleted && "border-green-600 bg-green-600 text-white",
                      !isActive && !isCompleted && "border-muted",
                    )}
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="h-4 w-4" />
                    ) : (
                      <Icon className="h-4 w-4" />
                    )}
                  </div>
                  <span className="text-sm font-medium">{step.title}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Current Step Content */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {stepInfo && <stepInfo.icon className="h-5 w-5" />}
            {stepInfo?.title}
          </CardTitle>
          <CardDescription>{stepInfo?.description}</CardDescription>
        </CardHeader>
        <CardContent>
          {/* Template Selection Step */}
          {currentStep === "template" && (
            <AgentTemplateSelector
              selectedTemplate={selectedTemplate}
              onTemplateSelect={handleTemplateSelect}
              showMathematicalDetails={true}
            />
          )}

          {/* Configuration Step */}
          {currentStep === "configure" && selectedTemplate && (
            <AgentConfigurationForm
              template={selectedTemplate}
              onSubmit={handleConfiguration}
              onCancel={goPrevious}
              isLoading={false}
            />
          )}

          {/* Review Step */}
          {currentStep === "review" &&
            selectedTemplate &&
            configurationData && (
              <div className="space-y-6">
                {/* Template Summary */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">
                    Agent Configuration Summary
                  </h3>

                  <div className="grid gap-4 md:grid-cols-2">
                    {/* Template Info */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base flex items-center gap-2">
                          {selectedTemplate.icon}
                          Selected Template
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="font-medium">
                            {selectedTemplate.name}
                          </span>
                          <Badge variant="outline">
                            {selectedTemplate.complexity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {selectedTemplate.description}
                        </p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="font-medium">States:</span>{" "}
                            {
                              selectedTemplate.mathematicalFoundation
                                .beliefsStates
                            }
                          </div>
                          <div>
                            <span className="font-medium">Actions:</span>{" "}
                            {
                              selectedTemplate.mathematicalFoundation
                                .actionSpaces
                            }
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Configuration Info */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">
                          Agent Configuration
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div>
                          <span className="font-medium">Name:</span>{" "}
                          {configurationData.name}
                        </div>
                        {configurationData.description && (
                          <div>
                            <span className="font-medium">Description:</span>{" "}
                            <span className="text-sm">
                              {configurationData.description}
                            </span>
                          </div>
                        )}
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="font-medium">
                              Sensory Precision:
                            </span>{" "}
                            {
                              configurationData.mathematics.precision
                                .sensoryPrecision
                            }
                          </div>
                          <div>
                            <span className="font-medium">
                              Policy Precision:
                            </span>{" "}
                            {
                              configurationData.mathematics.precision
                                .policyPrecision
                            }
                          </div>
                          <div>
                            <span className="font-medium">
                              Planning Horizon:
                            </span>{" "}
                            {configurationData.mathematics.planningHorizon}
                          </div>
                          <div>
                            <span className="font-medium">Learning:</span>{" "}
                            {configurationData.mathematics.enableLearning
                              ? "Enabled"
                              : "Disabled"}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Mathematical Parameters */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">
                        Mathematical Foundation
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div className="space-y-1">
                          <div className="font-medium text-muted-foreground">
                            A Matrix
                          </div>
                          <div>
                            {
                              configurationData.mathematics.matrices.aMatrix
                                .rows
                            }{" "}
                            ×{" "}
                            {
                              configurationData.mathematics.matrices.aMatrix
                                .cols
                            }
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="font-medium text-muted-foreground">
                            B Matrix
                          </div>
                          <div>
                            {
                              configurationData.mathematics.matrices.bMatrix
                                .observations
                            }{" "}
                            ×{" "}
                            {
                              configurationData.mathematics.matrices.bMatrix
                                .states
                            }
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="font-medium text-muted-foreground">
                            Memory
                          </div>
                          <div>
                            {configurationData.mathematics.memoryCapacity.toLocaleString()}
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="font-medium text-muted-foreground">
                            Hierarchical
                          </div>
                          <div>
                            {configurationData.mathematics.useHierarchical
                              ? "Yes"
                              : "No"}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Expert Review Box */}
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      <div className="space-y-2">
                        <p className="font-semibold">Ready for Expert Review</p>
                        <p className="text-sm">
                          This configuration follows Active Inference
                          mathematical principles and is ready for deployment.
                          The agent will use Bayesian belief updating with
                          precision parameters optimized for{" "}
                          {selectedTemplate.category} behavior.
                        </p>
                      </div>
                    </AlertDescription>
                  </Alert>
                </div>
              </div>
            )}
        </CardContent>
      </Card>

      {/* Navigation Footer */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          {canGoPrevious() && (
            <Button variant="outline" onClick={goPrevious}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Previous
            </Button>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {currentStep !== "review" && (
            <Button
              onClick={goNext}
              disabled={!canGoNext()}
              className="min-w-[100px]"
            >
              Next
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          )}

          {currentStep === "review" && (
            <Button
              onClick={handleCreateAgent}
              disabled={isCreating || !selectedTemplate || !configurationData}
              className="min-w-[140px]"
            >
              {isCreating ? (
                <>Creating...</>
              ) : (
                <>
                  <Rocket className="h-4 w-4 mr-2" />
                  Create Agent
                </>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export type { WizardStep };
export { WIZARD_STEPS };
