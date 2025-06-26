"use client";

import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Brain, Compass, Shield, BookOpen, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

// Template interface matching our backend Active Inference templates
export interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  category: "explorer" | "guardian" | "merchant" | "scholar";
  complexity: "beginner" | "intermediate" | "advanced";
  mathematicalFoundation: {
    beliefsStates: number;
    observationModalities: number;
    actionSpaces: number;
    defaultPrecision: {
      sensory: number;
      policy: number;
      state: number;
    };
  };
  capabilities: string[];
  useCases: string[];
  expertRecommendation?: string;
}

// Available agent templates based on our implementation
const AGENT_TEMPLATES: AgentTemplate[] = [
  {
    id: "explorer",
    name: "Explorer Agent",
    description: "Epistemic value maximization for environment exploration and discovery",
    icon: <Compass className="h-6 w-6" />,
    category: "explorer",
    complexity: "beginner",
    mathematicalFoundation: {
      beliefsStates: 64,
      observationModalities: 3,
      actionSpaces: 8,
      defaultPrecision: {
        sensory: 16.0,
        policy: 16.0,
        state: 1.0,
      },
    },
    capabilities: [
      "Spatial navigation",
      "Environment mapping",
      "Resource discovery",
      "Uncertainty reduction",
      "Curiosity-driven behavior",
    ],
    useCases: [
      "Territory mapping",
      "Resource scouting",
      "Environment analysis",
      "Path optimization",
    ],
    expertRecommendation: "Ideal for newcomers to Active Inference - clear epistemic objectives",
  },
  {
    id: "guardian",
    name: "Guardian Agent",
    description: "Risk assessment and protective behavior optimization",
    icon: <Shield className="h-6 w-6" />,
    category: "guardian",
    complexity: "intermediate",
    mathematicalFoundation: {
      beliefsStates: 128,
      observationModalities: 4,
      actionSpaces: 12,
      defaultPrecision: {
        sensory: 32.0,
        policy: 24.0,
        state: 2.0,
      },
    },
    capabilities: [
      "Threat detection",
      "Risk assessment",
      "Protective behavior",
      "Multi-agent coordination",
      "Emergency response",
    ],
    useCases: [
      "Coalition protection",
      "Territory defense",
      "Risk monitoring",
      "Safety coordination",
    ],
  },
  {
    id: "merchant",
    name: "Merchant Agent",
    description: "Economic optimization and resource trading behavior",
    icon: <Brain className="h-6 w-6" />,
    category: "merchant",
    complexity: "advanced",
    mathematicalFoundation: {
      beliefsStates: 256,
      observationModalities: 5,
      actionSpaces: 16,
      defaultPrecision: {
        sensory: 64.0,
        policy: 32.0,
        state: 4.0,
      },
    },
    capabilities: [
      "Economic modeling",
      "Resource valuation",
      "Trading strategies",
      "Market analysis",
      "Coalition economics",
    ],
    useCases: [
      "Resource trading",
      "Economic planning",
      "Market optimization",
      "Coalition economics",
    ],
    expertRecommendation: "Requires strong understanding of multi-agent economic dynamics",
  },
  {
    id: "scholar",
    name: "Scholar Agent",
    description: "Knowledge synthesis and information processing optimization",
    icon: <BookOpen className="h-6 w-6" />,
    category: "scholar",
    complexity: "intermediate",
    mathematicalFoundation: {
      beliefsStates: 512,
      observationModalities: 6,
      actionSpaces: 10,
      defaultPrecision: {
        sensory: 128.0,
        policy: 64.0,
        state: 8.0,
      },
    },
    capabilities: [
      "Information synthesis",
      "Knowledge graphs",
      "Pattern recognition",
      "Research coordination",
      "Decision support",
    ],
    useCases: [
      "Research coordination",
      "Information analysis",
      "Knowledge management",
      "Strategic planning",
    ],
  },
];

interface AgentTemplateSelectorProps {
  selectedTemplate?: AgentTemplate | null;
  onTemplateSelect: (template: AgentTemplate) => void;
  className?: string;
  showMathematicalDetails?: boolean;
}

export function AgentTemplateSelector({
  selectedTemplate,
  onTemplateSelect,
  className,
  showMathematicalDetails = true,
}: AgentTemplateSelectorProps) {
  const [hoveredTemplate, setHoveredTemplate] = useState<string | null>(null);

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case "beginner":
        return "bg-green-100 text-green-800 border-green-200";
      case "intermediate":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "advanced":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-2xl font-bold">Select Agent Template</h2>
        <p className="text-muted-foreground">
          Choose an Active Inference agent template based on your requirements. Each template
          includes mathematically rigorous belief state management and behavior optimization.
        </p>
      </div>

      {/* Template Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-2">
        {AGENT_TEMPLATES.map((template) => {
          const isSelected = selectedTemplate?.id === template.id;
          const isHovered = hoveredTemplate === template.id;

          return (
            <Card
              key={template.id}
              className={cn(
                "cursor-pointer transition-all duration-200 hover:shadow-lg",
                isSelected && "ring-2 ring-primary bg-primary/5",
                isHovered && !isSelected && "shadow-md border-primary/50"
              )}
              onMouseEnter={() => setHoveredTemplate(template.id)}
              onMouseLeave={() => setHoveredTemplate(null)}
              onClick={() => onTemplateSelect(template)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-lg bg-primary/10 text-primary">
                      {template.icon}
                    </div>
                    <div>
                      <CardTitle className="text-lg">{template.name}</CardTitle>
                      <div className="flex items-center space-x-2 mt-1">
                        <Badge
                          variant="outline"
                          className={getComplexityColor(template.complexity)}
                        >
                          {template.complexity}
                        </Badge>
                        {isSelected && (
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                <CardDescription className="text-sm">
                  {template.description}
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                {/* Mathematical Foundation */}
                {showMathematicalDetails && (
                  <div className="space-y-3 p-3 bg-muted/50 rounded-lg">
                    <h4 className="text-sm font-semibold text-muted-foreground">
                      Mathematical Foundation
                    </h4>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className="font-medium">Belief States:</span>{" "}
                        {template.mathematicalFoundation.beliefsStates}
                      </div>
                      <div>
                        <span className="font-medium">Modalities:</span>{" "}
                        {template.mathematicalFoundation.observationModalities}
                      </div>
                      <div>
                        <span className="font-medium">Actions:</span>{" "}
                        {template.mathematicalFoundation.actionSpaces}
                      </div>
                      <div>
                        <span className="font-medium">Precision γ:</span>{" "}
                        {template.mathematicalFoundation.defaultPrecision.sensory}
                      </div>
                    </div>
                  </div>
                )}

                {/* Capabilities */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-muted-foreground">
                    Core Capabilities
                  </h4>
                  <div className="flex flex-wrap gap-1">
                    {template.capabilities.slice(0, 3).map((capability) => (
                      <Badge key={capability} variant="secondary" className="text-xs">
                        {capability}
                      </Badge>
                    ))}
                    {template.capabilities.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{template.capabilities.length - 3} more
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Expert Recommendation */}
                {template.expertRecommendation && (isSelected || isHovered) && (
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-xs text-blue-800">
                      <span className="font-semibold">Expert Recommendation:</span>{" "}
                      {template.expertRecommendation}
                    </p>
                  </div>
                )}

                {/* Selection Button */}
                <Button
                  variant={isSelected ? "default" : "outline"}
                  className="w-full"
                  onClick={(e) => {
                    e.stopPropagation();
                    onTemplateSelect(template);
                  }}
                >
                  {isSelected ? "Selected" : "Select Template"}
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Mathematical Details Explainer */}
      {showMathematicalDetails && (
        <div className="p-4 bg-muted/50 rounded-lg border">
          <h3 className="text-sm font-semibold mb-2">Mathematical Parameters Explained</h3>
          <div className="text-xs text-muted-foreground space-y-1">
            <p>
              <strong>Belief States:</strong> Dimensionality of the state space |S| in the probability simplex Δ^|S|
            </p>
            <p>
              <strong>Modalities:</strong> Number of observation channels in the generative model
            </p>
            <p>
              <strong>Actions:</strong> Size of the action space for policy inference
            </p>
            <p>
              <strong>Precision γ:</strong> Sensory precision parameter controlling belief update confidence
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export { AGENT_TEMPLATES }; 