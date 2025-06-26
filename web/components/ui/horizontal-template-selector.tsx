"use client";

import React, { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, Brain, Compass, Shield, BookOpen, CheckCircle2 } from "lucide-react";
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

// Template data
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
      defaultPrecision: { sensory: 16.0, policy: 16.0, state: 1.0 },
    },
    capabilities: ["Spatial navigation", "Environment mapping", "Resource discovery", "Uncertainty reduction"],
    useCases: ["Territory mapping", "Resource scouting", "Environment analysis"],
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
      defaultPrecision: { sensory: 32.0, policy: 24.0, state: 2.0 },
    },
    capabilities: ["Threat detection", "Risk assessment", "Protective behavior", "Multi-agent coordination"],
    useCases: ["Coalition protection", "Territory defense", "Risk monitoring"],
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
      defaultPrecision: { sensory: 64.0, policy: 32.0, state: 4.0 },
    },
    capabilities: ["Economic modeling", "Resource valuation", "Trading strategies", "Market analysis"],
    useCases: ["Resource trading", "Economic planning", "Market optimization"],
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
      defaultPrecision: { sensory: 128.0, policy: 64.0, state: 8.0 },
    },
    capabilities: ["Information synthesis", "Knowledge graphs", "Pattern recognition", "Research coordination"],
    useCases: ["Research coordination", "Information analysis", "Knowledge management"],
  },
];

interface HorizontalTemplateSelectorProps {
  selectedTemplate?: AgentTemplate | null;
  onTemplateSelect: (template: AgentTemplate) => void;
  className?: string;
  showMathematicalDetails?: boolean;
  showNavigationButtons?: boolean;
}

export function HorizontalTemplateSelector({
  selectedTemplate,
  onTemplateSelect,
  className,
  showMathematicalDetails = false,
  showNavigationButtons = true,
}: HorizontalTemplateSelectorProps) {
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

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

  const updateScrollButtons = () => {
    if (scrollContainerRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollContainerRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 1);
    }
  };

  const scrollLeft = () => {
    if (scrollContainerRef.current) {
      const cardWidth = 320;
      scrollContainerRef.current.scrollBy({
        left: -cardWidth,
        behavior: "smooth",
      });
    }
  };

  const scrollRight = () => {
    if (scrollContainerRef.current) {
      const cardWidth = 320;
      scrollContainerRef.current.scrollBy({
        left: cardWidth,
        behavior: "smooth",
      });
    }
  };

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container) {
      const handleScroll = () => updateScrollButtons();
      container.addEventListener("scroll", handleScroll, { passive: true });
      updateScrollButtons();
      return () => container.removeEventListener("scroll", handleScroll);
    }
  }, []);

  return (
    <div className={cn("space-y-4", className)}>
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Choose Agent Template</h2>
        <p className="text-sm text-muted-foreground">
          Select from mathematically rigorous Active Inference agent templates
        </p>
      </div>

      <div className="relative">
        {showNavigationButtons && (
          <>
            <Button
              variant="outline"
              size="icon"
              className={cn(
                "absolute left-0 top-1/2 z-10 -translate-y-1/2 -translate-x-1/2 rounded-full shadow-lg",
                !canScrollLeft && "opacity-50 cursor-not-allowed"
              )}
              onClick={scrollLeft}
              disabled={!canScrollLeft}
              aria-label="Scroll templates left"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            
            <Button
              variant="outline"
              size="icon"
              className={cn(
                "absolute right-0 top-1/2 z-10 -translate-y-1/2 translate-x-1/2 rounded-full shadow-lg",
                !canScrollRight && "opacity-50 cursor-not-allowed"
              )}
              onClick={scrollRight}
              disabled={!canScrollRight}
              aria-label="Scroll templates right"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </>
        )}

        <div
          ref={scrollContainerRef}
          className="flex space-x-4 overflow-x-auto scrollbar-hide scroll-smooth"
          style={{
            scrollSnapType: "x mandatory",
            scrollbarWidth: "none",
            msOverflowStyle: "none",
          }}
          role="listbox"
          aria-label="Agent template selector"
        >
          {AGENT_TEMPLATES.map((template) => {
            const isSelected = selectedTemplate?.id === template.id;

            return (
              <Card
                key={template.id}
                className={cn(
                  "flex-none w-80 cursor-pointer transition-all duration-200 hover:shadow-lg",
                  "snap-start",
                  isSelected && "ring-2 ring-primary bg-primary/5"
                )}
                onClick={() => onTemplateSelect(template)}
                role="option"
                aria-selected={isSelected}
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
                  <CardDescription className="text-sm line-clamp-2">
                    {template.description}
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-3">
                  {showMathematicalDetails && (
                    <div className="space-y-2 p-2 bg-muted/50 rounded-lg">
                      <h4 className="text-xs font-semibold text-muted-foreground">
                        Mathematical Foundation
                      </h4>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="font-medium">States:</span>{" "}
                          {template.mathematicalFoundation.beliefsStates}
                        </div>
                        <div>
                          <span className="font-medium">Actions:</span>{" "}
                          {template.mathematicalFoundation.actionSpaces}
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-muted-foreground">
                      Key Capabilities
                    </h4>
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

                  {template.expertRecommendation && isSelected && (
                    <div className="p-2 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-xs text-blue-800 line-clamp-2">
                        <span className="font-semibold">Expert:</span>{" "}
                        {template.expertRecommendation}
                      </p>
                    </div>
                  )}

                  <Button
                    variant={isSelected ? "default" : "outline"}
                    size="sm"
                    className="w-full"
                    onClick={(e) => {
                      e.stopPropagation();
                      onTemplateSelect(template);
                    }}
                  >
                    {isSelected ? "Selected" : "Select"}
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      <div className="flex justify-center space-x-1">
        {AGENT_TEMPLATES.map((_, index) => (
          <div
            key={index}
            className={cn(
              "h-1.5 w-6 rounded-full transition-colors",
              index === 0 ? "bg-primary" : "bg-muted"
            )}
          />
        ))}
      </div>
    </div>
  );
}

export { AGENT_TEMPLATES };
