"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  ConversationPreview,
  GhostMessage,
  ProbabilityIndicator,
  ConversationPreset,
} from "@/lib/types";
import { Eye, Clock, RotateCcw, Zap } from "lucide-react";

interface RealTimePreviewProps {
  preview: ConversationPreview | null;
  preset: ConversationPreset | null;
  isPreviewMode: boolean;
  onRollback?: () => void;
  className?: string;
}

export function RealTimePreview({
  preview,
  preset,
  isPreviewMode,
  onRollback,
  className = "",
}: RealTimePreviewProps) {
  const [ghostMessages, setGhostMessages] = useState<GhostMessage[]>([]);
  const [probabilityIndicators, setProbabilityIndicators] = useState<
    ProbabilityIndicator[]
  >([]);
  const [isSimulating, setIsSimulating] = useState(false);

  // Simulate ghost messages based on current preset
  useEffect(() => {
    if (!preset || !isPreviewMode) {
      setGhostMessages([]);
      setProbabilityIndicators([]);
      return;
    }

    setIsSimulating(true);

    // Simulate some sample ghost messages
    const sampleGhosts: GhostMessage[] = [
      {
        id: "ghost-1",
        agentId: "explorer-1",
        agentName: "Explorer Alpha",
        content: "I think we should explore the northern sector first...",
        probability: preset.responseDynamics.turnTaking.responseThreshold,
        estimatedDelay: preset.timingControls.responseDelay.fixedDelay || 1000,
        confidence: 0.75,
        isVisible: true,
        fadeOutTime:
          preset.timingControls.realTimeControls.ghostMessageDuration,
      },
      {
        id: "ghost-2",
        agentId: "merchant-1",
        agentName: "Merchant Beta",
        content: "The resource costs for that operation might be high...",
        probability: preset.responseDynamics.turnTaking.responseThreshold * 0.8,
        estimatedDelay:
          (preset.timingControls.responseDelay.maxDelay || 2000) * 1.2,
        confidence: 0.65,
        isVisible: true,
        fadeOutTime:
          preset.timingControls.realTimeControls.ghostMessageDuration,
      },
    ];

    // Simulate probability indicators
    const sampleIndicators: ProbabilityIndicator[] = [
      {
        agentId: "explorer-1",
        agentName: "Explorer Alpha",
        responseprobability:
          preset.responseDynamics.turnTaking.responseThreshold,
        estimatedResponseTime:
          preset.timingControls.responseDelay.fixedDelay || 1000,
        factors: [
          { name: "Turn-taking enabled", weight: 0.3, contribution: 0.25 },
          {
            name: "Response threshold",
            weight: 0.4,
            contribution:
              preset.responseDynamics.turnTaking.responseThreshold * 0.4,
          },
          { name: "Agent expertise", weight: 0.3, contribution: 0.2 },
        ],
      },
      {
        agentId: "merchant-1",
        agentName: "Merchant Beta",
        responseprobability:
          preset.responseDynamics.turnTaking.responseThreshold * 0.8,
        estimatedResponseTime:
          (preset.timingControls.responseDelay.maxDelay || 2000) * 1.2,
        factors: [
          { name: "Turn-taking enabled", weight: 0.3, contribution: 0.2 },
          {
            name: "Response threshold",
            weight: 0.4,
            contribution:
              preset.responseDynamics.turnTaking.responseThreshold * 0.32,
          },
          { name: "Agent expertise", weight: 0.3, contribution: 0.15 },
        ],
      },
    ];

    setGhostMessages(sampleGhosts);
    setProbabilityIndicators(sampleIndicators);

    // Simulate processing delay
    setTimeout(() => setIsSimulating(false), 800);

    // Auto-fade ghost messages
    const fadeTimer = setTimeout(() => {
      setGhostMessages((prev) =>
        prev.map((msg) => ({ ...msg, isVisible: false })),
      );
    }, preset.timingControls.realTimeControls.ghostMessageDuration);

    return () => clearTimeout(fadeTimer);
  }, [preset, isPreviewMode]);

  if (!isPreviewMode) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-muted-foreground">
            <Eye className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>Enable Preview Mode to see live effects</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Real-Time Preview
          {isSimulating && (
            <Zap className="h-4 w-4 text-blue-500 animate-pulse" />
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Ghost Messages */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-sm">Ghost Messages</h4>
            <Badge variant="outline" className="text-xs">
              {ghostMessages.filter((m) => m.isVisible).length} active
            </Badge>
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {ghostMessages.map((message) => (
              <div
                key={message.id}
                className={`p-3 rounded-lg bg-muted/30 border border-dashed transition-all duration-1000 ${
                  message.isVisible ? "opacity-100" : "opacity-30"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {message.agentName}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {(message.probability * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {message.estimatedDelay}ms
                  </div>
                </div>
                <p className="text-sm text-muted-foreground italic">
                  &quot;{message.content}&quot;
                </p>
                <div className="mt-2">
                  <Progress value={message.confidence * 100} className="h-1" />
                  <div className="text-xs text-muted-foreground mt-1">
                    Confidence: {(message.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}

            {ghostMessages.length === 0 && (
              <div className="text-center py-4 text-muted-foreground text-sm">
                No ghost messages to display
              </div>
            )}
          </div>
        </div>

        {/* Probability Indicators */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-sm">Response Probabilities</h4>
            <Badge variant="outline" className="text-xs">
              {probabilityIndicators.length} agents
            </Badge>
          </div>

          <div className="space-y-3">
            {probabilityIndicators.map((indicator) => (
              <div key={indicator.agentId} className="space-y-2">
                <div className="flex items-center justify-between">
                  <Badge variant="secondary" className="text-xs">
                    {indicator.agentName}
                  </Badge>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>
                      {(indicator.responseprobability * 100).toFixed(0)}%
                    </span>
                    <Clock className="h-3 w-3" />
                    <span>{indicator.estimatedResponseTime}ms</span>
                  </div>
                </div>

                <Progress
                  value={indicator.responseprobability * 100}
                  className="h-2"
                />

                {/* Factor breakdown */}
                <div className="space-y-1">
                  {indicator.factors.map((factor, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between text-xs"
                    >
                      <span className="text-muted-foreground">
                        {factor.name}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">
                          {(factor.contribution * 100).toFixed(0)}%
                        </span>
                        <div className="w-12 bg-muted rounded-full h-1">
                          <div
                            className="bg-primary h-1 rounded-full transition-all"
                            style={{ width: `${factor.contribution * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Preview Controls */}
        <div className="pt-4 border-t space-y-2">
          <div className="flex items-center justify-between">
            <Badge variant="outline" className="text-xs">
              Preview Mode Active
            </Badge>
            {onRollback && (
              <Button
                variant="outline"
                size="sm"
                onClick={onRollback}
                className="h-7 text-xs"
              >
                <RotateCcw className="h-3 w-3 mr-1" />
                Rollback
              </Button>
            )}
          </div>

          {preset && (
            <div className="text-xs text-muted-foreground space-y-1">
              <div>
                Max Concurrent:{" "}
                {preset.responseDynamics.turnTaking.maxConcurrentResponses}
              </div>
              <div>
                Response Threshold:{" "}
                {(
                  preset.responseDynamics.turnTaking.responseThreshold * 100
                ).toFixed(0)}
                %
              </div>
              <div>Delay Type: {preset.timingControls.responseDelay.type}</div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
