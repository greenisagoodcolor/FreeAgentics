"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ConversationPreset } from "@/lib/types";
import { Clock, Timer, Play, Pause, RotateCcw, Zap } from "lucide-react";

interface TimingControlsProps {
  preset: ConversationPreset | null;
  onUpdate: (updates: Partial<ConversationPreset>) => void;
  className?: string;
}

export function TimingControls({
  preset,
  onUpdate,
  className = "",
}: TimingControlsProps) {
  if (!preset) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-muted-foreground">
            No preset selected
          </div>
        </CardContent>
      </Card>
    );
  }

  const { timingControls } = preset;

  /**
   * Update response delay settings
   */
  const updateResponseDelay = (updates: any) => {
    onUpdate({
      timingControls: {
        ...timingControls,
        responseDelay: {
          ...timingControls.responseDelay,
          ...updates,
        },
      },
    });
  };

  /**
   * Update conversation flow settings
   */
  const updateConversationFlow = (updates: any) => {
    onUpdate({
      timingControls: {
        ...timingControls,
        conversationFlow: {
          ...timingControls.conversationFlow,
          ...updates,
        },
      },
    });
  };

  /**
   * Update real-time controls settings
   */
  const updateRealTimeControls = (updates: any) => {
    onUpdate({
      timingControls: {
        ...timingControls,
        realTimeControls: {
          ...timingControls.realTimeControls,
          ...updates,
        },
      },
    });
  };

  /**
   * Format milliseconds to human readable
   */
  const formatMs = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5" />
          Timing Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Response Delay Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Timer className="h-4 w-4" />
            <Label className="font-medium">Response Delay</Label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Delay Type</Label>
              <Select
                value={timingControls.responseDelay.type}
                onValueChange={(type: any) => updateResponseDelay({ type })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed">Fixed Delay</SelectItem>
                  <SelectItem value="range">Range Delay</SelectItem>
                  <SelectItem value="adaptive">Adaptive Delay</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {timingControls.responseDelay.type === "fixed" && (
              <div className="space-y-2">
                <Label>
                  Fixed Delay (
                  {formatMs(timingControls.responseDelay.fixedDelay)})
                </Label>
                <div className="px-3">
                  <Slider
                    value={[timingControls.responseDelay.fixedDelay]}
                    onValueChange={(value) =>
                      updateResponseDelay({ fixedDelay: value[0] })
                    }
                    min={100}
                    max={5000}
                    step={100}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>100ms</span>
                    <span>
                      {formatMs(timingControls.responseDelay.fixedDelay)}
                    </span>
                    <span>5s</span>
                  </div>
                </div>
              </div>
            )}

            {timingControls.responseDelay.type === "range" && (
              <>
                <div className="space-y-2">
                  <Label>
                    Min Delay ({formatMs(timingControls.responseDelay.minDelay)}
                    )
                  </Label>
                  <div className="px-3">
                    <Slider
                      value={[timingControls.responseDelay.minDelay]}
                      onValueChange={(value) =>
                        updateResponseDelay({ minDelay: value[0] })
                      }
                      min={100}
                      max={timingControls.responseDelay.maxDelay - 100}
                      step={100}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>100ms</span>
                      <span>
                        {formatMs(timingControls.responseDelay.minDelay)}
                      </span>
                      <span>
                        {formatMs(timingControls.responseDelay.maxDelay - 100)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>
                    Max Delay ({formatMs(timingControls.responseDelay.maxDelay)}
                    )
                  </Label>
                  <div className="px-3">
                    <Slider
                      value={[timingControls.responseDelay.maxDelay]}
                      onValueChange={(value) =>
                        updateResponseDelay({ maxDelay: value[0] })
                      }
                      min={timingControls.responseDelay.minDelay + 100}
                      max={10000}
                      step={100}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>
                        {formatMs(timingControls.responseDelay.minDelay + 100)}
                      </span>
                      <span>
                        {formatMs(timingControls.responseDelay.maxDelay)}
                      </span>
                      <span>10s</span>
                    </div>
                  </div>
                </div>
              </>
            )}

            {timingControls.responseDelay.type === "adaptive" && (
              <div className="space-y-3">
                <Label>Adaptive Factors</Label>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm">Message Length</Label>
                    <Switch
                      checked={
                        timingControls.responseDelay.adaptiveFactors
                          .messageLength
                      }
                      onCheckedChange={(messageLength) =>
                        updateResponseDelay({
                          adaptiveFactors: {
                            ...timingControls.responseDelay.adaptiveFactors,
                            messageLength,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-sm">Agent Processing Time</Label>
                    <Switch
                      checked={
                        timingControls.responseDelay.adaptiveFactors
                          .agentProcessingTime
                      }
                      onCheckedChange={(agentProcessingTime) =>
                        updateResponseDelay({
                          adaptiveFactors: {
                            ...timingControls.responseDelay.adaptiveFactors,
                            agentProcessingTime,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-sm">Conversation Pace</Label>
                    <Switch
                      checked={
                        timingControls.responseDelay.adaptiveFactors
                          .conversationPace
                      }
                      onCheckedChange={(conversationPace) =>
                        updateResponseDelay({
                          adaptiveFactors: {
                            ...timingControls.responseDelay.adaptiveFactors,
                            conversationPace,
                          },
                        })
                      }
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Conversation Flow Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Play className="h-4 w-4" />
            <Label className="font-medium">Conversation Flow</Label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Max Autonomous Messages</Label>
              <div className="px-3">
                <Slider
                  value={[
                    timingControls.conversationFlow.maxAutonomousMessages,
                  ]}
                  onValueChange={(value) =>
                    updateConversationFlow({ maxAutonomousMessages: value[0] })
                  }
                  min={5}
                  max={100}
                  step={5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>5</span>
                  <span>
                    {timingControls.conversationFlow.maxAutonomousMessages}
                  </span>
                  <span>100</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>
                Stall Detection Timeout (
                {formatMs(
                  timingControls.conversationFlow.stallDetectionTimeout,
                )}
                )
              </Label>
              <div className="px-3">
                <Slider
                  value={[
                    timingControls.conversationFlow.stallDetectionTimeout,
                  ]}
                  onValueChange={(value) =>
                    updateConversationFlow({ stallDetectionTimeout: value[0] })
                  }
                  min={1000}
                  max={30000}
                  step={1000}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>1s</span>
                  <span>
                    {formatMs(
                      timingControls.conversationFlow.stallDetectionTimeout,
                    )}
                  </span>
                  <span>30s</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Stall Recovery Strategy</Label>
              <Select
                value={timingControls.conversationFlow.stallRecoveryStrategy}
                onValueChange={(stallRecoveryStrategy: any) =>
                  updateConversationFlow({ stallRecoveryStrategy })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="prompt_random">
                    Prompt Random Agent
                  </SelectItem>
                  <SelectItem value="prompt_expert">
                    Prompt Expert Agent
                  </SelectItem>
                  <SelectItem value="end_conversation">
                    End Conversation
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>
                Turn Timeout (
                {formatMs(timingControls.conversationFlow.turnTimeoutDuration)})
              </Label>
              <div className="px-3">
                <Slider
                  value={[timingControls.conversationFlow.turnTimeoutDuration]}
                  onValueChange={(value) =>
                    updateConversationFlow({ turnTimeoutDuration: value[0] })
                  }
                  min={5000}
                  max={60000}
                  step={5000}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>5s</span>
                  <span>
                    {formatMs(
                      timingControls.conversationFlow.turnTimeoutDuration,
                    )}
                  </span>
                  <span>60s</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>
                Pause Between Turns (
                {formatMs(timingControls.conversationFlow.pauseBetweenTurns)})
              </Label>
              <div className="px-3">
                <Slider
                  value={[timingControls.conversationFlow.pauseBetweenTurns]}
                  onValueChange={(value) =>
                    updateConversationFlow({ pauseBetweenTurns: value[0] })
                  }
                  min={0}
                  max={5000}
                  step={100}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0ms</span>
                  <span>
                    {formatMs(
                      timingControls.conversationFlow.pauseBetweenTurns,
                    )}
                  </span>
                  <span>5s</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Real-Time Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            <Label className="font-medium">Real-Time Controls</Label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Enable Typing Indicators</Label>
                <Switch
                  checked={
                    timingControls.realTimeControls.enableTypingIndicators
                  }
                  onCheckedChange={(enableTypingIndicators) =>
                    updateRealTimeControls({ enableTypingIndicators })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Message Preview Enabled</Label>
                <Switch
                  checked={
                    timingControls.realTimeControls.messagePreviewEnabled
                  }
                  onCheckedChange={(messagePreviewEnabled) =>
                    updateRealTimeControls({ messagePreviewEnabled })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>
                Typing Indicator Delay (
                {formatMs(timingControls.realTimeControls.typingIndicatorDelay)}
                )
              </Label>
              <div className="px-3">
                <Slider
                  value={[timingControls.realTimeControls.typingIndicatorDelay]}
                  onValueChange={(value) =>
                    updateRealTimeControls({ typingIndicatorDelay: value[0] })
                  }
                  min={100}
                  max={2000}
                  step={100}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>100ms</span>
                  <span>
                    {formatMs(
                      timingControls.realTimeControls.typingIndicatorDelay,
                    )}
                  </span>
                  <span>2s</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>
                Ghost Message Duration (
                {formatMs(timingControls.realTimeControls.ghostMessageDuration)}
                )
              </Label>
              <div className="px-3">
                <Slider
                  value={[timingControls.realTimeControls.ghostMessageDuration]}
                  onValueChange={(value) =>
                    updateRealTimeControls({ ghostMessageDuration: value[0] })
                  }
                  min={1000}
                  max={10000}
                  step={500}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>1s</span>
                  <span>
                    {formatMs(
                      timingControls.realTimeControls.ghostMessageDuration,
                    )}
                  </span>
                  <span>10s</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="pt-4 border-t">
          <div className="flex items-center gap-2 mb-2">
            <RotateCcw className="h-4 w-4" />
            <Label className="font-medium">Timing Summary</Label>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">
              {timingControls.responseDelay.type} delay
            </Badge>
            <Badge variant="outline">
              {timingControls.conversationFlow.maxAutonomousMessages} max
              messages
            </Badge>
            <Badge variant="outline">
              {timingControls.conversationFlow.stallRecoveryStrategy.replace(
                "_",
                " ",
              )}
            </Badge>
            {timingControls.realTimeControls.enableTypingIndicators && (
              <Badge variant="outline">Typing indicators</Badge>
            )}
            {timingControls.realTimeControls.messagePreviewEnabled && (
              <Badge variant="outline">Message preview</Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
