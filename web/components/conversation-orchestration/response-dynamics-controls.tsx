"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { ConversationPreset } from '@/lib/types';
import { 
  Users, 
  Target, 
  Brain, 
  Shuffle,
  MessageSquare,
  Sparkles
} from 'lucide-react';

interface ResponseDynamicsControlsProps {
  preset: ConversationPreset | null;
  onUpdate: (updates: Partial<ConversationPreset>) => void;
  className?: string;
}

export function ResponseDynamicsControls({
  preset,
  onUpdate,
  className = ""
}: ResponseDynamicsControlsProps) {
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

  const { responseDynamics } = preset;

  /**
   * Update turn-taking settings
   */
  const updateTurnTaking = (updates: Partial<typeof responseDynamics.turnTaking>) => {
    onUpdate({
      responseDynamics: {
        ...responseDynamics,
        turnTaking: {
          ...responseDynamics.turnTaking,
          ...updates
        }
      }
    });
  };

  /**
   * Update agent selection settings
   */
  const updateAgentSelection = (updates: Partial<typeof responseDynamics.agentSelection>) => {
    onUpdate({
      responseDynamics: {
        ...responseDynamics,
        agentSelection: {
          ...responseDynamics.agentSelection,
          ...updates
        }
      }
    });
  };

  /**
   * Update response generation settings
   */
  const updateResponseGeneration = (updates: Partial<typeof responseDynamics.responseGeneration>) => {
    onUpdate({
      responseDynamics: {
        ...responseDynamics,
        responseGeneration: {
          ...responseDynamics.responseGeneration,
          ...updates
        }
      }
    });
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          Response Dynamics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Turn-Taking Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            <Label className="font-medium">Turn-Taking Behavior</Label>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Enable Turn-Taking</Label>
                <Switch
                  checked={responseDynamics.turnTaking.enabled}
                  onCheckedChange={(enabled) => updateTurnTaking({ enabled })}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Max Concurrent Responses</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.turnTaking.maxConcurrentResponses]}
                  onValueChange={(value) => updateTurnTaking({ maxConcurrentResponses: value[0] })}
                  min={1}
                  max={5}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>1</span>
                  <span>{responseDynamics.turnTaking.maxConcurrentResponses}</span>
                  <span>5</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Response Threshold</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.turnTaking.responseThreshold]}
                  onValueChange={(value) => updateTurnTaking({ responseThreshold: value[0] })}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.1</span>
                  <span>{responseDynamics.turnTaking.responseThreshold.toFixed(1)}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Mention Response Rate</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.turnTaking.mentionResponseProbability]}
                  onValueChange={(value) => updateTurnTaking({ mentionResponseProbability: value[0] })}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.1</span>
                  <span>{responseDynamics.turnTaking.mentionResponseProbability.toFixed(1)}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Agent Selection Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            <Label className="font-medium">Agent Selection</Label>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Auto-Select Respondents</Label>
                <Switch
                  checked={responseDynamics.agentSelection.autoSelectRespondents}
                  onCheckedChange={(autoSelectRespondents) => 
                    updateAgentSelection({ autoSelectRespondents })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Selection Strategy</Label>
              <Select
                value={responseDynamics.agentSelection.selectionStrategy}
                onValueChange={(selectionStrategy: any) => 
                  updateAgentSelection({ selectionStrategy })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="random">Random</SelectItem>
                  <SelectItem value="round_robin">Round Robin</SelectItem>
                  <SelectItem value="expertise_based">Expertise Based</SelectItem>
                  <SelectItem value="engagement_based">Engagement Based</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Diversity Bonus</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.agentSelection.diversityBonus]}
                  onValueChange={(value) => updateAgentSelection({ diversityBonus: value[0] })}
                  min={0.0}
                  max={1.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.0</span>
                  <span>{responseDynamics.agentSelection.diversityBonus.toFixed(1)}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Expertise Weight</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.agentSelection.expertiseWeight]}
                  onValueChange={(value) => updateAgentSelection({ expertiseWeight: value[0] })}
                  min={0.0}
                  max={1.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.0</span>
                  <span>{responseDynamics.agentSelection.expertiseWeight.toFixed(1)}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Response Generation Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            <Label className="font-medium">Response Generation</Label>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Include Agent Knowledge</Label>
                <Switch
                  checked={responseDynamics.responseGeneration.includeAgentKnowledge}
                  onCheckedChange={(includeAgentKnowledge) => 
                    updateResponseGeneration({ includeAgentKnowledge })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Stream Response</Label>
                <Switch
                  checked={responseDynamics.responseGeneration.streamResponse}
                  onCheckedChange={(streamResponse) => 
                    updateResponseGeneration({ streamResponse })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Max Knowledge Entries</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.responseGeneration.maxKnowledgeEntries]}
                  onValueChange={(value) => updateResponseGeneration({ maxKnowledgeEntries: value[0] })}
                  min={0}
                  max={50}
                  step={5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0</span>
                  <span>{responseDynamics.responseGeneration.maxKnowledgeEntries}</span>
                  <span>50</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Response Length</Label>
              <Select
                value={responseDynamics.responseGeneration.responseLength}
                onValueChange={(responseLength: any) => 
                  updateResponseGeneration({ responseLength })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="short">Short</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="long">Long</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Creativity Level</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.responseGeneration.creativityLevel]}
                  onValueChange={(value) => updateResponseGeneration({ creativityLevel: value[0] })}
                  min={0.0}
                  max={1.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.0</span>
                  <span>{responseDynamics.responseGeneration.creativityLevel.toFixed(1)}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Coherence Weight</Label>
              <div className="px-3">
                <Slider
                  value={[responseDynamics.responseGeneration.coherenceWeight]}
                  onValueChange={(value) => updateResponseGeneration({ coherenceWeight: value[0] })}
                  min={0.0}
                  max={1.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0.0</span>
                  <span>{responseDynamics.responseGeneration.coherenceWeight.toFixed(1)}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="pt-4 border-t">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles className="h-4 w-4" />
            <Label className="font-medium">Current Settings Summary</Label>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">
              {responseDynamics.turnTaking.maxConcurrentResponses} Concurrent
            </Badge>
            <Badge variant="outline">
              {(responseDynamics.turnTaking.responseThreshold * 100).toFixed(0)}% Threshold
            </Badge>
            <Badge variant="outline">
              {responseDynamics.agentSelection.selectionStrategy.replace('_', ' ')}
            </Badge>
            <Badge variant="outline">
              {responseDynamics.responseGeneration.responseLength} Response
            </Badge>
            <Badge variant="outline">
              {responseDynamics.responseGeneration.maxKnowledgeEntries} Knowledge
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 