"use client";

import React, { useState, useCallback, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Brain,
  Zap,
  Activity,
  Settings,
  Play,
  Pause,
  RotateCcw,
  TrendingUp,
  Eye,
  Target,
  Info,
  AlertCircle,
} from "lucide-react";
import {
  BeliefStateVisualization,
  BeliefStateData,
} from "./belief-state-visualization";
import {
  FreeEnergyVisualization,
  FreeEnergyData,
} from "./free-energy-visualization";

// Types for precision monitoring
export interface PrecisionData {
  timestamp: number;
  sensory: number; // γ - sensory precision
  policy: number; // β - policy precision
  state: number; // α - state precision
  learning: {
    rate: number; // Learning rate
    momentum: number; // Momentum factor
  };
  performance: {
    accuracy: number; // Overall accuracy
    stability: number; // Precision stability
    convergence: number; // Convergence rate
  };
}

// Combined agent state
export interface AgentState {
  agentId: string;
  name: string;
  template: string;
  status: "active" | "inactive" | "learning" | "error";
  beliefs: BeliefStateData | null;
  freeEnergy: FreeEnergyData | null;
  precision: PrecisionData | null;
  lastUpdate: number;
}

interface ActiveInferenceDashboardProps {
  agentId: string;
  agentName?: string;
  template?: string;
  updateInterval?: number;
  className?: string;
  onStateChange?: (state: AgentState) => void;
  isRealTime?: boolean;
  stateLabels?: string[];
}

export function ActiveInferenceDashboard({
  agentId,
  agentName = `Agent ${agentId}`,
  template = "explorer",
  updateInterval = 1000,
  className,
  onStateChange,
  isRealTime = true,
  stateLabels = [
    "Explore",
    "Navigate",
    "Rest",
    "Communicate",
    "Learn",
    "Plan",
    "Execute",
    "Observe",
  ],
}: ActiveInferenceDashboardProps) {
  const [agentState, setAgentState] = useState<AgentState>({
    agentId,
    name: agentName,
    template,
    status: "active",
    beliefs: null,
    freeEnergy: null,
    precision: null,
    lastUpdate: Date.now(),
  });

  const [isGlobalPlay, setIsGlobalPlay] = useState(isRealTime);
  const [activeTab, setActiveTab] = useState("overview");
  const [alerts, setAlerts] = useState<string[]>([]);

  // Generate mock precision data
  const generatePrecisionData = useCallback(
    (
      beliefData?: BeliefStateData,
      freeEnergyData?: FreeEnergyData,
    ): PrecisionData => {
      const now = Date.now();

      // Use data from other components if available
      const sensory = beliefData?.precision.sensory || 16 + Math.random() * 16;
      const policy = beliefData?.precision.policy || 12 + Math.random() * 12;
      const state = beliefData?.precision.state || 2 + Math.random() * 3;

      // Performance metrics based on free energy
      const feNormalized = freeEnergyData
        ? Math.max(0, 1 - freeEnergyData.variationalFreeEnergy / 5)
        : 0.7;
      const accuracy = feNormalized * 0.8 + 0.1 + Math.random() * 0.1;
      const stability = Math.max(0.1, 1 - Math.abs(Math.sin(now / 5000)) * 0.3);
      const convergence = beliefData ? beliefData.confidence * 0.8 + 0.1 : 0.6;

      return {
        timestamp: now,
        sensory,
        policy,
        state,
        learning: {
          rate: 0.1 + Math.random() * 0.05,
          momentum: 0.9 + Math.random() * 0.05,
        },
        performance: {
          accuracy,
          stability,
          convergence,
        },
      };
    },
    [],
  );

  // Handle belief state updates
  const handleBeliefChange = useCallback(
    (beliefData: BeliefStateData) => {
      setAgentState((prev) => {
        const newPrecision = generatePrecisionData(beliefData, prev.freeEnergy);
        const newState = {
          ...prev,
          beliefs: beliefData,
          precision: newPrecision,
          lastUpdate: Date.now(),
        };

        // Check for alerts
        const newAlerts: string[] = [];
        if (beliefData.confidence < 0.3) {
          newAlerts.push("Low confidence in belief state");
        }
        if (beliefData.entropy > Math.log(stateLabels.length) * 0.9) {
          newAlerts.push("High uncertainty - agent may be confused");
        }

        setAlerts(newAlerts);
        onStateChange?.(newState);
        return newState;
      });
    },
    [generatePrecisionData, stateLabels.length, onStateChange],
  );

  // Handle free energy updates
  const handleFreeEnergyChange = useCallback(
    (freeEnergyData: FreeEnergyData) => {
      setAgentState((prev) => {
        const newPrecision = generatePrecisionData(
          prev.beliefs,
          freeEnergyData,
        );
        const newState = {
          ...prev,
          freeEnergy: freeEnergyData,
          precision: newPrecision,
          lastUpdate: Date.now(),
        };

        // Check for free energy alerts
        if (freeEnergyData.variationalFreeEnergy > 5.0) {
          setAlerts((prev) => [
            ...prev.filter((a) => !a.includes("High free energy")),
            "High free energy - agent struggling",
          ]);
        }

        onStateChange?.(newState);
        return newState;
      });
    },
    [generatePrecisionData, onStateChange],
  );

  // Global controls
  const handleGlobalPlayPause = () => {
    setIsGlobalPlay(!isGlobalPlay);
  };

  const handleGlobalReset = () => {
    setAgentState((prev) => ({
      ...prev,
      beliefs: null,
      freeEnergy: null,
      precision: null,
      lastUpdate: Date.now(),
    }));
    setAlerts([]);
  };

  // Determine agent status
  useEffect(() => {
    let status: AgentState["status"] = "active";

    if (
      agentState.freeEnergy &&
      agentState.freeEnergy.variationalFreeEnergy > 4.5
    ) {
      status = "error";
    } else if (
      agentState.precision &&
      agentState.precision.performance.convergence < 0.3
    ) {
      status = "learning";
    } else if (!isGlobalPlay) {
      status = "inactive";
    }

    if (status !== agentState.status) {
      setAgentState((prev) => ({ ...prev, status }));
    }
  }, [
    agentState.freeEnergy,
    agentState.precision,
    isGlobalPlay,
    agentState.status,
  ]);

  return (
    <div className={className}>
      {/* Dashboard Header */}
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Brain className="h-6 w-6 text-primary" />
                <div>
                  <CardTitle className="text-xl">{agentState.name}</CardTitle>
                  <CardDescription>
                    Active Inference Agent • Template: {agentState.template}
                  </CardDescription>
                </div>
              </div>

              <Badge
                variant={
                  agentState.status === "active"
                    ? "default"
                    : agentState.status === "learning"
                      ? "secondary"
                      : agentState.status === "error"
                        ? "destructive"
                        : "outline"
                }
              >
                {agentState.status.toUpperCase()}
              </Badge>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleGlobalPlayPause}
              >
                {isGlobalPlay ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {isGlobalPlay ? "Pause" : "Play"}
              </Button>
              <Button variant="outline" size="sm" onClick={handleGlobalReset}>
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-blue-600" />
              <div>
                <div className="text-sm font-medium">Confidence</div>
                <div className="text-lg font-bold">
                  {agentState.beliefs
                    ? `${(agentState.beliefs.confidence * 100).toFixed(0)}%`
                    : "--"}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-red-600" />
              <div>
                <div className="text-sm font-medium">Free Energy</div>
                <div className="text-lg font-bold">
                  {agentState.freeEnergy
                    ? agentState.freeEnergy.variationalFreeEnergy.toFixed(2)
                    : "--"}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Settings className="h-4 w-4 text-green-600" />
              <div>
                <div className="text-sm font-medium">Precision γ</div>
                <div className="text-lg font-bold">
                  {agentState.precision
                    ? agentState.precision.sensory.toFixed(1)
                    : "--"}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-600" />
              <div>
                <div className="text-sm font-medium">Performance</div>
                <div className="text-lg font-bold">
                  {agentState.precision
                    ? `${(agentState.precision.performance.accuracy * 100).toFixed(0)}%`
                    : "--"}
                </div>
              </div>
            </div>
          </div>

          {/* Alerts */}
          {alerts.length > 0 && (
            <div className="mt-4 space-y-2">
              {alerts.map((alert, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 p-2 bg-yellow-50 border border-yellow-200 rounded-lg"
                >
                  <AlertCircle className="h-4 w-4 text-yellow-600" />
                  <span className="text-sm text-yellow-800">{alert}</span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Dashboard */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="beliefs">Beliefs</TabsTrigger>
          <TabsTrigger value="energy">Free Energy</TabsTrigger>
          <TabsTrigger value="precision">Precision</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <BeliefStateVisualization
              agentId={agentId}
              stateLabels={stateLabels}
              height={300}
              updateInterval={updateInterval}
              onBeliefChange={handleBeliefChange}
              isRealTime={isGlobalPlay}
            />

            <FreeEnergyVisualization
              agentId={agentId}
              height={300}
              updateInterval={updateInterval}
              onFreeEnergyChange={handleFreeEnergyChange}
              isRealTime={isGlobalPlay}
            />
          </div>

          {/* Precision Overview */}
          {agentState.precision && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Precision Parameters
                </CardTitle>
                <CardDescription>
                  Real-time precision parameter monitoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Sensory (γ)</span>
                        <span className="text-sm text-muted-foreground">
                          {agentState.precision.sensory.toFixed(1)}
                        </span>
                      </div>
                      <Progress
                        value={Math.min(
                          100,
                          (agentState.precision.sensory / 50) * 100,
                        )}
                      />
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Policy (β)</span>
                        <span className="text-sm text-muted-foreground">
                          {agentState.precision.policy.toFixed(1)}
                        </span>
                      </div>
                      <Progress
                        value={Math.min(
                          100,
                          (agentState.precision.policy / 30) * 100,
                        )}
                      />
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">State (α)</span>
                        <span className="text-sm text-muted-foreground">
                          {agentState.precision.state.toFixed(1)}
                        </span>
                      </div>
                      <Progress
                        value={Math.min(
                          100,
                          (agentState.precision.state / 10) * 100,
                        )}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Detailed Belief States Tab */}
        <TabsContent value="beliefs">
          <BeliefStateVisualization
            agentId={agentId}
            stateLabels={stateLabels}
            height={500}
            updateInterval={updateInterval}
            onBeliefChange={handleBeliefChange}
            isRealTime={isGlobalPlay}
          />
        </TabsContent>

        {/* Detailed Free Energy Tab */}
        <TabsContent value="energy">
          <FreeEnergyVisualization
            agentId={agentId}
            height={500}
            updateInterval={updateInterval}
            onFreeEnergyChange={handleFreeEnergyChange}
            isRealTime={isGlobalPlay}
          />
        </TabsContent>

        {/* Detailed Precision Tab */}
        <TabsContent value="precision">
          <Card>
            <CardHeader>
              <CardTitle>Precision Parameter Analysis</CardTitle>
              <CardDescription>
                Detailed analysis of Active Inference precision parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              {agentState.precision ? (
                <div className="space-y-6">
                  {/* Precision Values */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">
                          Sensory Precision (γ)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-blue-600 mb-2">
                          {agentState.precision.sensory.toFixed(2)}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Controls confidence in sensory observations
                        </p>
                        <Progress
                          value={Math.min(
                            100,
                            (agentState.precision.sensory / 50) * 100,
                          )}
                          className="mt-3"
                        />
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">
                          Policy Precision (β)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-green-600 mb-2">
                          {agentState.precision.policy.toFixed(2)}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Controls determinism in policy selection
                        </p>
                        <Progress
                          value={Math.min(
                            100,
                            (agentState.precision.policy / 30) * 100,
                          )}
                          className="mt-3"
                        />
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">
                          State Precision (α)
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-purple-600 mb-2">
                          {agentState.precision.state.toFixed(2)}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Controls confidence in state transitions
                        </p>
                        <Progress
                          value={Math.min(
                            100,
                            (agentState.precision.state / 10) * 100,
                          )}
                          className="mt-3"
                        />
                      </CardContent>
                    </Card>
                  </div>

                  {/* Performance Metrics */}
                  <Separator />

                  <div>
                    <h3 className="text-lg font-semibold mb-4">
                      Performance Metrics
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-3">
                        <Target className="h-8 w-8 text-blue-600" />
                        <div>
                          <div className="font-medium">Accuracy</div>
                          <div className="text-2xl font-bold">
                            {(
                              agentState.precision.performance.accuracy * 100
                            ).toFixed(1)}
                            %
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <Activity className="h-8 w-8 text-green-600" />
                        <div>
                          <div className="font-medium">Stability</div>
                          <div className="text-2xl font-bold">
                            {(
                              agentState.precision.performance.stability * 100
                            ).toFixed(1)}
                            %
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <TrendingUp className="h-8 w-8 text-purple-600" />
                        <div>
                          <div className="font-medium">Convergence</div>
                          <div className="text-2xl font-bold">
                            {(
                              agentState.precision.performance.convergence * 100
                            ).toFixed(1)}
                            %
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Mathematical Explanation */}
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h4 className="font-semibold text-blue-900 mb-2">
                      Precision Parameter Mathematics
                    </h4>
                    <div className="text-sm text-blue-800 space-y-1">
                      <p>
                        • <strong>γ (Sensory)</strong>: Inverse variance of
                        sensory noise σ⁻²
                      </p>
                      <p>
                        • <strong>β (Policy)</strong>: Temperature parameter in
                        softmax policy selection
                      </p>
                      <p>
                        • <strong>α (State)</strong>: Precision of state
                        transition predictions
                      </p>
                      <p>
                        • Higher precision → Lower uncertainty → More confident
                        decisions
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-12">
                  <Settings className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No precision data available</p>
                  <p className="text-sm">
                    Start the agent to see precision parameters
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
