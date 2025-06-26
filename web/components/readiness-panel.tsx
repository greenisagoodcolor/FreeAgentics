"use client";

import React, { useState, useEffect, useCallback } from "react";
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
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Download,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Cpu,
  Brain,
  Target,
  Users,
  Battery,
  TrendingUp,
  Sparkles,
  Trophy,
  Rocket,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface IReadinessScore {
  agent_id: string;
  timestamp: string;
  scores: {
    knowledge_maturity: number;
    goal_achievement: number;
    model_stability: number;
    collaboration: number;
    resource_management: number;
    overall: number;
  };
  is_ready: boolean;
  metrics: {
    knowledge?: {
      experience_count: number;
      pattern_count: number;
      avg_pattern_confidence: number;
    };
    goals?: {
      total_attempted: number;
      successful: number;
      success_rate: number;
      complex_completed: number;
    };
    model_stability?: {
      update_count: number;
      is_converged: boolean;
      stable_iterations: number;
    };
    collaboration?: {
      total_interactions: number;
      successful_interactions: number;
      knowledge_shared: number;
      unique_collaborators: number;
    };
    resources?: {
      energy_efficiency: number;
      resource_efficiency: number;
      sustainability_score: number;
    };
  };
  recommendations: string[];
}

interface IHardwareTarget {
  id: string;
  name: string;
  platform: string;
  cpu_arch: string;
  ram_gb: number;
  storage_gb: number;
  accelerators: string[];
}

interface ReadinessPanelProps {
  agentId: string;
  className?: string;
}

const HARDWARE_TARGETS: IHardwareTarget[] = [
  {
    id: "raspberry_pi_4b",
    name: "Raspberry Pi 4B",
    platform: "raspberrypi",
    cpu_arch: "arm64",
    ram_gb: 8,
    storage_gb: 32,
    accelerators: ["coral_tpu"],
  },
  {
    id: "mac_mini_m2",
    name: "Mac Mini M2",
    platform: "mac",
    cpu_arch: "arm64",
    ram_gb: 8,
    storage_gb: 256,
    accelerators: ["metal"],
  },
  {
    id: "jetson_nano",
    name: "NVIDIA Jetson Nano",
    platform: "jetson",
    cpu_arch: "arm64",
    ram_gb: 4,
    storage_gb: 16,
    accelerators: ["cuda"],
  },
];

const DIMENSION_INFO = {
  knowledge_maturity: {
    label: "Knowledge Maturity",
    icon: Brain,
    color: "text-purple-600",
    bgColor: "bg-purple-100",
    description: "Experience and pattern recognition capabilities",
  },
  goal_achievement: {
    label: "Goal Achievement",
    icon: Target,
    color: "text-green-600",
    bgColor: "bg-green-100",
    description: "Success rate and complex goal completion",
  },
  model_stability: {
    label: "Model Stability",
    icon: TrendingUp,
    color: "text-blue-600",
    bgColor: "bg-blue-100",
    description: "GNN convergence and stability",
  },
  collaboration: {
    label: "Collaboration",
    icon: Users,
    color: "text-orange-600",
    bgColor: "bg-orange-100",
    description: "Interaction success and knowledge sharing",
  },
  resource_management: {
    label: "Resource Management",
    icon: Battery,
    color: "text-yellow-600",
    bgColor: "bg-yellow-100",
    description: "Efficiency and sustainability",
  },
};

export function ReadinessPanel({ agentId, className }: ReadinessPanelProps) {
  const [readinessScore, setReadinessScore] = useState<IReadinessScore | null>(
    null,
  );
  const [selectedTarget, setSelectedTarget] =
    useState<string>("raspberry_pi_4b");
  const [isLoading, setIsLoading] = useState(true);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [showCelebration, setShowCelebration] = useState(false);

  const fetchReadinessScore = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/agents/${agentId}/readiness`);
      if (response.ok) {
        const data = await response.json();
        setReadinessScore(data);

        // Show celebration if agent just became ready
        if (data.is_ready && !readinessScore?.is_ready) {
          setShowCelebration(true);
          setTimeout(() => setShowCelebration(false), 5000);
        }
      }
    } catch (error) {
      console.error("Failed to fetch readiness score:", error);
    } finally {
      setIsLoading(false);
    }
  }, [agentId, readinessScore?.is_ready]);

  useEffect(() => {
    fetchReadinessScore();
  }, [agentId, fetchReadinessScore]);

  const handleEvaluate = async () => {
    setIsEvaluating(true);
    try {
      const response = await fetch(`/api/agents/${agentId}/evaluate`, {
        method: "POST",
      });
      if (response.ok) {
        const data = await response.json();
        setReadinessScore(data);

        if (data.is_ready && !readinessScore?.is_ready) {
          setShowCelebration(true);
          setTimeout(() => setShowCelebration(false), 5000);
        }
      }
    } catch (error) {
      console.error("Failed to evaluate agent:", error);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleExport = async () => {
    if (!readinessScore?.is_ready) return;

    setIsExporting(true);
    try {
      const response = await fetch(`/api/agents/${agentId}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: selectedTarget }),
      });

      if (response.ok) {
        // Download the export package
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${agentId}_${selectedTarget}_export.tar.gz`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Failed to export agent:", error);
    } finally {
      setIsExporting(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "text-green-600";
    if (score >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const getScoreIcon = (score: number) => {
    if (score >= 0.8) return CheckCircle2;
    if (score >= 0.6) return AlertCircle;
    return XCircle;
  };

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading readiness data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Celebration Animation */}
      {showCelebration && (
        <div className="fixed inset-0 pointer-events-none z-50 flex items-center justify-center">
          <div className="animate-bounce">
            <Trophy className="h-32 w-32 text-yellow-500" />
            <Sparkles className="h-16 w-16 text-yellow-400 absolute -top-4 -right-4 animate-pulse" />
            <Sparkles className="h-12 w-12 text-yellow-400 absolute -bottom-2 -left-2 animate-pulse delay-150" />
          </div>
        </div>
      )}

      {/* Overall Status Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                Agent Readiness Evaluation
                {readinessScore?.is_ready && (
                  <Rocket className="h-5 w-5 text-green-600" />
                )}
              </CardTitle>
              <CardDescription>
                Comprehensive evaluation across 5 key dimensions
              </CardDescription>
            </div>
            <Button
              onClick={handleEvaluate}
              disabled={isEvaluating}
              variant="outline"
            >
              {isEvaluating ? "Evaluating..." : "Re-evaluate"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {readinessScore && (
            <div className="space-y-4">
              {/* Overall Score */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div
                    className={cn(
                      "rounded-full p-3",
                      readinessScore.is_ready ? "bg-green-100" : "bg-red-100",
                    )}
                  >
                    {React.createElement(
                      readinessScore.is_ready ? CheckCircle2 : XCircle,
                      {
                        className: cn(
                          "h-8 w-8",
                          readinessScore.is_ready
                            ? "text-green-600"
                            : "text-red-600",
                        ),
                      },
                    )}
                  </div>
                  <div>
                    <p className="text-2xl font-bold">
                      {(readinessScore.scores.overall * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Overall Readiness
                    </p>
                  </div>
                </div>
                <Badge
                  variant={readinessScore.is_ready ? "default" : "secondary"}
                  className={cn(
                    "text-lg px-4 py-2",
                    readinessScore.is_ready && "bg-green-600",
                  )}
                >
                  {readinessScore.is_ready
                    ? "READY FOR DEPLOYMENT"
                    : "NOT READY"}
                </Badge>
              </div>

              {/* Progress Bar */}
              <Progress
                value={readinessScore.scores.overall * 100}
                className="h-3"
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Detailed Scores */}
      <Tabs defaultValue="scores" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="scores">Dimension Scores</TabsTrigger>
          <TabsTrigger value="metrics">Detailed Metrics</TabsTrigger>
          <TabsTrigger value="export">Export Options</TabsTrigger>
        </TabsList>

        <TabsContent value="scores" className="space-y-4">
          {readinessScore &&
            Object.entries(DIMENSION_INFO).map(([key, info]) => {
              const score =
                readinessScore.scores[
                  key as keyof typeof readinessScore.scores
                ];
              const Icon = info.icon;
              const ScoreIcon = getScoreIcon(score);

              return (
                <Card key={key}>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className={cn("rounded-lg p-2", info.bgColor)}>
                          <Icon className={cn("h-5 w-5", info.color)} />
                        </div>
                        <div>
                          <p className="font-semibold">{info.label}</p>
                          <p className="text-sm text-muted-foreground">
                            {info.description}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <ScoreIcon
                          className={cn("h-5 w-5", getScoreColor(score))}
                        />
                        <span
                          className={cn(
                            "text-lg font-bold",
                            getScoreColor(score),
                          )}
                        >
                          {(score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <Progress value={score * 100} className="h-2" />
                  </CardContent>
                </Card>
              );
            })}
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          {readinessScore?.metrics && (
            <>
              {/* Knowledge Metrics */}
              {readinessScore.metrics.knowledge && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Knowledge Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-2xl font-bold">
                          {readinessScore.metrics.knowledge.experience_count}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Experiences
                        </p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold">
                          {readinessScore.metrics.knowledge.pattern_count}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Patterns
                        </p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold">
                          {(
                            readinessScore.metrics.knowledge
                              .avg_pattern_confidence * 100
                          ).toFixed(1)}
                          %
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Avg Confidence
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Goal Metrics */}
              {readinessScore.metrics.goals && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Goal Achievement</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-2xl font-bold">
                          {readinessScore.metrics.goals.successful}/
                          {readinessScore.metrics.goals.total_attempted}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Goals Completed
                        </p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold">
                          {(
                            readinessScore.metrics.goals.success_rate * 100
                          ).toFixed(1)}
                          %
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Success Rate
                        </p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold">
                          {readinessScore.metrics.goals.complex_completed}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Complex Goals
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="export" className="space-y-4">
          {readinessScore?.is_ready ? (
            <>
              <Alert>
                <Rocket className="h-4 w-4" />
                <AlertTitle>Ready for Deployment!</AlertTitle>
                <AlertDescription>
                  Your agent has met all readiness criteria and can be exported
                  to hardware.
                </AlertDescription>
              </Alert>

              <Card>
                <CardHeader>
                  <CardTitle>Select Target Hardware</CardTitle>
                  <CardDescription>
                    Choose the hardware platform for deployment
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Select
                    value={selectedTarget}
                    onValueChange={setSelectedTarget}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {HARDWARE_TARGETS.map((target) => (
                        <SelectItem key={target.id} value={target.id}>
                          <div className="flex items-center gap-2">
                            <Cpu className="h-4 w-4" />
                            <span>{target.name}</span>
                            <span className="text-sm text-muted-foreground">
                              ({target.ram_gb}GB RAM, {target.cpu_arch})
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  <div className="rounded-lg border p-4">
                    {(() => {
                      const target = HARDWARE_TARGETS.find(
                        (t) => t.id === selectedTarget,
                      );
                      return target ? (
                        <div className="space-y-2">
                          <p className="font-semibold">{target.name}</p>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <p>Platform: {target.platform}</p>
                            <p>Architecture: {target.cpu_arch}</p>
                            <p>RAM: {target.ram_gb}GB</p>
                            <p>Storage: {target.storage_gb}GB</p>
                          </div>
                          {target.accelerators.length > 0 && (
                            <p className="text-sm">
                              Accelerators: {target.accelerators.join(", ")}
                            </p>
                          )}
                        </div>
                      ) : null;
                    })()}
                  </div>

                  <Button
                    onClick={handleExport}
                    disabled={isExporting}
                    className="w-full"
                    size="lg"
                  >
                    {isExporting ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Preparing Export Package...
                      </>
                    ) : (
                      <>
                        <Download className="mr-2 h-4 w-4" />
                        Export Agent Package
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </>
          ) : (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Not Ready for Export</AlertTitle>
              <AlertDescription>
                Your agent needs to meet all readiness criteria before it can be
                exported. Review the recommendations below to improve readiness.
              </AlertDescription>
            </Alert>
          )}

          {/* Recommendations */}
          {readinessScore && readinessScore.recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recommendations</CardTitle>
                <CardDescription>
                  Actions to improve agent readiness
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[200px]">
                  <ul className="space-y-2">
                    {readinessScore.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <AlertCircle className="h-4 w-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
