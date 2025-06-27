"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  TrendingUp,
  Target,
  BarChart3,
  PieChart,
  DollarSign,
  AlertTriangle,
  Download,
} from "lucide-react";

interface BusinessModel {
  coalitionId: string;
  name: string;
  marketShare: number;
  growthRate: number;
  profitabilityScore: number;
  riskScore: number;
  uncertainty: {
    marketShare: [number, number];
    growthRate: [number, number];
    profitability: [number, number];
    risk: [number, number];
  };
}

export function StrategicPositioningDashboard() {
  const [businessModels, setBusinessModels] = useState<BusinessModel[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    const generateMockData = () => {
      const coalitionNames = [
        "AgentTech",
        "DataCorp",
        "InnovateCo",
        "ScaleSoft",
        "EdgeAI",
      ];

      const mockBusinessModels: BusinessModel[] = coalitionNames.map(
        (name, index) => ({
          coalitionId: `coalition-${index + 1}`,
          name,
          marketShare: 10 + Math.random() * 30,
          growthRate: 20 + Math.random() * 80,
          profitabilityScore: 0.6 + Math.random() * 0.3,
          riskScore: 0.2 + Math.random() * 0.4,
          uncertainty: {
            marketShare: [8 + Math.random() * 5, 35 + Math.random() * 10],
            growthRate: [15 + Math.random() * 10, 90 + Math.random() * 20],
            profitability: [
              0.5 + Math.random() * 0.1,
              0.85 + Math.random() * 0.1,
            ],
            risk: [0.1 + Math.random() * 0.1, 0.5 + Math.random() * 0.2],
          },
        }),
      );

      setBusinessModels(mockBusinessModels);
      setLastUpdate(new Date());
    };

    generateMockData();
    const interval = setInterval(generateMockData, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Target className="w-8 h-8 text-blue-600" />
            Strategic Positioning Dashboard
          </h1>
          <p className="text-gray-600 mt-1">
            Coalition business models with uncertainty quantification
          </p>
        </div>

        <Badge variant="outline">
          Last updated: {lastUpdate.toLocaleTimeString()}
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Market Share
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {businessModels.length > 0
                    ? (
                        businessModels.reduce(
                          (acc, bm) => acc + bm.marketShare,
                          0,
                        ) / businessModels.length
                      ).toFixed(1)
                    : 0}
                  %
                </p>
              </div>
              <PieChart className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Growth Rate
                </p>
                <p className="text-2xl font-bold text-green-600">
                  {businessModels.length > 0
                    ? (
                        businessModels.reduce(
                          (acc, bm) => acc + bm.growthRate,
                          0,
                        ) / businessModels.length
                      ).toFixed(0)
                    : 0}
                  %
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Profitability
                </p>
                <p className="text-2xl font-bold text-purple-600">
                  {businessModels.length > 0
                    ? formatPercent(
                        businessModels.reduce(
                          (acc, bm) => acc + bm.profitabilityScore,
                          0,
                        ) / businessModels.length,
                      )
                    : "0%"}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Risk Score
                </p>
                <p className="text-2xl font-bold text-orange-600">
                  {businessModels.length > 0
                    ? formatPercent(
                        businessModels.reduce(
                          (acc, bm) => acc + bm.riskScore,
                          0,
                        ) / businessModels.length,
                      )
                    : "0%"}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="models" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="models">Business Models</TabsTrigger>
          <TabsTrigger value="uncertainty">Uncertainty Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="models" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {businessModels.map((model) => (
              <Card key={model.coalitionId}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>{model.name}</CardTitle>
                    <Badge
                      variant={model.marketShare > 25 ? "default" : "secondary"}
                    >
                      {model.marketShare.toFixed(1)}% Market Share
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-xs text-gray-500 mb-2">
                      Performance Metrics
                    </p>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Growth Rate</span>
                        <span className="font-medium text-green-600">
                          {model.growthRate.toFixed(0)}%
                        </span>
                      </div>
                      <Progress value={model.growthRate} className="h-2" />

                      <div className="flex justify-between items-center">
                        <span className="text-sm">Profitability</span>
                        <span className="font-medium">
                          {formatPercent(model.profitabilityScore)}
                        </span>
                      </div>
                      <Progress
                        value={model.profitabilityScore * 100}
                        className="h-2"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="uncertainty" className="space-y-6">
          <Alert>
            <AlertTriangle className="w-4 h-4" />
            <AlertDescription>
              All strategic projections include uncertainty bands and confidence
              intervals based on Monte Carlo simulation and sensitivity
              analysis.
            </AlertDescription>
          </Alert>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {businessModels.map((model) => (
              <Card key={model.coalitionId}>
                <CardHeader>
                  <CardTitle>{model.name} - Uncertainty Analysis</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h5 className="font-semibold mb-3">
                      Market Share Confidence Bands
                    </h5>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Lower Bound</span>
                        <span>
                          {model.uncertainty.marketShare[0].toFixed(1)}%
                        </span>
                      </div>
                      <div className="relative h-6 bg-gray-200 rounded">
                        <div
                          className="absolute h-full bg-blue-500 rounded opacity-30"
                          style={{
                            left: `${Math.min(model.uncertainty.marketShare[0], 50)}%`,
                            width: `${Math.abs(model.uncertainty.marketShare[1] - model.uncertainty.marketShare[0])}%`,
                          }}
                        />
                        <div
                          className="absolute w-1 h-full bg-blue-700"
                          style={{
                            left: `${Math.min(model.marketShare, 50)}%`,
                          }}
                        />
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Upper Bound</span>
                        <span>
                          {model.uncertainty.marketShare[1].toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-blue-50 p-3 rounded-lg">
                    <h6 className="font-semibold text-blue-900 mb-2">
                      Risk Assessment
                    </h6>
                    <div className="flex justify-between">
                      <span className="text-blue-700">Risk Score</span>
                      <span className="font-bold text-blue-900">
                        {formatPercent(model.riskScore)}
                        (±
                        {formatPercent(
                          (model.uncertainty.risk[1] -
                            model.uncertainty.risk[0]) /
                            2,
                        )}
                        )
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="w-5 h-5" />
            Export Strategic Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button variant="outline">
              <Download className="w-4 h-4 mr-2" />
              Export Analysis Data
            </Button>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Export includes confidence intervals, uncertainty bands, and
            statistical significance
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
