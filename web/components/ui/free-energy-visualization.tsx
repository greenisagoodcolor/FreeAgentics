"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Zap, 
  Play, 
  Pause, 
  RotateCcw, 
  TrendingUp, 
  TrendingDown,
  Activity,
  Target
} from "lucide-react";

// Types for Free Energy data
export interface FreeEnergyData {
  timestamp: number;
  variationalFreeEnergy: number;    // F = E_q[ln q(s) - ln p(s, o)]
  accuracy: number;                 // -E_q[ln p(o|s)] (negative log likelihood)
  complexity: number;               // D_KL[q(s)||p(s)] (KL divergence)
  expectedFreeEnergy: number;       // G(π) = E_q[F(o, s)] for policy π
  precision: {
    sensory: number;                // γ - sensory precision
    policy: number;                 // β - policy precision
    state: number;                  // α - state precision
  };
  prediction: {
    error: number;                  // Prediction error magnitude
    confidence: number;             // Inverse prediction error
  };
}

export interface FreeEnergyHistory {
  data: FreeEnergyData[];
  maxLength: number;
}

interface FreeEnergyVisualizationProps {
  agentId: string;
  width?: number;
  height?: number;
  updateInterval?: number;
  timeWindow?: number;
  className?: string;
  onFreeEnergyChange?: (feData: FreeEnergyData) => void;
  isRealTime?: boolean;
}

export function FreeEnergyVisualization({
  agentId,
  width = 800,
  height = 400,
  updateInterval = 1000,
  timeWindow = 60000, // 60 seconds
  className,
  onFreeEnergyChange,
  isRealTime = true,
}: FreeEnergyVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(isRealTime);
  const [currentData, setCurrentData] = useState<FreeEnergyData | null>(null);
  const [freeEnergyHistory, setFreeEnergyHistory] = useState<FreeEnergyHistory>({
    data: [],
    maxLength: 200,
  });
  const [dimensions, setDimensions] = useState({ width, height });
  const [activeTab, setActiveTab] = useState("overview");

  // Responsive sizing
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: Math.max(400, rect.width - 32),
          height: Math.max(300, height),
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [height]);

  // Generate mock free energy data for demonstration
  const generateMockFreeEnergyData = useCallback((prevData?: FreeEnergyData): FreeEnergyData => {
    const now = Date.now();
    
    // Simulate dynamic free energy with temporal correlation
    const baseAccuracy = 2.5 + Math.sin(now / 10000) * 0.5 + (Math.random() - 0.5) * 0.3;
    const baseComplexity = 1.2 + Math.cos(now / 15000) * 0.3 + (Math.random() - 0.5) * 0.2;
    
    // Add some temporal smoothing if we have previous data
    const accuracy = prevData 
      ? prevData.accuracy * 0.8 + baseAccuracy * 0.2
      : baseAccuracy;
    const complexity = prevData 
      ? prevData.complexity * 0.8 + baseComplexity * 0.2
      : baseComplexity;
    
    const variationalFreeEnergy = accuracy + complexity;
    const expectedFreeEnergy = variationalFreeEnergy + Math.random() * 0.5;
    
    // Prediction error inversely related to accuracy
    const predictionError = Math.max(0.1, 2.0 - accuracy + (Math.random() - 0.5) * 0.4);
    const predictionConfidence = 1 / (1 + predictionError);
    
    // Precision parameters with some correlation to free energy
    const sensoryPrecision = 16 + (3 - variationalFreeEnergy) * 2 + Math.random() * 8;
    const policyPrecision = 12 + (3 - variationalFreeEnergy) * 1.5 + Math.random() * 6;
    const statePrecision = 2 + (3 - variationalFreeEnergy) * 0.5 + Math.random() * 2;
    
    return {
      timestamp: now,
      variationalFreeEnergy,
      accuracy,
      complexity,
      expectedFreeEnergy,
      precision: {
        sensory: Math.max(0.1, sensoryPrecision),
        policy: Math.max(0.1, policyPrecision),
        state: Math.max(0.1, statePrecision),
      },
      prediction: {
        error: predictionError,
        confidence: predictionConfidence,
      },
    };
  }, []);

  // Data fetching/updating
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isPlaying && isRealTime) {
      interval = setInterval(() => {
        const newData = generateMockFreeEnergyData(currentData || undefined);
        setCurrentData(newData);
        
        setFreeEnergyHistory(prev => {
          const cutoffTime = Date.now() - timeWindow;
          const filteredData = prev.data.filter(d => d.timestamp > cutoffTime);
          return {
            ...prev,
            data: [...filteredData.slice(-prev.maxLength + 1), newData],
          };
        });

        onFreeEnergyChange?.(newData);
      }, updateInterval);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, isRealTime, updateInterval, timeWindow, generateMockFreeEnergyData, currentData, onFreeEnergyChange]);

  // D3.js visualization rendering
  useEffect(() => {
    if (!svgRef.current || freeEnergyHistory.data.length === 0) return;

    const svg = d3.select(svgRef.current);
    const { width: w, height: h } = dimensions;
    
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 80, bottom: 60, left: 80 };
    const chartWidth = w - margin.left - margin.right;
    const chartHeight = h - margin.top - margin.bottom;

    const g = svg
      .attr("width", w)
      .attr("height", h)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Filter data for current time window
    const now = Date.now();
    const windowStart = now - timeWindow;
    const visibleData = freeEnergyHistory.data.filter(d => d.timestamp >= windowStart);

    if (visibleData.length === 0) return;

    // Create scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(visibleData, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, chartWidth]);

    const yDomain = d3.extent([
      ...visibleData.map(d => d.variationalFreeEnergy),
      ...visibleData.map(d => d.accuracy),
      ...visibleData.map(d => d.complexity),
      ...visibleData.map(d => d.expectedFreeEnergy),
    ]) as [number, number];
    
    const yScale = d3
      .scaleLinear()
      .domain([yDomain[0] * 0.9, yDomain[1] * 1.1])
      .range([chartHeight, 0]);

    // Line generators
    const lineVFE = d3.line<FreeEnergyData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.variationalFreeEnergy))
      .curve(d3.curveMonotoneX);

    const lineAccuracy = d3.line<FreeEnergyData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.accuracy))
      .curve(d3.curveMonotoneX);

    const lineComplexity = d3.line<FreeEnergyData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.complexity))
      .curve(d3.curveMonotoneX);

    const lineEFE = d3.line<FreeEnergyData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.expectedFreeEnergy))
      .curve(d3.curveMonotoneX);

    // Draw grid
    const xTicks = xScale.ticks(6);
    const yTicks = yScale.ticks(6);

    g.selectAll(".grid-x")
      .data(xTicks)
      .enter()
      .append("line")
      .attr("class", "grid-x")
      .attr("x1", d => xScale(d))
      .attr("x2", d => xScale(d))
      .attr("y1", 0)
      .attr("y2", chartHeight)
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 1)
      .attr("opacity", 0.5);

    g.selectAll(".grid-y")
      .data(yTicks)
      .enter()
      .append("line")
      .attr("class", "grid-y")
      .attr("x1", 0)
      .attr("x2", chartWidth)
      .attr("y1", d => yScale(d))
      .attr("y2", d => yScale(d))
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 1)
      .attr("opacity", 0.5);

    // Draw lines
    if (activeTab === "overview" || activeTab === "components") {
      // Variational Free Energy (main line)
      g.append("path")
        .datum(visibleData)
        .attr("fill", "none")
        .attr("stroke", "#dc2626")
        .attr("stroke-width", 3)
        .attr("d", lineVFE);

      // Accuracy component
      g.append("path")
        .datum(visibleData)
        .attr("fill", "none")
        .attr("stroke", "#2563eb")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .attr("d", lineAccuracy);

      // Complexity component
      g.append("path")
        .datum(visibleData)
        .attr("fill", "none")
        .attr("stroke", "#16a34a")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "3,3")
        .attr("d", lineComplexity);
    }

    if (activeTab === "overview" || activeTab === "expected") {
      // Expected Free Energy
      g.append("path")
        .datum(visibleData)
        .attr("fill", "none")
        .attr("stroke", "#7c3aed")
        .attr("stroke-width", 2)
        .attr("d", lineEFE);
    }

    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(6)
      .tickFormat(d3.timeFormat("%H:%M:%S"));
    
    g.append("g")
      .attr("transform", `translate(0,${chartHeight})`)
      .call(xAxis)
      .selectAll("text")
      .attr("font-size", "11px");

    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d3.format(".2f"));
    
    g.append("g")
      .call(yAxis)
      .selectAll("text")
      .attr("font-size", "11px");

    // Axis labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (chartHeight / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "600")
      .text("Free Energy (nats)");

    g.append("text")
      .attr("transform", `translate(${chartWidth / 2}, ${chartHeight + margin.bottom - 10})`)
      .style("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "600")
      .text("Time");

    // Current value indicators
    if (currentData) {
      const currentX = xScale(new Date(currentData.timestamp));
      
      // Current VFE point
      g.append("circle")
        .attr("cx", currentX)
        .attr("cy", yScale(currentData.variationalFreeEnergy))
        .attr("r", 4)
        .attr("fill", "#dc2626")
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 2);

      // Current value text
      g.append("text")
        .attr("x", chartWidth - 10)
        .attr("y", 15)
        .attr("text-anchor", "end")
        .attr("font-size", "12px")
        .attr("font-weight", "600")
        .attr("fill", "#dc2626")
        .text(`F = ${currentData.variationalFreeEnergy.toFixed(3)}`);
    }

    // Legend
    const legend = g.append("g")
      .attr("transform", `translate(${chartWidth - 150}, 30)`);

    const legendData = [
      { name: "VFE", color: "#dc2626", style: "solid" },
      { name: "Accuracy", color: "#2563eb", style: "dashed" },
      { name: "Complexity", color: "#16a34a", style: "dotted" },
      { name: "EFE", color: "#7c3aed", style: "solid" },
    ];

    legendData.forEach((item, i) => {
      const legendItem = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);

      legendItem.append("line")
        .attr("x1", 0)
        .attr("x2", 20)
        .attr("y1", 0)
        .attr("y2", 0)
        .attr("stroke", item.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", item.style === "dashed" ? "5,5" : item.style === "dotted" ? "3,3" : "none");

      legendItem.append("text")
        .attr("x", 25)
        .attr("y", 0)
        .attr("dy", "0.35em")
        .attr("font-size", "11px")
        .attr("fill", "#374151")
        .text(item.name);
    });

  }, [freeEnergyHistory, dimensions, currentData, activeTab, timeWindow]);

  // Control handlers
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setFreeEnergyHistory({ data: [], maxLength: 200 });
    setCurrentData(null);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            <CardTitle>Free Energy Dynamics</CardTitle>
            <Badge variant="outline">Agent {agentId}</Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePlayPause}
              disabled={!isRealTime}
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <CardDescription>
          Real-time Active Inference free energy minimization dynamics
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <div ref={containerRef} className="w-full">
          {/* Current Values Dashboard */}
          {currentData && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-red-600" />
                <div>
                  <div className="text-sm font-medium">Variational FE</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.variationalFreeEnergy.toFixed(3)} nats
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-blue-600" />
                <div>
                  <div className="text-sm font-medium">Accuracy</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.accuracy.toFixed(3)} nats
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-green-600" />
                <div>
                  <div className="text-sm font-medium">Complexity</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.complexity.toFixed(3)} nats
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-purple-600" />
                <div>
                  <div className="text-sm font-medium">Expected FE</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.expectedFreeEnergy.toFixed(3)} nats
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Visualization Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="components">Components</TabsTrigger>
              <TabsTrigger value="expected">Expected FE</TabsTrigger>
            </TabsList>
          </Tabs>

          {/* D3.js Visualization Container */}
          <svg ref={svgRef} className="w-full border rounded-lg" />
          
          {/* Mathematical Explanation */}
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="text-sm text-red-800">
              <p className="font-semibold mb-1">Free Energy Mathematics:</p>
              <p className="text-xs space-y-1">
                • <strong>F</strong>: Variational Free Energy = Accuracy + Complexity
                <br />
                • <strong>Accuracy</strong>: -E<sub>q</sub>[ln p(o|s)] (negative log likelihood)
                <br />
                • <strong>Complexity</strong>: D<sub>KL</sub>[q(s)||p(s)] (KL divergence prior)
                <br />
                • <strong>G(π)</strong>: Expected Free Energy for policy π (planning objective)
              </p>
            </div>
          </div>

          {/* Status Information */}
          <div className="flex items-center justify-between mt-4 text-sm text-muted-foreground">
            <span>
              {isRealTime ? "Real-time updates" : "Static display"} • 
              {freeEnergyHistory.data.length} data points • 
              {timeWindow / 1000}s window
            </span>
            <span>
              {currentData && `Last update: ${new Date(currentData.timestamp).toLocaleTimeString()}`}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 