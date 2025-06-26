"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  Brain, 
  Play, 
  Pause, 
  RotateCcw, 
  TrendingUp, 
  Info,
  Zap
} from "lucide-react";

// Types for Active Inference data
export interface BeliefStateData {
  timestamp: number;
  beliefs: number[];         // q(s) - belief distribution over states
  entropy: number;          // H[q(s)] - Shannon entropy
  confidence: number;       // 1 - normalized entropy
  mostLikelyState: number;  // argmax q(s)
  precision: {
    sensory: number;        // γ - sensory precision
    policy: number;         // β - policy precision
    state: number;          // α - state precision
  };
}

export interface BeliefHistory {
  data: BeliefStateData[];
  maxLength: number;
}

interface BeliefStateVisualizationProps {
  agentId: string;
  stateLabels?: string[];
  width?: number;
  height?: number;
  updateInterval?: number;
  className?: string;
  onBeliefChange?: (beliefData: BeliefStateData) => void;
  isRealTime?: boolean;
}

export function BeliefStateVisualization({
  agentId,
  stateLabels = [],
  width = 800,
  height = 400,
  updateInterval = 1000,
  className,
  onBeliefChange,
  isRealTime = true,
}: BeliefStateVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(isRealTime);
  const [currentData, setCurrentData] = useState<BeliefStateData | null>(null);
  const [beliefHistory, setBeliefHistory] = useState<BeliefHistory>({
    data: [],
    maxLength: 100,
  });
  const [dimensions, setDimensions] = useState({ width, height });

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

  // Generate mock belief data for demonstration
  const generateMockBeliefData = useCallback((): BeliefStateData => {
    const numStates = stateLabels.length || 8;
    const beliefs = new Array(numStates).fill(0).map(() => Math.random());
    const sum = beliefs.reduce((a, b) => a + b, 0);
    const normalizedBeliefs = beliefs.map(b => b / sum);
    
    // Calculate entropy: H[q(s)] = -Σ q(s) * log(q(s))
    const entropy = -normalizedBeliefs.reduce((h, q) => 
      h + (q > 0 ? q * Math.log(q) : 0), 0
    );
    const maxEntropy = Math.log(numStates);
    const confidence = 1 - (entropy / maxEntropy);
    
    const mostLikelyState = normalizedBeliefs.indexOf(Math.max(...normalizedBeliefs));
    
    return {
      timestamp: Date.now(),
      beliefs: normalizedBeliefs,
      entropy,
      confidence,
      mostLikelyState,
      precision: {
        sensory: 16 + Math.random() * 32,
        policy: 8 + Math.random() * 24,
        state: 1 + Math.random() * 4,
      },
    };
  }, [stateLabels.length]);

  // Data fetching/updating
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isPlaying && isRealTime) {
      interval = setInterval(() => {
        const newData = generateMockBeliefData();
        setCurrentData(newData);
        
        setBeliefHistory(prev => ({
          ...prev,
          data: [...prev.data.slice(-prev.maxLength + 1), newData],
        }));

        onBeliefChange?.(newData);
      }, updateInterval);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, isRealTime, updateInterval, generateMockBeliefData, onBeliefChange]);

  // D3.js visualization rendering
  useEffect(() => {
    if (!svgRef.current || !currentData) return;

    const svg = d3.select(svgRef.current);
    const { width: w, height: h } = dimensions;
    
    // Clear previous content
    svg.selectAll("*").remove();

    // Set up main group with margins
    const margin = { top: 20, right: 60, bottom: 80, left: 60 };
    const chartWidth = w - margin.left - margin.right;
    const chartHeight = h - margin.top - margin.bottom;

    const g = svg
      .attr("width", w)
      .attr("height", h)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const xScale = d3
      .scaleBand()
      .domain(currentData.beliefs.map((_, i) => i.toString()))
      .range([0, chartWidth])
      .padding(0.1);

    const yScale = d3
      .scaleLinear()
      .domain([0, Math.max(...currentData.beliefs) * 1.1])
      .range([chartHeight, 0]);

    // Color scale for belief states
    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([0, currentData.beliefs.length - 1]);

    // Create bars for belief distribution
    const bars = g
      .selectAll(".belief-bar")
      .data(currentData.beliefs)
      .enter()
      .append("rect")
      .attr("class", "belief-bar")
      .attr("x", (_, i) => xScale(i.toString()) || 0)
      .attr("y", d => yScale(d))
      .attr("width", xScale.bandwidth())
      .attr("height", d => chartHeight - yScale(d))
      .attr("fill", (_, i) => colorScale(i))
      .attr("opacity", 0.8)
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 1);

    // Highlight most likely state
    bars
      .filter((_, i) => i === currentData.mostLikelyState)
      .attr("stroke", "#ff6b35")
      .attr("stroke-width", 3)
      .attr("opacity", 1);

    // Add probability values as text
    g.selectAll(".prob-text")
      .data(currentData.beliefs)
      .enter()
      .append("text")
      .attr("class", "prob-text")
      .attr("x", (_, i) => (xScale(i.toString()) || 0) + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d) - 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .attr("font-weight", "500")
      .attr("fill", "#374151")
      .text(d => d.toFixed(3));

    // X-axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(i => stateLabels[parseInt(i)] || `S${i}`);
    
    g.append("g")
      .attr("transform", `translate(0,${chartHeight})`)
      .call(xAxis)
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .attr("font-size", "11px");

    // Y-axis
    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d3.format(".3f"));
    
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
      .text("Belief Probability q(s)");

    g.append("text")
      .attr("transform", `translate(${chartWidth / 2}, ${chartHeight + margin.bottom - 10})`)
      .style("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "600")
      .text("Hidden States");

    // Mathematical annotation
    g.append("text")
      .attr("x", chartWidth - 10)
      .attr("y", 15)
      .attr("text-anchor", "end")
      .attr("font-size", "12px")
      .attr("font-style", "italic")
      .attr("fill", "#6b7280")
      .text(`H[q(s)] = ${currentData.entropy.toFixed(3)} bits`);

    // Confidence indicator
    g.append("text")
      .attr("x", chartWidth - 10)
      .attr("y", 35)
      .attr("text-anchor", "end")
      .attr("font-size", "12px")
      .attr("font-style", "italic")
      .attr("fill", "#6b7280")
      .text(`Confidence = ${(currentData.confidence * 100).toFixed(1)}%`);

    // Add normalization validation
    const beliefSum = currentData.beliefs.reduce((sum, b) => sum + b, 0);
    const isNormalized = Math.abs(beliefSum - 1.0) < 1e-10;
    
    g.append("text")
      .attr("x", 10)
      .attr("y", 15)
      .attr("font-size", "11px")
      .attr("fill", isNormalized ? "#059669" : "#dc2626")
      .text(`Σq(s) = ${beliefSum.toFixed(6)} ${isNormalized ? "✓" : "⚠"}`);

  }, [currentData, dimensions, stateLabels]);

  // Control handlers
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setBeliefHistory({ data: [], maxLength: 100 });
    setCurrentData(null);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle>Belief State Distribution</CardTitle>
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
          Real-time visualization of q(s) - agent beliefs over hidden states
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <div ref={containerRef} className="w-full">
          {/* Mathematical Information Panel */}
          {currentData && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-blue-600" />
                <div>
                  <div className="text-sm font-medium">Entropy</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.entropy.toFixed(3)} bits
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-green-600" />
                <div>
                  <div className="text-sm font-medium">Confidence</div>
                  <div className="text-xs text-muted-foreground">
                    {(currentData.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-purple-600" />
                <div>
                  <div className="text-sm font-medium">Most Likely</div>
                  <div className="text-xs text-muted-foreground">
                    State {currentData.mostLikelyState}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-orange-600" />
                <div>
                  <div className="text-sm font-medium">Precision γ</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.precision.sensory.toFixed(1)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* D3.js Visualization Container */}
          <svg ref={svgRef} className="w-full border rounded-lg" />
          
          {/* Mathematical Explanation */}
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-sm text-blue-800">
              <p className="font-semibold mb-1">Active Inference Mathematics:</p>
              <p className="text-xs space-y-1">
                • <strong>q(s)</strong>: Belief distribution over hidden states (must sum to 1)
                <br />
                • <strong>H[q(s)]</strong>: Shannon entropy = -Σ q(s) log q(s)
                <br />
                • <strong>Confidence</strong>: 1 - H[q(s)]/log(|S|) (normalized uncertainty)
                <br />
                • <strong>γ</strong>: Sensory precision parameter controlling belief updates
              </p>
            </div>
          </div>

          {/* Status Information */}
          <div className="flex items-center justify-between mt-4 text-sm text-muted-foreground">
            <span>
              {isRealTime ? "Real-time updates" : "Static display"} • 
              {beliefHistory.data.length} data points
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