"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Alert, AlertDescription } from './ui/alert';
import { Separator } from './ui/separator';

/**
 * Markov Blanket Radar Chart Visualization Component
 * 
 * Interactive D3.js visualization of Markov Blanket dimensions with real-time
 * boundary monitoring, violation alerts, and agent position tracking.
 * 
 * Implements Task 53.2 requirements for radar chart visualization of:
 * - Internal states (μ): Agent's internal beliefs and hidden states
 * - Sensory states (s): Observations from the environment  
 * - Active states (a): Actions the agent can perform
 * - External states (η): Environment states beyond the agent's influence
 */

interface MarkovBlanketDimensions {
  internal_states: number[];
  sensory_states: number[];
  active_states: number[];
  external_states: number[];
  internal_dimension: number;
  sensory_dimension: number;
  active_dimension: number;
  external_dimension: number;
}

export interface BoundaryViolationEvent {
  event_id: string;
  agent_id: string;
  violation_type: string;
  timestamp: string;
  severity: number;
  independence_measure: number;
  threshold_violated: number;
  free_energy: number;
  expected_free_energy: number;
  kl_divergence: number;
  acknowledged: boolean;
  mitigated: boolean;
}

interface BoundaryMetrics {
  free_energy: number;
  expected_free_energy: number;
  kl_divergence: number;
  boundary_integrity: number;
  conditional_independence: number;
  stability_over_time: number;
  violation_count: number;
  last_violation_time?: string;
}

interface AgentPosition {
  agent_id: string;
  position: {
    internal: number;
    sensory: number;
    active: number;
    external: number;
  };
  boundary_distance: number;
  is_within_boundary: boolean;
}

interface MarkovBlanketVisualizationProps {
  agentId: string;
  dimensions: MarkovBlanketDimensions;
  metrics: BoundaryMetrics;
  violations: BoundaryViolationEvent[];
  agentPosition: AgentPosition;
  boundaryThresholds: {
    internal: number;
    sensory: number;
    active: number;
    external: number;
  };
  realTimeUpdates?: boolean;
  showViolations?: boolean;
  showMetrics?: boolean;
  onViolationAcknowledge?: (violationId: string) => void;
  onBoundaryThresholdChange?: (dimension: string, value: number) => void;
}

export const MarkovBlanketVisualization: React.FC<
  MarkovBlanketVisualizationProps
> = ({
  agentId,
  dimensions,
  metrics,
  violations,
  agentPosition,
  boundaryThresholds,
  realTimeUpdates = true,
  showViolations = true,
  showMetrics = true,
  onViolationAcknowledge,
  onBoundaryThresholdChange
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedDimension, setSelectedDimension] = useState<string | null>(
    null
  );
  const [alertThreshold, setAlertThreshold] = useState([0.8]);
  const [showAgentTrail, setShowAgentTrail] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState([1]);
  const [zoomLevel, setZoomLevel] = useState([1]);
  const [agentTrail, setAgentTrail] = useState<AgentPosition[]>([]);

  // Add current position to trail
  useEffect(() => {
    if (showAgentTrail) {
      setAgentTrail(prev => [...prev.slice(-20), agentPosition]);
    }
  }, [agentPosition, showAgentTrail]);

  // D3 radar chart visualization
  useEffect(() => {
    if (!svgRef.current || !dimensions) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 600;
    const height = 600;
    const margin = { top: 40, right: 40, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const radius = Math.min(innerWidth, innerHeight) / 2;
    const center = { x: width / 2, y: height / 2 };

    // Create main group
    const g = svg.append("g")
      .attr("transform", `translate(${center.x},${center.y})`);

    // Define the four dimensions for the radar chart
    const radarDimensions = [
      { 
        name: "Internal States", 
        key: "internal",
        value: dimensions.internal_dimension,
        threshold: boundaryThresholds.internal,
        color: "#3b82f6",
        angle: 0
      },
      { 
        name: "Sensory States", 
        key: "sensory",
        value: dimensions.sensory_dimension,
        threshold: boundaryThresholds.sensory,
        color: "#10b981",
        angle: Math.PI / 2
      },
      { 
        name: "Active States", 
        key: "active",
        value: dimensions.active_dimension,
        threshold: boundaryThresholds.active,
        color: "#f59e0b",
        angle: Math.PI
      },
      { 
        name: "External States", 
        key: "external",
        value: dimensions.external_dimension,
        threshold: boundaryThresholds.external,
        color: "#ef4444",
        angle: 3 * Math.PI / 2
      }
    ];

    // Create scales
    const maxValue = Math.max(
      ...radarDimensions.map(d => Math.max(d.value, d.threshold)),
      10
    );
    const radiusScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([0, radius * 0.8]);

    // Draw concentric circles (grid)
    const gridLevels = 5;
    for (let i = 1; i <= gridLevels; i++) {
      const gridRadius = (radius * 0.8 * i) / gridLevels;
      g.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", gridRadius)
        .attr("fill", "none")
        .attr("stroke", "#e5e7eb")
        .attr("stroke-width", 1)
        .attr("opacity", 0.5);
      
      // Add grid labels
      g.append("text")
        .attr("x", 5)
        .attr("y", -gridRadius)
        .attr("text-anchor", "start")
        .attr("font-size", "10px")
        .attr("fill", "#6b7280")
        .text((maxValue * i / gridLevels).toFixed(1));
    }

    // Draw axis lines and labels
    radarDimensions.forEach(dimension => {
      const x = Math.cos(dimension.angle - Math.PI / 2) * radius * 0.9;
      const y = Math.sin(dimension.angle - Math.PI / 2) * radius * 0.9;
      
      // Axis line
      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", x)
        .attr("y2", y)
        .attr("stroke", "#9ca3af")
        .attr("stroke-width", 2);
      
      // Axis label
      const labelX = Math.cos(dimension.angle - Math.PI / 2) * radius * 1.1;
      const labelY = Math.sin(dimension.angle - Math.PI / 2) * radius * 1.1;
      
      g.append("text")
        .attr("x", labelX)
        .attr("y", labelY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "12px")
        .attr("font-weight", "bold")
        .attr("fill", dimension.color)
        .text(dimension.name);
    });

    // Draw boundary thresholds
    const boundaryPath = d3.line<any>()
      .x(d => Math.cos(d.angle - Math.PI / 2) * radiusScale(d.threshold))
      .y(d => Math.sin(d.angle - Math.PI / 2) * radiusScale(d.threshold))
      .curve(d3.curveLinearClosed);

    g.append("path")
      .datum(radarDimensions)
      .attr("d", boundaryPath)
      .attr("fill", "rgba(239, 68, 68, 0.1)")
      .attr("stroke", "#ef4444")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    // Draw current dimensions
    const currentPath = d3.line<any>()
      .x(d => Math.cos(d.angle - Math.PI / 2) * radiusScale(d.value))
      .y(d => Math.sin(d.angle - Math.PI / 2) * radiusScale(d.value))
      .curve(d3.curveLinearClosed);

    g.append("path")
      .datum(radarDimensions)
      .attr("d", currentPath)
      .attr("fill", "rgba(59, 130, 246, 0.3)")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 3);

    // Draw agent position
    if (agentPosition) {
      const agentData = [
        { ...radarDimensions[0], value: agentPosition.position.internal },
        { ...radarDimensions[1], value: agentPosition.position.sensory },
        { ...radarDimensions[2], value: agentPosition.position.active },
        { ...radarDimensions[3], value: agentPosition.position.external }
      ];

      // Agent position points
      agentData.forEach(d => {
        const x = Math.cos(d.angle - Math.PI / 2) * radiusScale(d.value);
        const y = Math.sin(d.angle - Math.PI / 2) * radiusScale(d.value);
        
        g.append("circle")
          .attr("cx", x)
          .attr("cy", y)
          .attr("r", 6)
          .attr("fill", agentPosition.is_within_boundary ? "#10b981" : "#ef4444")
          .attr("stroke", "#ffffff")
          .attr("stroke-width", 2)
          .style("cursor", "pointer")
          .on("click", () => setSelectedDimension(d.key));
      });

      // Agent trail
      if (showAgentTrail && agentTrail.length > 1) {
        const trailPath = d3.line<AgentPosition>()
          .x(d => {
            const avgX = (
              Math.cos(-Math.PI / 2) * radiusScale(d.position.internal) +
              Math.cos(Math.PI / 2 - Math.PI / 2) * radiusScale(d.position.sensory) +
              Math.cos(Math.PI - Math.PI / 2) * radiusScale(d.position.active) +
              Math.cos(3 * Math.PI / 2 - Math.PI / 2) * radiusScale(d.position.external)
            ) / 4;
            return avgX;
          })
          .y(d => {
            const avgY = (
              Math.sin(-Math.PI / 2) * radiusScale(d.position.internal) +
              Math.sin(Math.PI / 2 - Math.PI / 2) * radiusScale(d.position.sensory) +
              Math.sin(Math.PI - Math.PI / 2) * radiusScale(d.position.active) +
              Math.sin(3 * Math.PI / 2 - Math.PI / 2) * radiusScale(d.position.external)
            ) / 4;
            return avgY;
          })
          .curve(d3.curveCardinal);

        g.append("path")
          .datum(agentTrail)
          .attr("d", trailPath)
          .attr("fill", "none")
          .attr("stroke", "#8b5cf6")
          .attr("stroke-width", 2)
          .attr("opacity", 0.6)
          .attr("stroke-dasharray", "3,3");
      }
    }

    // Add violation indicators
    if (showViolations && violations.length > 0) {
      const recentViolations = violations.filter(v => !v.acknowledged)
        .slice(-5);
      
      recentViolations.forEach((violation, index) => {
        const angle = (index / recentViolations.length) * 2 * Math.PI;
        const x = Math.cos(angle) * radius * 0.95;
        const y = Math.sin(angle) * radius * 0.95;
        
        g.append("circle")
          .attr("cx", x)
          .attr("cy", y)
          .attr("r", 8)
          .attr("fill", "#ef4444")
          .attr("stroke", "#ffffff")
          .attr("stroke-width", 2)
          .style("cursor", "pointer")
          .append("title")
          .text(`Violation: ${violation.violation_type}\nSeverity: ${violation.severity.toFixed(2)}`);
      });
    }

    // Add interaction handlers
    g.selectAll("circle")
      .on("mouseover", function(event, d) {
        d3.select(this).attr("r", 8);
      })
      .on("mouseout", function(event, d) {
        d3.select(this).attr("r", 6);
      });

  }, [dimensions, metrics, violations, agentPosition, boundaryThresholds, 
      showViolations, showAgentTrail, agentTrail, selectedDimension]);

  // Handle threshold changes
  const handleThresholdChange = useCallback((dimension: string, value: number) => {
    if (onBoundaryThresholdChange) {
      onBoundaryThresholdChange(dimension, value);
    }
  }, [onBoundaryThresholdChange]);

  // Handle violation acknowledgment
  const handleViolationAcknowledge = useCallback((violationId: string) => {
    if (onViolationAcknowledge) {
      onViolationAcknowledge(violationId);
    }
  }, [onViolationAcknowledge]);

  const unacknowledgedViolations = violations.filter(v => !v.acknowledged);
  const criticalViolations = violations.filter(v => v.severity >= alertThreshold[0]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Markov Blanket Visualization</h3>
          <p className="text-sm text-muted-foreground">
            Agent {agentId} - Real-time boundary monitoring
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge 
            variant={agentPosition?.is_within_boundary ? "default" : "destructive"}
          >
            {agentPosition?.is_within_boundary ? "Within Boundary" : "Boundary Violation"}
          </Badge>
          {realTimeUpdates && (
            <Badge variant="outline">Live</Badge>
          )}
        </div>
      </div>

      {/* Alerts */}
      {criticalViolations.length > 0 && (
        <Alert className="border-red-200 bg-red-50">
          <AlertDescription>
            {criticalViolations.length} critical boundary violation{criticalViolations.length > 1 ? 's' : ''} detected
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Radar Chart */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Boundary Radar Chart</CardTitle>
            </CardHeader>
            <CardContent>
              <svg
                ref={svgRef}
                width="600"
                height="600"
                className="w-full h-auto"
              />
            </CardContent>
          </Card>
        </div>

        {/* Controls and Metrics */}
        <div className="space-y-4">
          {/* Visualization Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Alert Threshold</Label>
                <Slider
                  value={alertThreshold}
                  onValueChange={setAlertThreshold}
                  max={1}
                  min={0}
                  step={0.1}
                  className="w-full"
                />
                <div className="text-xs text-muted-foreground">
                  {alertThreshold[0].toFixed(1)}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Animation Speed</Label>
                <Slider
                  value={animationSpeed}
                  onValueChange={setAnimationSpeed}
                  max={3}
                  min={0.1}
                  step={0.1}
                  className="w-full"
                />
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  checked={showAgentTrail}
                  onCheckedChange={setShowAgentTrail}
                />
                <Label>Show Agent Trail</Label>
              </div>
            </CardContent>
          </Card>

          {/* Boundary Metrics */}
          {showMetrics && (
            <Card>
              <CardHeader>
                <CardTitle>Boundary Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Free Energy:</div>
                  <div className="font-mono">{metrics.free_energy.toFixed(3)}</div>
                  
                  <div>Boundary Integrity:</div>
                  <div className="font-mono">
                    {(metrics.boundary_integrity * 100).toFixed(1)}%
                  </div>
                  
                  <div>KL Divergence:</div>
                  <div className="font-mono">{metrics.kl_divergence.toFixed(3)}</div>
                  
                  <div>Violations:</div>
                  <div className="font-mono">{metrics.violation_count}</div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Recent Violations */}
          {showViolations && unacknowledgedViolations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recent Violations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {unacknowledgedViolations.slice(0, 3).map((violation) => (
                    <div key={violation.event_id} className="p-2 bg-red-50 rounded border">
                      <div className="flex items-center justify-between">
                        <div className="text-sm">
                          <div className="font-medium">{violation.violation_type}</div>
                          <div className="text-xs text-muted-foreground">
                            Severity: {violation.severity.toFixed(2)}
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleViolationAcknowledge(violation.event_id)}
                        >
                          Acknowledge
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default MarkovBlanketVisualization; 