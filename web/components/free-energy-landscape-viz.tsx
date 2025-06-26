"use client";

import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';

/**
 * Free Energy Landscape Visualization Component
 * 
 * Interactive D3.js visualization of free energy surfaces with decision boundaries,
 * convergence tracking, and uncertainty regions for scientific publication quality.
 * 
 * Implements Task 37.3 requirements for mathematical rigor and real-time updates.
 */

interface FreeEnergyDataPoint {
  x: number;
  y: number;
  free_energy: number;
  belief_state: number[];
  uncertainty: number;
  decision_boundary: boolean;
  convergence_score: number;
}

interface DecisionBoundary {
  path: Array<[number, number]>;
  confidence: number;
  boundary_type: string;
}

interface ConvergencePoint {
  x: number;
  y: number;
  timestamp: string;
  convergence_value: number;
  stability: number;
}

interface FreeEnergyLandscapeProps {
  data: FreeEnergyDataPoint[];
  decisionBoundaries: DecisionBoundary[];
  convergencePoints: ConvergencePoint[];
  agentId: string;
  realTimeUpdates?: boolean;
  showUncertainty?: boolean;
  showConvergence?: boolean;
  mathematicalAnnotations?: boolean;
}

export const FreeEnergyLandscapeViz: React.FC<FreeEnergyLandscapeProps> = ({
  data,
  decisionBoundaries,
  convergencePoints,
  agentId,
  realTimeUpdates = true,
  showUncertainty = true,
  showConvergence = true,
  mathematicalAnnotations = true
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedPoint, setSelectedPoint] = useState<FreeEnergyDataPoint | null>(null);
  const [energyThreshold, setEnergyThreshold] = useState([0, 10]);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState([0, 1]);
  const [timeSlider, setTimeSlider] = useState(100); // Percentage through time series
  const [viewMode, setViewMode] = useState<'surface' | 'contour' | 'heatmap'>('surface');

  // D3 visualization setup
  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 80, bottom: 60, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create main group
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const xExtent = d3.extent(data, d => d.x) as [number, number];
    const yExtent = d3.extent(data, d => d.y) as [number, number];
    const energyExtent = d3.extent(data, d => d.free_energy) as [number, number];
    
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([innerHeight, 0]);
    
    const energyColorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain(energyExtent);
    
    const uncertaintyColorScale = d3.scaleSequential(d3.interpolateReds)
      .domain([0, 1]);

    // Filter data based on thresholds
    const filteredData = data.filter(d => 
      d.free_energy >= energyThreshold[0] && 
      d.free_energy <= energyThreshold[1] &&
      d.uncertainty >= uncertaintyThreshold[0] &&
      d.uncertainty <= uncertaintyThreshold[1]
    );

    // Render based on view mode
    if (viewMode === 'heatmap') {
      renderHeatmap(g, filteredData, xScale, yScale, energyColorScale);
    } else if (viewMode === 'contour') {
      renderContourPlot(g, filteredData, xScale, yScale, energyColorScale, innerWidth, innerHeight);
    } else {
      renderSurfacePlot(g, filteredData, xScale, yScale, energyColorScale);
    }

    // Render decision boundaries
    if (decisionBoundaries.length > 0) {
      renderDecisionBoundaries(g, decisionBoundaries, xScale, yScale);
    }

    // Render uncertainty regions
    if (showUncertainty) {
      renderUncertaintyRegions(g, filteredData, xScale, yScale, uncertaintyColorScale);
    }

    // Render convergence points
    if (showConvergence && convergencePoints.length > 0) {
      renderConvergencePoints(g, convergencePoints, xScale, yScale);
    }

    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d => (typeof d === 'number' ? d : d.valueOf()).toFixed(2));
    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => (typeof d === 'number' ? d : d.valueOf()).toFixed(2));

    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Add axis labels
    g.append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "middle")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + 40)
      .text("Belief State Dimension 1");

    g.append("text")
      .attr("class", "y-label")
      .attr("text-anchor", "middle")
      .attr("transform", `translate(-50,${innerHeight / 2})rotate(-90)`)
      .text("Belief State Dimension 2");

    // Add mathematical annotations
    if (mathematicalAnnotations) {
      addMathematicalAnnotations(g, innerWidth, innerHeight, energyExtent);
    }

    // Add color legend
    addColorLegend(svg, energyColorScale, energyExtent, width, height);

    // Add interaction handlers
    addInteractionHandlers(g, filteredData, xScale, yScale, setSelectedPoint);

  }, [data, decisionBoundaries, convergencePoints, energyThreshold, uncertaintyThreshold, 
      viewMode, showUncertainty, showConvergence, mathematicalAnnotations]);

  // Rendering functions
  const renderHeatmap = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    data: FreeEnergyDataPoint[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>,
    colorScale: d3.ScaleSequential<string>
  ) => {
    g.selectAll(".energy-point")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", "energy-point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 3)
      .attr("fill", d => colorScale(d.free_energy))
      .attr("opacity", 0.8)
      .style("cursor", "pointer");
  };

  const renderContourPlot = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    data: FreeEnergyDataPoint[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>,
    colorScale: d3.ScaleSequential<string>,
    width: number,
    height: number
  ) => {
    // Create contour generator
    const contourGenerator = d3.contours()
      .size([50, 50])
      .thresholds(10);

    // Convert data to grid for contouring
    const gridData = new Array(50 * 50);
    const xStep = (xScale.domain()[1] - xScale.domain()[0]) / 50;
    const yStep = (yScale.domain()[1] - yScale.domain()[0]) / 50;

    for (let i = 0; i < 50; i++) {
      for (let j = 0; j < 50; j++) {
        const x = xScale.domain()[0] + i * xStep;
        const y = yScale.domain()[0] + j * yStep;
        
        // Find nearest data point
        const nearest = data.reduce((prev, curr) => 
          Math.sqrt(Math.pow(curr.x - x, 2) + Math.pow(curr.y - y, 2)) < 
          Math.sqrt(Math.pow(prev.x - x, 2) + Math.pow(prev.y - y, 2)) ? curr : prev
        );
        
        gridData[i + j * 50] = nearest.free_energy;
      }
    }

    const contours = contourGenerator(gridData);

    g.selectAll(".contour")
      .data(contours)
      .enter()
      .append("path")
      .attr("class", "contour")
      .attr("d", d3.geoPath())
      .attr("fill", d => colorScale(d.value))
      .attr("fill-opacity", 0.3)
      .attr("stroke", d => colorScale(d.value))
      .attr("stroke-width", 1);
  };

  const renderSurfacePlot = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    data: FreeEnergyDataPoint[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>,
    colorScale: d3.ScaleSequential<string>
  ) => {
    // Create Voronoi diagram for smooth surface
    const voronoi = d3.Delaunay.from(data, d => xScale(d.x), d => yScale(d.y));
    const voronoiPolygons = voronoi.voronoi([0, 0, xScale.range()[1], yScale.range()[0]]);

    g.selectAll(".voronoi-cell")
      .data(data)
      .enter()
      .append("path")
      .attr("class", "voronoi-cell")
      .attr("d", (d, i) => voronoiPolygons.renderCell(i))
      .attr("fill", d => colorScale(d.free_energy))
      .attr("fill-opacity", 0.6)
      .attr("stroke", "white")
      .attr("stroke-width", 0.5);
  };

  const renderDecisionBoundaries = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    boundaries: DecisionBoundary[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>
  ) => {
    const line = d3.line<[number, number]>()
      .x(d => xScale(d[0]))
      .y(d => yScale(d[1]))
      .curve(d3.curveCardinal);

    g.selectAll(".decision-boundary")
      .data(boundaries)
      .enter()
      .append("path")
      .attr("class", "decision-boundary")
      .attr("d", d => line(d.path))
      .attr("fill", "none")
      .attr("stroke", "#ff6b6b")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5")
      .attr("opacity", d => d.confidence);
  };

  const renderUncertaintyRegions = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    data: FreeEnergyDataPoint[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>,
    uncertaintyColorScale: d3.ScaleSequential<string>
  ) => {
    g.selectAll(".uncertainty-region")
      .data(data.filter(d => d.uncertainty > 0.5))
      .enter()
      .append("circle")
      .attr("class", "uncertainty-region")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", d => 5 + d.uncertainty * 10)
      .attr("fill", "none")
      .attr("stroke", d => uncertaintyColorScale(d.uncertainty))
      .attr("stroke-width", 2)
      .attr("opacity", 0.6);
  };

  const renderConvergencePoints = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    points: ConvergencePoint[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>
  ) => {
    g.selectAll(".convergence-point")
      .data(points)
      .enter()
      .append("circle")
      .attr("class", "convergence-point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", d => 3 + d.convergence_value * 5)
      .attr("fill", "#4ecdc4")
      .attr("stroke", "#2d9cdb")
      .attr("stroke-width", 2)
      .attr("opacity", 0.8);
  };

  const addMathematicalAnnotations = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    width: number,
    height: number,
    energyExtent: [number, number]
  ) => {
    // Add free energy equation
    g.append("text")
      .attr("x", width - 200)
      .attr("y", 30)
      .attr("class", "math-annotation")
      .style("font-family", "KaTeX_Main")
      .style("font-size", "12px")
      .text("F = -log P(o) + KL[Q(s)||P(s)]");

    // Add energy range annotation
    g.append("text")
      .attr("x", width - 200)
      .attr("y", 50)
      .attr("class", "energy-range")
      .style("font-size", "10px")
      .text(`Energy range: [${energyExtent[0].toFixed(2)}, ${energyExtent[1].toFixed(2)}]`);
  };

  const addColorLegend = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    colorScale: d3.ScaleSequential<string>,
    energyExtent: [number, number],
    width: number,
    height: number
  ) => {
    const legendHeight = 200;
    const legendWidth = 20;
    
    const legend = svg.append("g")
      .attr("class", "color-legend")
      .attr("transform", `translate(${width - 60}, ${(height - legendHeight) / 2})`);

    // Create gradient
    const gradient = svg.append("defs")
      .append("linearGradient")
      .attr("id", "energy-gradient")
      .attr("x1", "0%")
      .attr("y1", "100%")
      .attr("x2", "0%")
      .attr("y2", "0%");

    gradient.selectAll("stop")
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append("stop")
      .attr("offset", d => `${d * 100}%`)
      .attr("stop-color", d => colorScale(energyExtent[0] + d * (energyExtent[1] - energyExtent[0])));

    // Add legend rectangle
    legend.append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#energy-gradient)");

    // Add legend axis
    const legendScale = d3.scaleLinear()
      .domain(energyExtent)
      .range([legendHeight, 0]);

    const legendAxis = d3.axisRight(legendScale)
      .tickFormat(d => (typeof d === 'number' ? d : d.valueOf()).toFixed(1));

    legend.append("g")
      .attr("transform", `translate(${legendWidth}, 0)`)
      .call(legendAxis);

    // Add legend title
    legend.append("text")
      .attr("x", legendWidth / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Free Energy");
  };

  const addInteractionHandlers = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    data: FreeEnergyDataPoint[],
    xScale: d3.ScaleLinear<number, number>,
    yScale: d3.ScaleLinear<number, number>,
    setSelectedPoint: (point: FreeEnergyDataPoint | null) => void
  ) => {
    g.selectAll(".energy-point, .voronoi-cell")
      .style("cursor", "pointer")
      .on("click", function(event, d) {
        setSelectedPoint(d as FreeEnergyDataPoint);
      })
      .on("mouseover", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("opacity", 1.0);
      })
      .on("mouseout", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("opacity", 0.8);
      });
  };

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            Free Energy Landscape - Agent {agentId}
            <div className="flex gap-2">
              <Badge variant={realTimeUpdates ? 'default' : 'secondary'}>
                {realTimeUpdates ? 'Real-time' : 'Static'}
              </Badge>
              <Badge variant="outline">
                {data.length} data points
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Controls */}
          <div className="flex gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-2">View Mode</label>
              <div className="flex gap-2">
                {(['surface', 'contour', 'heatmap'] as const).map(mode => (
                  <Button
                    key={mode}
                    variant={viewMode === mode ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode(mode)}
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </Button>
                ))}
              </div>
            </div>
            
            <div className="flex-1">
              <label className="block text-sm font-medium mb-2">
                Energy Threshold: [{energyThreshold[0].toFixed(1)}, {energyThreshold[1].toFixed(1)}]
              </label>
              <Slider
                value={energyThreshold}
                onValueChange={setEnergyThreshold}
                min={0}
                max={20}
                step={0.1}
                className="w-full"
              />
            </div>

            <div className="flex-1">
              <label className="block text-sm font-medium mb-2">
                Uncertainty Threshold: [{uncertaintyThreshold[0].toFixed(2)}, {uncertaintyThreshold[1].toFixed(2)}]
              </label>
              <Slider
                value={uncertaintyThreshold}
                onValueChange={setUncertaintyThreshold}
                min={0}
                max={1}
                step={0.01}
                className="w-full"
              />
            </div>
          </div>

          {/* Main visualization */}
          <div className="flex gap-4">
            <div className="flex-1">
              <svg
                ref={svgRef}
                width="800"
                height="600"
                className="border rounded-lg"
                style={{ background: 'white' }}
              />
            </div>

            {/* Selected point details */}
            {selectedPoint && (
              <div className="w-80">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Point Details</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Position:</div>
                      <div className="font-mono">
                        ({selectedPoint.x.toFixed(3)}, {selectedPoint.y.toFixed(3)})
                      </div>
                      
                      <div>Free Energy:</div>
                      <div className="font-mono font-semibold">
                        {selectedPoint.free_energy.toFixed(4)}
                      </div>
                      
                      <div>Uncertainty:</div>
                      <div className="font-mono">
                        {selectedPoint.uncertainty.toFixed(4)}
                      </div>
                      
                      <div>Convergence:</div>
                      <div className="font-mono">
                        {selectedPoint.convergence_score.toFixed(4)}
                      </div>
                      
                      <div>Decision Boundary:</div>
                      <div>
                        <Badge variant={selectedPoint.decision_boundary ? 'default' : 'secondary'}>
                          {selectedPoint.decision_boundary ? 'Yes' : 'No'}
                        </Badge>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Belief State</h4>
                      <div className="space-y-1">
                        {selectedPoint.belief_state.map((belief, idx) => (
                          <div key={idx} className="flex justify-between text-xs">
                            <span>State {idx}:</span>
                            <span className="font-mono">{belief.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Mathematical information */}
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold mb-2">Mathematical Foundation</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <strong>Free Energy:</strong> F = -log P(o) + KL[Q(s)||P(s)]
              </div>
              <div>
                <strong>Decision Boundaries:</strong> ∇F = 0
              </div>
              <div>
                <strong>Uncertainty:</strong> H[Q(s)] = -Σ Q(s) log Q(s)
              </div>
              <div>
                <strong>Convergence:</strong> ||Q_t - Q_{"t-1"}|| {"<"} ε
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default FreeEnergyLandscapeViz;
