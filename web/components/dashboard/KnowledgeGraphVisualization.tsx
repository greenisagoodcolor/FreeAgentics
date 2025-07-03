"use client";

import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { useAppSelector } from "@/store/hooks";

interface KnowledgeGraphVisualizationProps {
  testMode?: boolean;
  zoom?: number;
}

const KnowledgeGraphVisualization: React.FC<KnowledgeGraphVisualizationProps> = ({ 
  testMode = false,
  zoom = 1 
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Redux state
  const knowledgeGraph = useAppSelector((state) => state.knowledge.graph);

  useEffect(() => {
    if (!svgRef.current) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current);
    const g = svg.append("g");

    // Sample data for visualization
    const nodes = [
      { id: "1", label: "Coalition Formation", x: 200, y: 200, type: "belief" },
      { id: "2", label: "Active Inference", x: 400, y: 150, type: "fact" },
      { id: "3", label: "Resource Allocation", x: 600, y: 200, type: "hypothesis" },
      { id: "4", label: "Agent Alpha", x: 300, y: 350, type: "agent" },
      { id: "5", label: "Agent Beta", x: 500, y: 350, type: "agent" },
      { id: "6", label: "Belief Propagation", x: 400, y: 250, type: "belief" },
    ];

    const links = [
      { source: "1", target: "2" },
      { source: "2", target: "3" },
      { source: "4", target: "1" },
      { source: "5", target: "3" },
      { source: "6", target: "2" },
      { source: "4", target: "6" },
      { source: "5", target: "6" },
    ];

    // Create links
    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("x1", d => nodes.find(n => n.id === d.source)!.x)
      .attr("y1", d => nodes.find(n => n.id === d.source)!.y)
      .attr("x2", d => nodes.find(n => n.id === d.target)!.x)
      .attr("y2", d => nodes.find(n => n.id === d.target)!.y)
      .attr("stroke", "#4B5563")
      .attr("stroke-width", 2)
      .attr("opacity", 0.6);

    // Create nodes
    const node = g.append("g")
      .selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("transform", d => `translate(${d.x},${d.y})`);

    // Add circles
    node.append("circle")
      .attr("r", 30)
      .attr("fill", d => {
        switch(d.type) {
          case "belief": return "#4F46E5";
          case "fact": return "#10B981";
          case "hypothesis": return "#F59E0B";
          case "agent": return "#EF4444";
          default: return "#6B7280";
        }
      })
      .attr("stroke", "#1F2937")
      .attr("stroke-width", 2);

    // Add labels
    node.append("text")
      .text(d => d.label)
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("fill", "white")
      .attr("font-size", "12px")
      .attr("pointer-events", "none")
      .style("user-select", "none");

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 3])
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom);

    // Initial zoom to fit content
    const bounds = g.node()?.getBBox();
    if (bounds) {
      const fullWidth = bounds.width;
      const fullHeight = bounds.height;
      const midX = bounds.x + fullWidth / 2;
      const midY = bounds.y + fullHeight / 2;
      const scale = 0.9 / Math.max(fullWidth / dimensions.width, fullHeight / dimensions.height);
      
      svg.call(
        zoom.transform,
        d3.zoomIdentity
          .translate(dimensions.width / 2, dimensions.height / 2)
          .scale(scale)
          .translate(-midX, -midY)
      );
    }

  }, [dimensions, knowledgeGraph, testMode]);

  // Apply zoom changes
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const g = svg.select('g');
    
    if (g.node()) {
      const currentTransform = d3.zoomTransform(svg.node()!);
      const newTransform = currentTransform.scale(zoom);
      svg.transition()
        .duration(300)
        .call(d3.zoom<SVGSVGElement, unknown>().transform, newTransform);
    }
  }, [zoom]);

  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current?.parentElement) {
        const { width, height } = svgRef.current.parentElement.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  return (
    <div className="w-full h-full" ref={svgRef.current?.parentElement as any}>
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="knowledge-graph-svg"
        data-testid="knowledge-graph-svg"
        style={{ backgroundColor: "transparent" }}
      />
    </div>
  );
};

export default KnowledgeGraphVisualization;