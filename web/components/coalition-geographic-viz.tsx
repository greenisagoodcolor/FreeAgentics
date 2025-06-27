"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

/**
 * Coalition Geographic Visualization Component
 *
 * Renders coalition locations on H3 hexagonal grid using D3.js
 * Integrates with /world architecture and displays real-time business value
 * Following ADR-002 canonical structure and Task 36.4 requirements
 */

interface H3Cell {
  hex_id: string;
  coordinates: [number, number];
  biome: string;
  terrain: string;
  elevation: number;
  temperature: number;
  moisture: number;
  resources: Record<string, number>;
  movement_cost: number;
  visibility_range: number;
}

interface Coalition {
  coalition_id: string;
  name: string;
  description: string;
  members: Array<{
    agent_id: string;
    role: string;
    capabilities: string[];
    resources: Record<string, number>;
    location?: [number, number]; // lat, lng
  }>;
  business_value?: {
    synergy_score: number;
    risk_reduction: number;
    market_positioning: number;
    sustainability_score: number;
    total_value: number;
  };
  status: string;
  formation_timestamp: string;
}

interface CoalitionGeographicVizProps {
  coalitions: Coalition[];
  h3Cells: H3Cell[];
  onCoalitionSelect?: (coalition: Coalition) => void;
  realTimeUpdates?: boolean;
  showBusinessMetrics?: boolean;
}

export const CoalitionGeographicViz: React.FC<CoalitionGeographicVizProps> = ({
  coalitions,
  h3Cells,
  onCoalitionSelect,
  realTimeUpdates = true,
  showBusinessMetrics = true,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedCoalition, setSelectedCoalition] = useState<Coalition | null>(
    null,
  );
  const [viewMode, setViewMode] = useState<
    "geographic" | "business" | "temporal"
  >("geographic");
  const [zoomLevel, setZoomLevel] = useState(1);
  const [center, setCenter] = useState<[number, number]>([0, 0]);

  const handleCoalitionClick = useCallback(
    (coalition: Coalition) => {
      setSelectedCoalition(coalition);
      onCoalitionSelect?.(coalition);
    },
    [onCoalitionSelect],
  );

  // D3 setup and rendering
  useEffect(() => {
    if (!svgRef.current || !h3Cells.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const width = 800;
    const height = 600;

    // Set up projection for geographic coordinates
    const projection = d3
      .geoMercator()
      .scale(150 * zoomLevel)
      .translate([width / 2 + center[0], height / 2 + center[1]]);

    const g = svg.append("g");

    // Create hexagon path generator
    const hexRadius = 5 * zoomLevel;
    const hexPath = (d: [number, number]) => {
      const [x, y] = projection(d) || [0, 0];
      const points = [];
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI * 2 * i) / 6;
        const px = x + hexRadius * Math.cos(angle);
        const py = y + hexRadius * Math.sin(angle);
        points.push(`${px},${py}`);
      }
      return `M${points.join("L")}Z`;
    };

    // Render H3 cells
    g.selectAll(".hex-cell")
      .data(h3Cells)
      .enter()
      .append("path")
      .attr("class", "hex-cell")
      .attr("d", (d) => hexPath(d.coordinates))
      .attr("fill", (d) => getBiomeColor(d.biome))
      .attr("stroke", "#fff")
      .attr("stroke-width", 0.5)
      .attr("opacity", 0.6);

    // Render coalitions
    const coalitionData = coalitions.filter((c) =>
      c.members.some((m) => m.location),
    );

    coalitionData.forEach((coalition) => {
      const memberLocations = coalition.members
        .filter((m) => m.location)
        .map((m) => m.location!);

      if (memberLocations.length === 0) return;

      // Calculate coalition center
      const centerLat =
        memberLocations.reduce((sum, loc) => sum + loc[0], 0) /
        memberLocations.length;
      const centerLng =
        memberLocations.reduce((sum, loc) => sum + loc[1], 0) /
        memberLocations.length;
      const coalitionCenter = projection([centerLng, centerLat]);

      if (!coalitionCenter) return;

      // Draw coalition boundary (convex hull of member locations)
      const projectedLocations = memberLocations
        .map((loc) => projection([loc[1], loc[0]]))
        .filter((p) => p !== null) as Array<[number, number]>;

      if (projectedLocations.length > 2) {
        const hull = d3.polygonHull(projectedLocations);
        if (hull) {
          g.append("path")
            .datum(hull)
            .attr("class", "coalition-boundary")
            .attr("d", d3.line().curve(d3.curveCardinalClosed))
            .attr("fill", getCoalitionColor(coalition))
            .attr("fill-opacity", 0.2)
            .attr("stroke", getCoalitionColor(coalition))
            .attr("stroke-width", 2)
            .style("cursor", "pointer")
            .on("click", () => handleCoalitionClick(coalition));
        }
      }

      // Draw member positions
      memberLocations.forEach((location, index) => {
        const member = coalition.members[index];
        const pos = projection([location[1], location[0]]);
        if (!pos) return;

        g.append("circle")
          .attr("class", "coalition-member")
          .attr("cx", pos[0])
          .attr("cy", pos[1])
          .attr("r", 4 * zoomLevel)
          .attr("fill", getCoalitionColor(coalition))
          .attr("stroke", "#fff")
          .attr("stroke-width", 1)
          .style("cursor", "pointer")
          .on("click", () => handleCoalitionClick(coalition));
      });

      // Draw coalition center with business value indicator
      g.append("circle")
        .attr("class", "coalition-center")
        .attr("cx", coalitionCenter[0])
        .attr("cy", coalitionCenter[1])
        .attr(
          "r",
          getBusinessValueRadius(coalition.business_value?.total_value || 0),
        )
        .attr(
          "fill",
          getBusinessValueColor(coalition.business_value?.total_value || 0),
        )
        .attr("stroke", "#333")
        .attr("stroke-width", 2)
        .style("cursor", "pointer")
        .on("click", () => handleCoalitionClick(coalition));

      // Add coalition label
      g.append("text")
        .attr("class", "coalition-label")
        .attr("x", coalitionCenter[0])
        .attr("y", coalitionCenter[1] - 15)
        .attr("text-anchor", "middle")
        .attr("font-size", `${10 * zoomLevel}px`)
        .attr("font-weight", "bold")
        .attr("fill", "#333")
        .text(coalition.name)
        .style("cursor", "pointer")
        .on("click", () => handleCoalitionClick(coalition));
    });

    // Add zoom functionality
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .on("zoom", (event) => {
        const { transform } = event;
        setZoomLevel(transform.k);
        setCenter([transform.x, transform.y]);
        g.attr("transform", transform);
      });

    svg.call(zoom);
  }, [h3Cells, coalitions, zoomLevel, center, viewMode, handleCoalitionClick]);

  const getBiomeColor = (biome: string): string => {
    const biomeColors: Record<string, string> = {
      forest: "#228B22",
      grassland: "#9ACD32",
      desert: "#F4A460",
      mountain: "#696969",
      ocean: "#4682B4",
      arctic: "#E0FFFF",
      jungle: "#006400",
      coastal: "#20B2AA",
      savanna: "#BDB76B",
      tundra: "#D3D3D3",
    };
    return biomeColors[biome] || "#DDD";
  };

  const getCoalitionColor = (coalition: Coalition): string => {
    // Use coalition ID hash for consistent coloring
    const hash = coalition.coalition_id
      .split("")
      .reduce((acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0);
    const colors = [
      "#FF6B6B",
      "#4ECDC4",
      "#45B7D1",
      "#FFA07A",
      "#98D8C8",
      "#F7DC6F",
    ];
    return colors[Math.abs(hash) % colors.length];
  };

  const getBusinessValueRadius = (value: number): number => {
    return 6 + value * 10; // 6-16px radius based on business value
  };

  const getBusinessValueColor = (value: number): string => {
    if (value > 0.8) return "#00C851"; // High value - green
    if (value > 0.6) return "#FFB347"; // Medium value - orange
    if (value > 0.3) return "#FF6B6B"; // Low value - red
    return "#DDD"; // No/unknown value - gray
  };

  const formatBusinessValue = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            Coalition Geographic Visualization
            <div className="flex gap-2">
              <Button
                variant={viewMode === "geographic" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("geographic")}
              >
                Geographic
              </Button>
              <Button
                variant={viewMode === "business" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("business")}
              >
                Business Value
              </Button>
              <Button
                variant={viewMode === "temporal" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("temporal")}
              >
                Timeline
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            {/* Main visualization */}
            <div className="flex-1">
              <svg
                ref={svgRef}
                width="800"
                height="600"
                className="border rounded-lg"
                style={{ background: "#f8f9fa" }}
              />
            </div>

            {/* Coalition details panel */}
            {selectedCoalition && (
              <div className="w-80 space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">
                      {selectedCoalition.name}
                    </CardTitle>
                    <Badge variant="outline">{selectedCoalition.status}</Badge>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-gray-600">
                      {selectedCoalition.description}
                    </p>

                    <div>
                      <h4 className="font-semibold mb-2">
                        Members ({selectedCoalition.members.length})
                      </h4>
                      <div className="space-y-1">
                        {selectedCoalition.members.map((member) => (
                          <div
                            key={member.agent_id}
                            className="flex justify-between text-sm"
                          >
                            <span>{member.agent_id}</span>
                            <Badge variant="secondary">{member.role}</Badge>
                          </div>
                        ))}
                      </div>
                    </div>

                    {showBusinessMetrics &&
                      selectedCoalition.business_value && (
                        <div>
                          <h4 className="font-semibold mb-2">Business Value</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Total Value:</span>
                              <span className="font-semibold">
                                {formatBusinessValue(
                                  selectedCoalition.business_value.total_value,
                                )}
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Synergy:</span>
                              <span>
                                {formatBusinessValue(
                                  selectedCoalition.business_value
                                    .synergy_score,
                                )}
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Risk Reduction:</span>
                              <span>
                                {formatBusinessValue(
                                  selectedCoalition.business_value
                                    .risk_reduction,
                                )}
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Market Position:</span>
                              <span>
                                {formatBusinessValue(
                                  selectedCoalition.business_value
                                    .market_positioning,
                                )}
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>Sustainability:</span>
                              <span>
                                {formatBusinessValue(
                                  selectedCoalition.business_value
                                    .sustainability_score,
                                )}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                    <div className="text-xs text-gray-500">
                      Formed:{" "}
                      {new Date(
                        selectedCoalition.formation_timestamp,
                      ).toLocaleDateString()}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="mt-4 flex gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span>High Business Value (&gt;80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-orange-400"></div>
              <span>Medium Business Value (60-80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-400"></div>
              <span>Low Business Value (30-60%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-400"></div>
              <span>Unknown Value</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CoalitionGeographicViz;
