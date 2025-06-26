"use client";

import { useState } from "react";
import { BackendGridWorld } from "@/components/backend-grid-world";
import { SimulationControls } from "@/components/simulation-controls";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  Map,
  Layers,
  Settings,
  Info,
  Mountain,
  Trees,
  Waves,
} from "lucide-react";

export default function WorldPage() {
  const [selectedHex, setSelectedHex] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"terrain" | "resources" | "agents">(
    "terrain",
  );

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">World Simulation</h1>
          <p className="text-muted-foreground mt-1">
            H3 hexagonal world with dynamic environments and agent interactions
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            World Settings
          </Button>
          <Button variant="outline" size="sm">
            <Info className="h-4 w-4 mr-2" />
            Help
          </Button>
        </div>
      </div>

      {/* World Stats */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">World Size</CardTitle>
            <Map className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1,261</div>
            <p className="text-xs text-muted-foreground">hexagons</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Forest</CardTitle>
            <Trees className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">25%</div>
            <p className="text-xs text-muted-foreground">315 hexes</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Mountains</CardTitle>
            <Mountain className="h-4 w-4 text-gray-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">20%</div>
            <p className="text-xs text-muted-foreground">252 hexes</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Water</CardTitle>
            <Waves className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">10%</div>
            <p className="text-xs text-muted-foreground">126 hexes</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Resources</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">450</div>
            <p className="text-xs text-muted-foreground">total available</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-4">
        {/* World View - Takes up 3 columns */}
        <div className="lg:col-span-3 space-y-4">
          {/* View Controls */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex justify-between items-center">
                <CardTitle className="text-lg">World View</CardTitle>
                <Tabs
                  value={viewMode}
                  onValueChange={(v) => setViewMode(v as any)}
                >
                  <TabsList className="grid w-[300px] grid-cols-3">
                    <TabsTrigger value="terrain">Terrain</TabsTrigger>
                    <TabsTrigger value="resources">Resources</TabsTrigger>
                    <TabsTrigger value="agents">Agents</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            </CardHeader>
            <CardContent>
              <div
                className="relative bg-slate-900 rounded-lg overflow-hidden"
                style={{ height: "600px" }}
              >
                <BackendGridWorld
                  onHexClick={setSelectedHex}
                  selectedHex={selectedHex}
                  viewMode={viewMode}
                />
              </div>
            </CardContent>
          </Card>

          {/* Simulation Controls */}
          <SimulationControls />
        </div>

        {/* Side Panel - Takes up 1 column */}
        <div className="space-y-4">
          {/* Selected Hex Info */}
          {selectedHex && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Hex Details</CardTitle>
                <CardDescription>ID: {selectedHex}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="font-medium">Biome:</div>
                  <div>Forest</div>
                  <div className="font-medium">Elevation:</div>
                  <div>45m</div>
                  <div className="font-medium">Resources:</div>
                  <div>Materials (50)</div>
                  <div className="font-medium">Agents:</div>
                  <div>2 present</div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* World Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">World Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button className="w-full" variant="outline" size="sm">
                Add Resource
              </Button>
              <Button className="w-full" variant="outline" size="sm">
                Modify Terrain
              </Button>
              <Button className="w-full" variant="outline" size="sm">
                Place Agent
              </Button>
              <Button className="w-full" variant="outline" size="sm">
                Create Event
              </Button>
            </CardContent>
          </Card>

          {/* Legend */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Legend</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-green-600 rounded"></div>
                  <span>Forest</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-yellow-600 rounded"></div>
                  <span>Plains</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gray-600 rounded"></div>
                  <span>Mountains</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-orange-600 rounded"></div>
                  <span>Desert</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-600 rounded"></div>
                  <span>Water</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
