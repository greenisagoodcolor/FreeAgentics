"use client";

import { useState } from "react";
import { KnowledgeGraphViz } from "@/components/KnowledgeGraph-viz";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Brain,
  Network,
  TrendingUp,
  Share2,
  Filter,
  Download,
  Maximize2,
} from "lucide-react";

export default function KnowledgePage() {
  const [selectedAgent, setSelectedAgent] = useState<string>("agent_123");
  const [graphMode, setGraphMode] = useState<"individual" | "collective">(
    "individual",
  );
  const [filterType, setFilterType] = useState<string>("all");

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Knowledge Graphs</h1>
          <p className="text-muted-foreground mt-1">
            Visualize agent knowledge, patterns, and collective intelligence
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm">
            <Maximize2 className="h-4 w-4 mr-2" />
            Fullscreen
          </Button>
        </div>
      </div>

      {/* Knowledge Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2,847</div>
            <p className="text-xs text-muted-foreground">+342 this session</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Connections</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">5,621</div>
            <p className="text-xs text-muted-foreground">2.0 avg per node</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Patterns</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">156</div>
            <p className="text-xs text-muted-foreground">23 high confidence</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Shared Knowledge
            </CardTitle>
            <Share2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">892</div>
            <p className="text-xs text-muted-foreground">31% of total</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-4">
        {/* Graph View - Takes up 3 columns */}
        <div className="lg:col-span-3">
          <Card className="h-full">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Knowledge Graph Visualization</CardTitle>
                <div className="flex gap-2">
                  <Select
                    value={graphMode}
                    onValueChange={(v) => setGraphMode(v as any)}
                  >
                    <SelectTrigger className="w-[150px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="individual">Individual</SelectItem>
                      <SelectItem value="collective">Collective</SelectItem>
                    </SelectContent>
                  </Select>
                  {graphMode === "individual" && (
                    <Select
                      value={selectedAgent}
                      onValueChange={setSelectedAgent}
                    >
                      <SelectTrigger className="w-[200px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="agent_123">
                          CuriousExplorer
                        </SelectItem>
                        <SelectItem value="agent_456">WiseScholar</SelectItem>
                        <SelectItem value="agent_789">BravePioneer</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div
                className="relative bg-slate-900 rounded-lg overflow-hidden"
                style={{ height: "600px" }}
              >
                <KnowledgeGraphViz
                  agentId={
                    graphMode === "individual" ? selectedAgent : undefined
                  }
                  mode={graphMode}
                  filter={filterType}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Side Panel - Takes up 1 column */}
        <div className="space-y-4">
          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Filter className="h-4 w-4" />
                Filters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Knowledge</SelectItem>
                  <SelectItem value="locations">Locations</SelectItem>
                  <SelectItem value="agents">Agent Relations</SelectItem>
                  <SelectItem value="resources">Resources</SelectItem>
                  <SelectItem value="patterns">Patterns</SelectItem>
                  <SelectItem value="recent">Recent (last hour)</SelectItem>
                </SelectContent>
              </Select>
              <div className="space-y-2 pt-2">
                <label className="text-sm font-medium">Node Size</label>
                <Select defaultValue="connections">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="uniform">Uniform</SelectItem>
                    <SelectItem value="connections">By Connections</SelectItem>
                    <SelectItem value="importance">By Importance</SelectItem>
                    <SelectItem value="age">By Age</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Graph Stats */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Graph Statistics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Nodes Visible:</span>
                <span className="font-medium">127</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Edges Visible:</span>
                <span className="font-medium">243</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Clusters:</span>
                <span className="font-medium">8</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Path Length:</span>
                <span className="font-medium">3.2</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Density:</span>
                <span className="font-medium">0.15</span>
              </div>
            </CardContent>
          </Card>

          {/* Top Patterns */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Top Patterns</CardTitle>
              <CardDescription>Most confident discoveries</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">
                      Resource Clustering
                    </span>
                    <span className="text-xs text-green-600">92%</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Materials often found near forest edges
                  </p>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">
                      Agent Cooperation
                    </span>
                    <span className="text-xs text-green-600">87%</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Explorers and Scholars share knowledge frequently
                  </p>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">
                      Path Optimization
                    </span>
                    <span className="text-xs text-yellow-600">76%</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Mountain passes provide shortest routes
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button className="w-full" variant="outline" size="sm">
                Compare Agents
              </Button>
              <Button className="w-full" variant="outline" size="sm">
                Find Shortest Path
              </Button>
              <Button className="w-full" variant="outline" size="sm">
                Analyze Clusters
              </Button>
              <Button className="w-full" variant="outline" size="sm">
                Export Subgraph
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
