"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ExperimentExportModal } from "@/components/ui/experiment-export-modal";
import { ExperimentImportModal } from "@/components/ui/experiment-import-modal";
import { ExperimentSharingModal } from "@/components/ui/experiment-sharing-modal";
import { useToast } from "@/components/ui/use-toast";
import {
  Download,
  Upload,
  Search,
  Calendar,
  User,
  FileJson,
  Package,
  Trash2,
  Share2,
  Download as DownloadIcon,
  Clock,
  Filter,
  FileText,
  Info
} from "lucide-react";

interface ExperimentExport {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  createdBy: string;
  components: string[];
  totalAgents: number;
  totalConversations: number;
  totalMessages: number;
  totalKnowledgeNodes: number;
  fileSizeMb: number;
}

// Mock data for exports
const mockExports: ExperimentExport[] = [
  {
    id: "exp_a1b2c3d4",
    name: "Coalition Formation Experiment",
    description: "Baseline experiment for coalition formation with 5 agents",
    createdAt: "2025-06-20T14:30:00Z",
    createdBy: "researcher@example.com",
    components: ["Agents", "Conversations", "Knowledge Graphs", "Coalitions"],
    totalAgents: 5,
    totalConversations: 12,
    totalMessages: 156,
    totalKnowledgeNodes: 48,
    fileSizeMb: 2.4
  },
  {
    id: "exp_e5f6g7h8",
    name: "Knowledge Transfer Study",
    description: "Experiment studying knowledge transfer between agents with different belief systems",
    createdAt: "2025-06-15T09:45:00Z",
    createdBy: "admin@example.com",
    components: ["Agents", "Conversations", "Knowledge Graphs"],
    totalAgents: 3,
    totalConversations: 8,
    totalMessages: 97,
    totalKnowledgeNodes: 32,
    fileSizeMb: 1.8
  },
  {
    id: "exp_i9j0k1l2",
    name: "Active Inference Parameter Tuning",
    description: "Parameter optimization for active inference models",
    createdAt: "2025-06-10T16:20:00Z",
    createdBy: "researcher@example.com",
    components: ["Agents", "Inference Models", "Parameters"],
    totalAgents: 2,
    totalConversations: 0,
    totalMessages: 0,
    totalKnowledgeNodes: 15,
    fileSizeMb: 3.2
  }
];

// Mock data for agents
const mockAgents = [
  { id: "agent_001", name: "Explorer Agent", type: "explorer" },
  { id: "agent_002", name: "Guardian Agent", type: "guardian" },
  { id: "agent_003", name: "Merchant Agent", type: "merchant" },
  { id: "agent_004", name: "Scholar Agent", type: "scholar" }
];

// Mock data for conversations
const mockConversations = [
  { id: "conv_001", participants: ["agent_001", "agent_002"], messageCount: 24 },
  { id: "conv_002", participants: ["agent_001", "agent_003", "agent_004"], messageCount: 36 },
  { id: "conv_003", participants: ["agent_002", "agent_004"], messageCount: 18 }
];

export default function ExperimentsPage() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredExports, setFilteredExports] = useState<ExperimentExport[]>(mockExports);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  const [isSharingModalOpen, setIsSharingModalOpen] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentExport | null>(null);
  const [activeTab, setActiveTab] = useState("exports");

  useEffect(() => {
    // Filter exports based on search query
    const filtered = mockExports.filter(exp => 
      exp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      exp.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      exp.id.toLowerCase().includes(searchQuery.toLowerCase())
    );
    setFilteredExports(filtered);
  }, [searchQuery]);

  const handleExportComplete = (exportId: string) => {
    toast({
      title: "Export Complete",
      description: `Experiment export ${exportId} has been created successfully.`,
      duration: 5000,
    });
  };

  const handleImportComplete = (importId: string) => {
    toast({
      title: "Import Complete",
      description: `Experiment import ${importId} has been completed successfully.`,
      duration: 5000,
    });
  };

  const handleDeleteExport = (exportId: string) => {
    toast({
      title: "Export Deleted",
      description: `Experiment export ${exportId} has been deleted.`,
      duration: 3000,
    });
  };

  const handleShareExport = (experiment: ExperimentExport) => {
    setSelectedExperiment(experiment);
    setIsSharingModalOpen(true);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="container mx-auto py-8 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Experiment Management</h1>
          <p className="text-muted-foreground mt-1">
            Export, import, and manage experiment states for reproducible research
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" onClick={() => setIsImportModalOpen(true)}>
            <Upload className="mr-2 h-4 w-4" />
            Import
          </Button>
          <Button onClick={() => setIsExportModalOpen(true)}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      <Tabs defaultValue="exports" className="space-y-6" onValueChange={setActiveTab}>
        <div className="flex justify-between items-center">
          <TabsList>
            <TabsTrigger value="exports">Exports</TabsTrigger>
            <TabsTrigger value="imports">Imports</TabsTrigger>
            <TabsTrigger value="templates">Templates</TabsTrigger>
          </TabsList>

          <div className="relative w-72">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search experiments..."
              className="pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <TabsContent value="exports" className="space-y-4">
          {filteredExports.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Package className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No exports found</h3>
              <p className="text-sm text-muted-foreground mt-1 mb-6">
                {searchQuery ? "No exports match your search query." : "You haven't created any exports yet."}
              </p>
              {!searchQuery && (
                <Button onClick={() => setIsExportModalOpen(true)}>
                  <Download className="mr-2 h-4 w-4" />
                  Create Export
                </Button>
              )}
            </div>
          ) : (
            <div className="grid gap-4">
              {filteredExports.map((exp) => (
                <Card key={exp.id} className="overflow-hidden">
                  <CardHeader className="pb-3">
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle>{exp.name}</CardTitle>
                        <CardDescription className="mt-1">{exp.description}</CardDescription>
                      </div>
                      <Badge variant="outline" className="ml-2 text-xs">
                        {exp.components.length} components
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pb-3">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{formatDate(exp.createdAt)}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <User className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{exp.createdBy.split('@')[0]}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <FileJson className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{exp.fileSizeMb.toFixed(1)} MB</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Info className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">ID: {exp.id.substring(0, 8)}</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-3 text-center">
                      <div className="bg-primary/5 rounded-md p-2">
                        <div className="text-lg font-semibold">{exp.totalAgents}</div>
                        <div className="text-xs text-muted-foreground">Agents</div>
                      </div>
                      <div className="bg-primary/5 rounded-md p-2">
                        <div className="text-lg font-semibold">{exp.totalConversations}</div>
                        <div className="text-xs text-muted-foreground">Conversations</div>
                      </div>
                      <div className="bg-primary/5 rounded-md p-2">
                        <div className="text-lg font-semibold">{exp.totalMessages}</div>
                        <div className="text-xs text-muted-foreground">Messages</div>
                      </div>
                      <div className="bg-primary/5 rounded-md p-2">
                        <div className="text-lg font-semibold">{exp.totalKnowledgeNodes}</div>
                        <div className="text-xs text-muted-foreground">Knowledge Nodes</div>
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-between pt-2">
                    <div className="flex gap-1">
                      {exp.components.map((component, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {component}
                        </Badge>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <Button variant="ghost" size="icon" title="Delete export" onClick={() => handleDeleteExport(exp.id)}>
                        <Trash2 className="h-4 w-4 text-muted-foreground" />
                      </Button>
                      <Button variant="ghost" size="icon" title="Share export" onClick={() => handleShareExport(exp)}>
                        <Share2 className="h-4 w-4 text-muted-foreground" />
                      </Button>
                      <Button variant="ghost" size="icon" title="Download export">
                        <DownloadIcon className="h-4 w-4 text-muted-foreground" />
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="imports" className="space-y-4">
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Upload className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">No imported experiments</h3>
            <p className="text-sm text-muted-foreground mt-1 mb-6">
              Import an experiment to reproduce research or collaborate with others.
            </p>
            <Button onClick={() => setIsImportModalOpen(true)}>
              <Upload className="mr-2 h-4 w-4" />
              Import Experiment
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="templates" className="space-y-4">
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <FileText className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">No experiment templates</h3>
            <p className="text-sm text-muted-foreground mt-1 mb-6">
              Templates help you quickly set up standardized experiments.
            </p>
            <Button disabled>
              Create Template
            </Button>
            <p className="text-xs text-muted-foreground mt-4">
              Coming soon in a future update
            </p>
          </div>
        </TabsContent>
      </Tabs>

      <ExperimentExportModal
        open={isExportModalOpen}
        onOpenChange={setIsExportModalOpen}
        onExportComplete={handleExportComplete}
        agents={mockAgents}
        conversations={mockConversations}
      />

      <ExperimentImportModal
        open={isImportModalOpen}
        onOpenChange={setIsImportModalOpen}
        onImportComplete={handleImportComplete}
      />

      {selectedExperiment && (
        <ExperimentSharingModal
          open={isSharingModalOpen}
          onOpenChange={setIsSharingModalOpen}
          exportId={selectedExperiment.id}
          exportName={selectedExperiment.name}
          exportDescription={selectedExperiment.description}
        />
      )}
    </div>
  );
}
