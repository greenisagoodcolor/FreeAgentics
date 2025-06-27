"use client";

import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/components/ui/use-toast";
import {
  Download,
  Database,
  Users,
  MessageSquare,
  Network,
  Brain,
  Globe,
  Settings,
  Loader2,
  CheckCircle,
  AlertCircle,
  Share2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ExperimentSharingModal } from "./experiment-sharing-modal";

interface ExportOptions {
  name: string;
  description: string;
  includeAgents: boolean;
  includeConversations: boolean;
  includeKnowledgeGraphs: boolean;
  includeCoalitions: boolean;
  includeInferenceModels: boolean;
  includeWorldState: boolean;
  includeParameters: boolean;
  compression: boolean;
  selectedAgentIds: string[];
  selectedConversationIds: string[];
}

interface ExportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExportComplete?: (exportId: string) => void;
  agents?: { id: string; name: string; type: string }[];
  conversations?: {
    id: string;
    participants: string[];
    messageCount: number;
  }[];
}

export function ExperimentExportModal({
  open,
  onOpenChange,
  onExportComplete,
  agents = [],
  conversations = [],
}: ExportModalProps) {
  const { toast } = useToast();
  const [step, setStep] = useState<
    "config" | "progress" | "complete" | "error"
  >("config");
  const [progress, setProgress] = useState(0);
  const [exportId, setExportId] = useState<string | null>(null);
  const [isSharingModalOpen, setIsSharingModalOpen] = useState(false);
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    name: `Experiment Export - ${new Date().toLocaleDateString()}`,
    description: "",
    includeAgents: true,
    includeConversations: true,
    includeKnowledgeGraphs: true,
    includeCoalitions: true,
    includeInferenceModels: true,
    includeWorldState: true,
    includeParameters: true,
    compression: true,
    selectedAgentIds: [],
    selectedConversationIds: [],
  });

  const handleExport = async () => {
    if (!exportOptions.name.trim()) {
      toast({
        title: "Export name required",
        description: "Please provide a name for this experiment export",
      });
      return;
    }

    setStep("progress");
    setProgress(0);

    // Simulate export process with progress updates
    const interval = setInterval(() => {
      setProgress((prev) => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(interval);
          // Simulate API call completion
          setTimeout(() => {
            const mockExportId = `exp_${Math.random().toString(36).substring(2, 10)}`;
            setExportId(mockExportId);
            setStep("complete");
            if (onExportComplete) {
              onExportComplete(mockExportId);
            }
          }, 500);
          return 100;
        }
        return newProgress;
      });
    }, 300);
  };

  const handleClose = () => {
    // Reset state when closing
    if (step === "complete" || step === "error") {
      setTimeout(() => {
        setStep("config");
        setProgress(0);
        setExportId(null);
      }, 300);
    }
    onOpenChange(false);
  };

  const handleAgentToggle = (agentId: string) => {
    setExportOptions((prev) => {
      const selectedAgentIds = prev.selectedAgentIds.includes(agentId)
        ? prev.selectedAgentIds.filter((id) => id !== agentId)
        : [...prev.selectedAgentIds, agentId];

      return { ...prev, selectedAgentIds };
    });
  };

  const handleConversationToggle = (conversationId: string) => {
    setExportOptions((prev) => {
      const selectedConversationIds = prev.selectedConversationIds.includes(
        conversationId,
      )
        ? prev.selectedConversationIds.filter((id) => id !== conversationId)
        : [...prev.selectedConversationIds, conversationId];

      return { ...prev, selectedConversationIds };
    });
  };

  const handleSelectAllAgents = () => {
    setExportOptions((prev) => ({
      ...prev,
      selectedAgentIds:
        prev.selectedAgentIds.length === agents.length
          ? []
          : agents.map((a) => a.id),
    }));
  };

  const handleSelectAllConversations = () => {
    setExportOptions((prev) => ({
      ...prev,
      selectedConversationIds:
        prev.selectedConversationIds.length === conversations.length
          ? []
          : conversations.map((c) => c.id),
    }));
  };

  const renderConfigStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Export Experiment State</DialogTitle>
        <DialogDescription>
          Export your experiment state for reproducible research, collaboration,
          or backup.
        </DialogDescription>
      </DialogHeader>

      <div className="grid gap-4 py-4">
        <div className="grid gap-2">
          <Label htmlFor="export-name">Export Name</Label>
          <Input
            id="export-name"
            value={exportOptions.name}
            onChange={(e) =>
              setExportOptions({ ...exportOptions, name: e.target.value })
            }
            placeholder="My Experiment Export"
          />
        </div>

        <div className="grid gap-2">
          <Label htmlFor="export-description">Description (Optional)</Label>
          <Textarea
            id="export-description"
            value={exportOptions.description}
            onChange={(e) =>
              setExportOptions({
                ...exportOptions,
                description: e.target.value,
              })
            }
            placeholder="Describe the contents and purpose of this export"
            rows={2}
          />
        </div>

        <div className="grid gap-2 pt-2">
          <Label className="text-base">Components to Include</Label>

          <div className="grid grid-cols-2 gap-4 pt-1">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-agents"
                checked={exportOptions.includeAgents}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeAgents: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-agents"
                  className="flex items-center gap-1"
                >
                  <Users className="h-4 w-4 text-muted-foreground" />
                  <span>Agents</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-conversations"
                checked={exportOptions.includeConversations}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeConversations: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-conversations"
                  className="flex items-center gap-1"
                >
                  <MessageSquare className="h-4 w-4 text-muted-foreground" />
                  <span>Conversations</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-knowledge"
                checked={exportOptions.includeKnowledgeGraphs}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeKnowledgeGraphs: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-knowledge"
                  className="flex items-center gap-1"
                >
                  <Database className="h-4 w-4 text-muted-foreground" />
                  <span>Knowledge Graphs</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-coalitions"
                checked={exportOptions.includeCoalitions}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeCoalitions: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-coalitions"
                  className="flex items-center gap-1"
                >
                  <Network className="h-4 w-4 text-muted-foreground" />
                  <span>Coalitions</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-inference"
                checked={exportOptions.includeInferenceModels}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeInferenceModels: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-inference"
                  className="flex items-center gap-1"
                >
                  <Brain className="h-4 w-4 text-muted-foreground" />
                  <span>Inference Models</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-world"
                checked={exportOptions.includeWorldState}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeWorldState: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-world"
                  className="flex items-center gap-1"
                >
                  <Globe className="h-4 w-4 text-muted-foreground" />
                  <span>World State</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-parameters"
                checked={exportOptions.includeParameters}
                onCheckedChange={(checked) =>
                  setExportOptions({
                    ...exportOptions,
                    includeParameters: !!checked,
                  })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="include-parameters"
                  className="flex items-center gap-1"
                >
                  <Settings className="h-4 w-4 text-muted-foreground" />
                  <span>Parameters</span>
                </Label>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="compression"
                checked={exportOptions.compression}
                onCheckedChange={(checked) =>
                  setExportOptions({ ...exportOptions, compression: !!checked })
                }
              />
              <div className="grid gap-1.5">
                <Label
                  htmlFor="compression"
                  className="flex items-center gap-1"
                >
                  <Download className="h-4 w-4 text-muted-foreground" />
                  <span>Compression</span>
                </Label>
              </div>
            </div>
          </div>
        </div>

        {exportOptions.includeAgents && agents.length > 0 && (
          <div className="grid gap-2 pt-2">
            <div className="flex items-center justify-between">
              <Label className="text-base">Select Agents</Label>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSelectAllAgents}
                className="h-7 text-xs"
              >
                {exportOptions.selectedAgentIds.length === agents.length
                  ? "Deselect All"
                  : "Select All"}
              </Button>
            </div>

            <div className="max-h-40 overflow-y-auto border rounded-md p-2">
              {agents.map((agent) => (
                <div
                  key={agent.id}
                  className="flex items-center space-x-2 py-1"
                >
                  <Checkbox
                    id={`agent-${agent.id}`}
                    checked={exportOptions.selectedAgentIds.includes(agent.id)}
                    onCheckedChange={() => handleAgentToggle(agent.id)}
                  />
                  <Label
                    htmlFor={`agent-${agent.id}`}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <span>{agent.name}</span>
                    <span className="text-xs text-muted-foreground">
                      ({agent.type})
                    </span>
                  </Label>
                </div>
              ))}
              {agents.length === 0 && (
                <div className="text-sm text-muted-foreground py-2 text-center">
                  No agents available
                </div>
              )}
            </div>
          </div>
        )}

        {exportOptions.includeConversations && conversations.length > 0 && (
          <div className="grid gap-2 pt-2">
            <div className="flex items-center justify-between">
              <Label className="text-base">Select Conversations</Label>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSelectAllConversations}
                className="h-7 text-xs"
              >
                {exportOptions.selectedConversationIds.length ===
                conversations.length
                  ? "Deselect All"
                  : "Select All"}
              </Button>
            </div>

            <div className="max-h-40 overflow-y-auto border rounded-md p-2">
              {conversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className="flex items-center space-x-2 py-1"
                >
                  <Checkbox
                    id={`conversation-${conversation.id}`}
                    checked={exportOptions.selectedConversationIds.includes(
                      conversation.id,
                    )}
                    onCheckedChange={() =>
                      handleConversationToggle(conversation.id)
                    }
                  />
                  <Label
                    htmlFor={`conversation-${conversation.id}`}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <span>Conversation {conversation.id.substring(0, 8)}</span>
                    <span className="text-xs text-muted-foreground">
                      ({conversation.messageCount} messages,{" "}
                      {conversation.participants.length} participants)
                    </span>
                  </Label>
                </div>
              ))}
              {conversations.length === 0 && (
                <div className="text-sm text-muted-foreground py-2 text-center">
                  No conversations available
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={handleClose}>
          Cancel
        </Button>
        <Button onClick={handleExport}>
          <Download className="mr-2 h-4 w-4" />
          Export
        </Button>
      </DialogFooter>
    </>
  );

  const renderProgressStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Exporting Experiment State</DialogTitle>
        <DialogDescription>
          Please wait while we prepare your experiment export...
        </DialogDescription>
      </DialogHeader>

      <div className="py-8 space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="relative w-20 h-20 flex items-center justify-center">
            <div className="absolute inset-0 border-4 border-primary/30 rounded-full" />
            <div
              className="absolute inset-0 border-4 border-primary rounded-full"
              style={{
                clipPath: `polygon(0% 0%, ${progress}% 0%, ${progress}% 100%, 0% 100%)`,
                transition: "clip-path 0.3s ease-in-out",
              }}
            />
            <span className="text-lg font-semibold">
              {Math.round(progress)}%
            </span>
          </div>
        </div>

        <div className="space-y-2">
          <Progress value={progress} className="h-2" />
          <div className="text-center text-sm text-muted-foreground">
            Exporting experiment data...
          </div>
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" disabled>
          Cancel
        </Button>
        <Button disabled>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Exporting...
        </Button>
      </DialogFooter>
    </>
  );

  const renderCompleteStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Export Complete</DialogTitle>
        <DialogDescription>
          Your experiment has been successfully exported.
        </DialogDescription>
      </DialogHeader>

      <div className="py-8 space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-3 rounded-full bg-green-100">
            <CheckCircle className="h-10 w-10 text-green-600" />
          </div>
          <div className="text-center">
            <h3 className="text-lg font-medium">Export Successful</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Export ID: {exportId}
            </p>
          </div>
        </div>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Export Summary</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Name:</span>
                <span className="font-medium">{exportOptions.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Components:</span>
                <span className="font-medium">
                  {[
                    exportOptions.includeAgents && "Agents",
                    exportOptions.includeConversations && "Conversations",
                    exportOptions.includeKnowledgeGraphs && "Knowledge Graphs",
                    exportOptions.includeCoalitions && "Coalitions",
                    exportOptions.includeInferenceModels && "Inference Models",
                    exportOptions.includeWorldState && "World State",
                    exportOptions.includeParameters && "Parameters",
                  ]
                    .filter(Boolean)
                    .join(", ")}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Date:</span>
                <span className="font-medium">
                  {new Date().toLocaleString()}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <DialogFooter className="space-x-2">
        <Button variant="outline" onClick={handleClose}>
          Close
        </Button>
        <Button variant="secondary" onClick={() => setIsSharingModalOpen(true)}>
          <Share2 className="mr-2 h-4 w-4" />
          Share Export
        </Button>
        <Button>
          <Download className="mr-2 h-4 w-4" />
          Download
        </Button>
      </DialogFooter>
    </>
  );

  const renderErrorStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Export Failed</DialogTitle>
        <DialogDescription>
          There was an error exporting your experiment.
        </DialogDescription>
      </DialogHeader>

      <div className="py-8 space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-3 rounded-full bg-red-100">
            <AlertCircle className="h-10 w-10 text-red-600" />
          </div>
          <div className="text-center">
            <h3 className="text-lg font-medium">Export Failed</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Please try again or contact support if the issue persists.
            </p>
          </div>
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={handleClose}>
          Close
        </Button>
        <Button onClick={() => setStep("config")}>Try Again</Button>
      </DialogFooter>
    </>
  );

  return (
    <>
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="sm:max-w-[600px]">
          {step === "config" && renderConfigStep()}
          {step === "progress" && renderProgressStep()}
          {step === "complete" && renderCompleteStep()}
          {step === "error" && renderErrorStep()}
        </DialogContent>
      </Dialog>

      {exportId && (
        <ExperimentSharingModal
          open={isSharingModalOpen}
          onOpenChange={setIsSharingModalOpen}
          exportId={exportId}
          exportName={exportOptions.name}
          exportDescription={exportOptions.description}
        />
      )}
    </>
  );
}
