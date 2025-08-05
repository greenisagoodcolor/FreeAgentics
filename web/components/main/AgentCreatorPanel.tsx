"use client";

import React, { useState } from "react";
import { Plus, Edit2, Trash2, Loader2, AlertCircle, Wifi, WifiOff } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useAgents, type Agent, type AgentStatus } from "@/hooks/use-agents";
import { useConversation } from "@/hooks/use-conversation";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";
import { cn } from "@/lib/utils";

export function AgentCreatorPanel() {
  const { connectionState, connectionError } = usePromptProcessor();
  const { agents, createAgent, updateAgent, deleteAgent, isLoading, error } = useAgents();
  const { setGoalPrompt } = useConversation();

  const [description, setDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [editingAgent, setEditingAgent] = useState<string | null>(null);
  const [editName, setEditName] = useState("");

  const handleCreateAgent = async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!description.trim()) return;

    setIsCreating(true);
    setCreateError(null);

    try {
      await createAgent({ description: description.trim() });
      // Set the goal prompt so it's visible in the conversation
      setGoalPrompt(description.trim());
      setDescription("");
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setIsCreating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleCreateAgent();
    }
  };

  const handleEditAgent = (agent: Agent) => {
    setEditingAgent(agent.id);
    setEditName(agent.name);
  };

  const handleSaveEdit = async () => {
    if (!editingAgent || !editName.trim()) return;

    try {
      await updateAgent(editingAgent, { name: editName.trim() });
      setEditingAgent(null);
      setEditName("");
    } catch (err) {
      console.error("Failed to update agent:", err);
    }
  };

  const handleDeleteAgent = async (id: string) => {
    try {
      await deleteAgent(id);
    } catch (err) {
      console.error("Failed to delete agent:", err);
    }
  };

  const getStatusColor = (status: AgentStatus) => {
    switch (status) {
      case "active":
        return "bg-green-500";
      case "idle":
        return "bg-yellow-500";
      case "error":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusBadgeVariant = (status: AgentStatus) => {
    switch (status) {
      case "active":
        return "default";
      case "idle":
        return "secondary";
      case "error":
        return "destructive";
      default:
        return "outline";
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Agent Creator</CardTitle>
            <CardDescription>Create and manage your AI agents</CardDescription>
          </div>
          <div className="flex items-center gap-2" title={connectionError || `WebSocket ${connectionState}`}>
            <div
              data-testid="connection-status"
              className={cn(
                "w-2 h-2 rounded-full",
                connectionState === "connected" && "bg-green-500",
                connectionState === "connecting" && "bg-yellow-500 animate-pulse",
                connectionState === "disconnected" && "bg-gray-500",
                connectionState === "error" && "bg-red-500"
              )}
            />
            {connectionState === "connected" ? (
              <Wifi className="h-4 w-4 text-green-600" />
            ) : connectionState === "connecting" ? (
              <Loader2 className="h-4 w-4 text-yellow-600 animate-spin" />
            ) : (
              <WifiOff className="h-4 w-4 text-red-600" />
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col gap-4">
        {/* Create Agent Form */}
        <form onSubmit={handleCreateAgent} className="space-y-3">
          <div className="space-y-2">
            <Label htmlFor="agent-description">Agent Description</Label>
            <Textarea
              id="agent-description"
              placeholder="Describe the agent you want to create..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isCreating || isLoading}
              className="min-h-[80px] resize-none"
            />
          </div>

          <Button
            type="submit"
            disabled={!description.trim() || isCreating || isLoading}
            className="w-full"
          >
            {isCreating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Plus className="mr-2 h-4 w-4" />
                Create Agent
              </>
            )}
          </Button>
        </form>

        {/* Error Messages */}
        {(createError || error || (connectionState === 'error' && connectionError)) && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {createError || error?.message || connectionError}
            </AlertDescription>
          </Alert>
        )}

        {/* Agent List */}
        <div className="flex-1 overflow-hidden">
          <h3 className="text-sm font-medium mb-2">Your Agents</h3>

          {agents.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p className="text-sm">No agents created yet</p>
              <p className="text-xs mt-1">Create your first agent to get started</p>
            </div>
          ) : (
            <ScrollArea className="h-full">
              <div className="space-y-2">
                {agents.map((agent) => (
                  <div
                    key={agent.id}
                    className="p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                  >
                    {editingAgent === agent.id ? (
                      <div className="space-y-2">
                        <Input
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleSaveEdit();
                            if (e.key === "Escape") setEditingAgent(null);
                          }}
                          autoFocus
                        />
                        <div className="flex gap-2">
                          <Button size="sm" onClick={handleSaveEdit} disabled={!editName.trim()}>
                            Save
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => setEditingAgent(null)}>
                            Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <div
                              data-testid={`agent-status-${agent.id}`}
                              className={cn("w-2 h-2 rounded-full", getStatusColor(agent.status))}
                            />
                            <h4 className="font-medium text-sm">{agent.name}</h4>
                            <Badge variant={getStatusBadgeVariant(agent.status)}>
                              {agent.status}
                            </Badge>
                          </div>
                          {agent.description && (
                            <p className="text-xs text-muted-foreground mt-1">
                              {agent.description}
                            </p>
                          )}
                        </div>

                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleEditAgent(agent)}
                            aria-label={`Edit ${agent.name}`}
                          >
                            <Edit2 className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDeleteAgent(agent.id)}
                            disabled={agent.status === "active"}
                            aria-label={`Delete ${agent.name}`}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
