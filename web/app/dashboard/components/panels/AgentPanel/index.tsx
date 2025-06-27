"use client";

import React, { useState } from "react";
import { useAppSelector } from "@/store/hooks";
import { DashboardView } from "../../../page";
import { Users, Plus, Settings, Activity, Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import AgentTemplateSelector from "@/components/dashboard/AgentTemplateSelector";

interface AgentPanelProps {
  view: DashboardView;
}

export default function AgentPanel({ view }: AgentPanelProps) {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [showTemplateSelector, setShowTemplateSelector] = useState(false);

  // Redux state
  const agents = useAppSelector((state) => state.agents?.agents) || {};
  const agentOrder = useAppSelector((state) => state.agents?.agentOrder) || [];

  const getStatusColor = (status: string) => {
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

  const getStatusText = (status: string) => {
    switch (status) {
      case "active":
        return "ACTIVE";
      case "idle":
        return "IDLE";
      case "error":
        return "ERROR";
      default:
        return "UNKNOWN";
    }
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Panel Header */}
      <div className="flex items-center justify-between p-4 border-b border-[var(--bg-tertiary)]">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-[var(--accent-primary)]" />
          <h3 className="font-semibold text-[var(--text-primary)]">
            Agent Management
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs">
            {Object.keys(agents).length} Total
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setShowTemplateSelector(true)}
          >
            <Plus className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Agent List */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {agentOrder.length === 0 ? (
            <div className="text-center py-8">
              <Users className="w-12 h-12 text-[var(--text-tertiary)] mx-auto mb-3" />
              <p className="text-[var(--text-secondary)] text-sm">
                No agents configured
              </p>
              <Button
                size="sm"
                className="mt-3"
                onClick={() => setShowTemplateSelector(true)}
              >
                <Plus className="w-4 h-4 mr-2" />
                Create Agent
              </Button>
            </div>
          ) : (
            agentOrder.map((agentId) => {
              const agent = agents[agentId];
              if (!agent) return null;

              return (
                <div
                  key={agentId}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedAgent === agentId
                      ? "border-[var(--accent-primary)] bg-[var(--bg-secondary)]"
                      : "border-[var(--bg-tertiary)] bg-[var(--bg-secondary)] hover:border-[var(--bg-quaternary)]"
                  }`}
                  onClick={() =>
                    setSelectedAgent(selectedAgent === agentId ? null : agentId)
                  }
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)}`}
                      />
                      <span className="font-medium text-[var(--text-primary)] text-sm">
                        {agent.name || agentId}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Badge variant="outline" className="text-xs">
                        {getStatusText(agent.status)}
                      </Badge>
                      <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                        {agent.status === "active" ? (
                          <Pause className="w-3 h-3" />
                        ) : (
                          <Play className="w-3 h-3" />
                        )}
                      </Button>
                    </div>
                  </div>

                  <div className="text-xs text-[var(--text-secondary)]">
                    Template: {agent.templateId || "Unknown"}
                  </div>

                  {agent.biography && (
                    <div className="text-xs text-[var(--text-tertiary)] mt-1 line-clamp-2">
                      {agent.biography}
                    </div>
                  )}

                  {selectedAgent === agentId && (
                    <div className="mt-3 pt-3 border-t border-[var(--bg-tertiary)] space-y-2">
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-[var(--text-tertiary)]">
                            Messages:
                          </span>
                          <span className="ml-1 text-[var(--text-secondary)]">
                            {agent.activityMetrics?.messagesCount || 0}
                          </span>
                        </div>
                        <div>
                          <span className="text-[var(--text-tertiary)]">
                            Active:
                          </span>
                          <span className="ml-1 text-[var(--text-secondary)]">
                            {agent.inConversation ? "Yes" : "No"}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-xs h-6"
                        >
                          Configure
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-xs h-6"
                        >
                          Logs
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-xs h-6"
                        >
                          Reset
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>

      {/* Panel Footer */}
      <div className="p-4 border-t border-[var(--bg-tertiary)]">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-[var(--text-secondary)]">
                {
                  Object.values(agents).filter(
                    (a: any) => a.status === "active",
                  ).length
                }{" "}
                Active
              </span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-yellow-500" />
              <span className="text-[var(--text-secondary)]">
                {
                  Object.values(agents).filter((a: any) => a.status === "idle")
                    .length
                }{" "}
                Idle
              </span>
            </div>
          </div>
          <Button size="sm" variant="ghost" className="text-xs">
            <Settings className="w-3 h-3 mr-1" />
            Manage All
          </Button>
        </div>
      </div>

      {/* Agent Template Selector Modal */}
      <Dialog
        open={showTemplateSelector}
        onOpenChange={setShowTemplateSelector}
      >
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]">
          <DialogHeader>
            <DialogTitle className="text-[var(--text-primary)]">
              Create New Agent
            </DialogTitle>
          </DialogHeader>
          <AgentTemplateSelector />
        </DialogContent>
      </Dialog>
    </div>
  );
}
