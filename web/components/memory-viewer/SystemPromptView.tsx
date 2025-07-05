import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Edit3, Save, X, FileText } from "lucide-react";
import type { Agent } from "@/lib/types";

interface SystemPromptViewProps {
  selectedAgent: Agent | null;
  systemPrompt: string;
  editingSystemPrompt: boolean;
  onSystemPromptChange: (value: string) => void;
  onSaveSystemPrompt: () => void;
  onEditSystemPrompt: (editing: boolean) => void;
}

export function SystemPromptView({
  selectedAgent,
  systemPrompt,
  editingSystemPrompt,
  onSystemPromptChange,
  onSaveSystemPrompt,
  onEditSystemPrompt,
}: SystemPromptViewProps) {
  if (!selectedAgent) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent>
          <p className="text-center text-muted-foreground">
            Select an agent to view their system prompt
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardContent className="p-6 h-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <FileText className="h-5 w-5" />
            System Prompt
          </h3>
          {!editingSystemPrompt ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onEditSystemPrompt(true)}
            >
              <Edit3 className="h-4 w-4 mr-2" />
              Edit
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onEditSystemPrompt(false);
                  onSystemPromptChange(selectedAgent.systemPrompt || "");
                }}
              >
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button size="sm" onClick={onSaveSystemPrompt}>
                <Save className="h-4 w-4 mr-2" />
                Save System Prompt
              </Button>
            </div>
          )}
        </div>

        <div className="mb-4 p-3 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            The system prompt defines the agent&apos;s behavior, personality,
            and capabilities. It is used as the initial context for all
            interactions.
          </p>
        </div>

        {editingSystemPrompt ? (
          <Textarea
            value={systemPrompt}
            onChange={(e) => onSystemPromptChange(e.target.value)}
            placeholder="Enter system prompt..."
            className="flex-1 min-h-[350px] resize-none font-mono text-sm"
          />
        ) : (
          <div className="flex-1 overflow-y-auto">
            <pre className="text-sm font-mono whitespace-pre-wrap bg-muted p-4 rounded-lg">
              {systemPrompt || "No system prompt configured for this agent."}
            </pre>
          </div>
        )}

        {/* Prompt statistics */}
        <div className="mt-4 pt-4 border-t">
          <div className="flex items-center justify-between text-sm">
            <div className="space-y-1">
              <p className="text-muted-foreground">Character count:</p>
              <p className="font-medium">{systemPrompt.length} characters</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Word count:</p>
              <p className="font-medium">
                {systemPrompt.split(/\s+/).filter(Boolean).length} words
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Line count:</p>
              <p className="font-medium">
                {systemPrompt.split("\n").length} lines
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
