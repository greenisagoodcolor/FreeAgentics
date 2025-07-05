import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Edit3, Save, X, Settings2 } from "lucide-react";
import type { Agent } from "@/lib/types";
import type { AgentToolPermissions } from "@/types/memory-viewer";

interface ToolsViewProps {
  selectedAgent: Agent | null;
  toolPermissions: AgentToolPermissions;
  editingTools: boolean;
  onToolPermissionChange: (
    tool: keyof AgentToolPermissions,
    value: boolean,
  ) => void;
  onSaveTools: () => void;
  onEditTools: (editing: boolean) => void;
}

interface ToolCategory {
  name: string;
  tools: Array<{
    key: keyof AgentToolPermissions;
    label: string;
    description: string;
  }>;
}

const toolCategories: ToolCategory[] = [
  {
    name: "Information Access Tools",
    tools: [
      {
        key: "internetSearch",
        label: "Internet Search",
        description: "Search the web for information",
      },
      {
        key: "webScraping",
        label: "Web Scraping",
        description: "Extract data from websites",
      },
      {
        key: "wikipediaAccess",
        label: "Wikipedia Access",
        description: "Access Wikipedia articles",
      },
      {
        key: "newsApi",
        label: "News API",
        description: "Access current news articles",
      },
      {
        key: "academicSearch",
        label: "Academic Search",
        description: "Search academic papers",
      },
      {
        key: "documentRetrieval",
        label: "Document Retrieval",
        description: "Access stored documents",
      },
    ],
  },
  {
    name: "Content Generation & Processing",
    tools: [
      {
        key: "imageGeneration",
        label: "Image Generation",
        description: "Create images from text",
      },
      {
        key: "textSummarization",
        label: "Text Summarization",
        description: "Summarize long texts",
      },
      {
        key: "translation",
        label: "Translation",
        description: "Translate between languages",
      },
      {
        key: "codeExecution",
        label: "Code Execution",
        description: "Run code snippets",
      },
    ],
  },
  {
    name: "Knowledge & Reasoning Tools",
    tools: [
      {
        key: "calculator",
        label: "Calculator",
        description: "Perform calculations",
      },
      {
        key: "knowledgeGraphQuery",
        label: "Knowledge Graph Query",
        description: "Query knowledge graphs",
      },
      {
        key: "factChecking",
        label: "Fact Checking",
        description: "Verify facts and claims",
      },
      {
        key: "timelineGenerator",
        label: "Timeline Generator",
        description: "Create timelines of events",
      },
    ],
  },
  {
    name: "External Integrations",
    tools: [
      {
        key: "weatherData",
        label: "Weather Data",
        description: "Access weather information",
      },
      {
        key: "mapLocationData",
        label: "Map & Location Data",
        description: "Access geographic data",
      },
      {
        key: "financialData",
        label: "Financial Data",
        description: "Access financial markets data",
      },
      {
        key: "publicDatasets",
        label: "Public Datasets",
        description: "Access public data sources",
      },
    ],
  },
  {
    name: "Agent-Specific Tools",
    tools: [
      {
        key: "memorySearch",
        label: "Memory Search",
        description: "Search agent's memory",
      },
      {
        key: "crossAgentKnowledge",
        label: "Cross-Agent Knowledge",
        description: "Access other agents' knowledge",
      },
      {
        key: "conversationAnalysis",
        label: "Conversation Analysis",
        description: "Analyze conversations",
      },
    ],
  },
];

export function ToolsView({
  selectedAgent,
  toolPermissions,
  editingTools,
  onToolPermissionChange,
  onSaveTools,
  onEditTools,
}: ToolsViewProps) {
  if (!selectedAgent) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent>
          <p className="text-center text-muted-foreground">
            Select an agent to manage their tool permissions
          </p>
        </CardContent>
      </Card>
    );
  }

  const enabledCount = Object.values(toolPermissions).filter(Boolean).length;
  const totalCount = Object.keys(toolPermissions).length;

  return (
    <div className="h-full space-y-4">
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Settings2 className="h-5 w-5" />
              Tool Permissions
            </h3>
            <div className="flex items-center gap-4">
              <span className="text-sm text-muted-foreground">
                {enabledCount} of {totalCount} tools enabled
              </span>
              {!editingTools ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onEditTools(true)}
                >
                  <Edit3 className="h-4 w-4 mr-2" />
                  Edit
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onEditTools(false)}
                  >
                    <X className="h-4 w-4 mr-2" />
                    Cancel
                  </Button>
                  <Button size="sm" onClick={onSaveTools}>
                    <Save className="h-4 w-4 mr-2" />
                    Save Tool Settings
                  </Button>
                </div>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="space-y-4 max-h-[450px] overflow-y-auto pr-2">
        {toolCategories.map((category) => (
          <Card key={category.name}>
            <CardContent className="p-4">
              <h4 className="font-medium mb-3">{category.name}</h4>
              <div className="space-y-3">
                {category.tools.map((tool, toolIndex) => (
                  <div key={tool.key}>
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label
                          htmlFor={`tool-${tool.key}`}
                          className="text-sm font-medium"
                        >
                          {tool.label}
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          {tool.description}
                        </p>
                      </div>
                      <Switch
                        id={`tool-${tool.key}`}
                        checked={toolPermissions[tool.key]}
                        onCheckedChange={(checked) =>
                          onToolPermissionChange(tool.key, checked)
                        }
                        disabled={!editingTools}
                      />
                    </div>
                    {toolIndex < category.tools.length - 1 && (
                      <Separator className="my-3" />
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {editingTools && (
        <Card>
          <CardContent className="p-4">
            <p className="text-sm text-muted-foreground">
              <strong>Note:</strong> Tool permissions determine what
              capabilities this agent has access to. Disabling tools may limit
              the agent&apos;s functionality but can improve security and
              performance.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
