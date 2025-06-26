"use client";
import { Button } from "@/components/ui/button";
import type React from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import type { Agent } from "@/lib/types";
import type { LLMSettings } from "@/lib/llm-settings";
import {
  Plus,
  Trash,
  UserPlus,
  UserMinus,
  Power,
  PowerOff,
  Download,
  Upload,
  AlertCircle,
} from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useState, useRef } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Spinner } from "@/components/ui/spinner";
import JSZip from "jszip";

interface AgentListProps {
  agents: Agent[];
  selectedAgent: Agent | null;
  onSelectAgent: (agent: Agent) => void;
  onCreateAgent: () => void;
  onCreateAgentWithName: (name: string) => void;
  onDeleteAgent: (agentId: string) => void;
  onAddToConversation: (agentId: string) => void;
  onRemoveFromConversation: (agentId: string) => void;
  onUpdateAgentColor: (agentId: string, color: string) => void;
  onToggleAutonomy: (agentId: string, enabled: boolean) => void;
  onExportAgents: (
    agentIds: string[],
    options: {
      includeSettings: boolean;
      includeApiKeys: boolean;
      includeConversations: boolean; // New option
    },
  ) => void;
  onImportAgents: (
    file: File,
    options: {
      mode: "replace" | "new" | "merge" | "settings-only";
      importSettings: boolean;
      importApiKeys: boolean;
      importConversations: boolean; // New option
    },
  ) => void;
  activeConversation: boolean;
  llmSettings?: LLMSettings;
}

// Predefined color palette
const colorPalette = [
  "#4f46e5", // indigo
  "#10b981", // emerald
  "#ef4444", // red
  "#f59e0b", // amber
  "#6366f1", // violet
  "#ec4899", // pink
  "#8b5cf6", // purple
  "#06b6d4", // cyan
  "#84cc16", // lime
  "#f97316", // orange
  "#14b8a6", // teal
  "#8b5cf6", // purple
];

export default function AgentList({
  agents,
  selectedAgent,
  onSelectAgent,
  onCreateAgent,
  onCreateAgentWithName,
  onDeleteAgent,
  onAddToConversation,
  onRemoveFromConversation,
  onUpdateAgentColor,
  onToggleAutonomy,
  onExportAgents,
  onImportAgents,
  activeConversation,
  llmSettings,
}: AgentListProps) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newAgentName, setNewAgentName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // State for export dialog
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const [selectedAgentsForExport, setSelectedAgentsForExport] = useState<
    Record<string, boolean>
  >({});
  const [exportAllAgents, setExportAllAgents] = useState(true);
  const [includeSettings, setIncludeSettings] = useState(false);
  const [includeApiKeys, setIncludeApiKeys] = useState(false);
  const [includeConversations, setIncludeConversations] = useState(false); // New state for conversations

  // State for import dialog
  const [isImportDialogOpen, setIsImportDialogOpen] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importMode, setImportMode] = useState<
    "replace" | "new" | "merge" | "settings-only"
  >("new");
  const [importSettings, setImportSettings] = useState(true);
  const [importApiKeys, setImportApiKeys] = useState(false);
  const [importConversations, setImportConversations] = useState(false); // New state for conversations
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [hasSettingsInImport, setHasSettingsInImport] = useState(false);
  const [hasApiKeysInImport, setHasApiKeysInImport] = useState(false);
  const [hasConversationsInImport, setHasConversationsInImport] =
    useState(false); // New state for conversations

  // Handle export dialog open
  const handleExportClick = () => {
    // Initialize with all agents selected
    const initialSelection = agents.reduce(
      (acc, agent) => {
        acc[agent.id] = true;
        return acc;
      },
      {} as Record<string, boolean>,
    );
    setSelectedAgentsForExport(initialSelection);
    setExportAllAgents(true);
    setIncludeSettings(false);
    setIncludeApiKeys(false);
    setIncludeConversations(false); // Initialize conversation export option
    setIsExportDialogOpen(true);
  };

  // Handle export confirmation
  const handleExportConfirm = () => {
    let agentIdsToExport: string[] = [];

    if (exportAllAgents) {
      // Export all agents
      agentIdsToExport = agents.map((agent) => agent.id);
    } else {
      // Export only selected agents
      agentIdsToExport = Object.entries(selectedAgentsForExport)
        .filter(([_, isSelected]) => isSelected)
        .map(([agentId]) => agentId);
    }

    onExportAgents(agentIdsToExport, {
      includeSettings,
      includeApiKeys: includeSettings && includeApiKeys,
      includeConversations, // Pass the new option
    });
    setIsExportDialogOpen(false);
  };

  // Toggle selection of an agent for export
  const toggleAgentSelection = (agentId: string) => {
    setSelectedAgentsForExport((prev) => ({
      ...prev,
      [agentId]: !prev[agentId],
    }));
  };

  // Handle import button click
  const handleImportClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle file selection
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImportFile(file);
      setImportError(null);

      // Check if the file contains settings and conversations
      try {
        const zip = new JSZip();
        const zipContent = await zip.loadAsync(file);

        // Check if settings.json exists in the zip
        const hasSettings = !!zipContent.files["settings.json"];
        setHasSettingsInImport(hasSettings);

        // Check if conversations folder exists in the zip
        const hasConversations = Object.keys(zipContent.files).some(
          (path) =>
            path.startsWith("conversations/") && path !== "conversations/",
        );
        setHasConversationsInImport(hasConversations);
        console.log("Import file check - Has conversations:", hasConversations);

        // If settings exist, check if they contain API keys
        if (hasSettings) {
          const settingsJSON =
            await zipContent.files["settings.json"].async("string");
          const settings = JSON.parse(settingsJSON);
          setHasApiKeysInImport(
            !!settings.apiKey &&
              typeof settings.apiKey === "string" &&
              settings.apiKey.trim() !== "",
          );
        } else {
          setHasApiKeysInImport(false);
        }
      } catch (error) {
        console.error("Error checking zip contents:", error);
        setHasSettingsInImport(false);
        setHasApiKeysInImport(false);
        setHasConversationsInImport(false);
      }

      setImportSettings(true);
      setImportApiKeys(false);
      setImportConversations(false); // Initialize conversation import option
      setImportMode("new");
      setIsImportDialogOpen(true);
    }

    // Reset the input so the same file can be selected again
    if (e.target) {
      e.target.value = "";
    }
  };

  // Handle import confirmation
  const handleImportConfirm = () => {
    if (!importFile) return;

    setIsImporting(true);
    setImportError(null);

    try {
      // Pass the file and import options to the parent component
      onImportAgents(importFile, {
        mode: importMode,
        importSettings: importSettings && hasSettingsInImport,
        importApiKeys: importSettings && importApiKeys && hasApiKeysInImport,
        importConversations: importConversations && hasConversationsInImport, // Pass the new option
      });

      // Close the dialog after a short delay to allow the toast to show
      setTimeout(() => {
        setIsImportDialogOpen(false);
        setImportFile(null);
      }, 500);
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Unknown error occurred during import",
      );
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-border">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-bold text-white">Agents</h2>
          <div className="flex gap-2">
            <Button
              onClick={handleImportClick}
              size="sm"
              variant="outline"
              className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <Upload size={16} className="mr-1" />
              Import
            </Button>
            <Button
              onClick={handleExportClick}
              size="sm"
              variant="outline"
              className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <Download size={16} className="mr-1" />
              Export
            </Button>

            {/* Hidden file input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".zip"
              className="hidden"
            />
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-2">
          {agents.map((agent) => (
            <div
              key={agent.id}
              className={`p-3 rounded-md border cursor-pointer transition-colors ${
                selectedAgent?.id === agent.id
                  ? "border-primary bg-primary/10"
                  : "border-border hover:bg-muted"
              }`}
              onClick={() => onSelectAgent(agent)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Popover>
                    <PopoverTrigger asChild>
                      <div
                        className="w-4 h-4 rounded-full cursor-pointer hover:ring-2 hover:ring-offset-2 hover:ring-primary transition-all"
                        style={{ backgroundColor: agent.color }}
                        onClick={(e) => e.stopPropagation()}
                        title="Change agent color"
                      />
                    </PopoverTrigger>
                    <PopoverContent
                      className="w-64"
                      align="start"
                      side="right"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="space-y-2">
                        <h4 className="font-medium text-sm">
                          Select agent color
                        </h4>
                        <div className="grid grid-cols-6 gap-2">
                          {colorPalette.map((color) => (
                            <div
                              key={color}
                              className={`w-6 h-6 rounded-full cursor-pointer hover:scale-110 transition-transform ${
                                agent.color === color
                                  ? "ring-2 ring-primary ring-offset-2"
                                  : ""
                              }`}
                              style={{ backgroundColor: color }}
                              onClick={() =>
                                onUpdateAgentColor(agent.id, color)
                              }
                            />
                          ))}
                        </div>
                        <div className="pt-2">
                          <label
                            htmlFor={`custom-color-${agent.id}`}
                            className="text-xs text-muted-foreground"
                          >
                            Custom color:
                          </label>
                          <input
                            id={`custom-color-${agent.id}`}
                            type="color"
                            value={agent.color}
                            onChange={(e) =>
                              onUpdateAgentColor(agent.id, e.target.value)
                            }
                            className="w-full h-8 mt-1 cursor-pointer"
                          />
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                  <span
                    className="font-medium text-white"
                    title="Select agent to edit name and details"
                  >
                    {agent.name}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {/* Autonomy status indicator */}
                  <div
                    className={`w-3 h-3 rounded-full ${agent.autonomyEnabled ? "bg-green-500" : "bg-gray-300"}`}
                    title={
                      agent.autonomyEnabled
                        ? "Autonomy enabled"
                        : "Autonomy disabled"
                    }
                  />
                  {/* Conversation status indicator */}
                  <div
                    className={`w-3 h-3 rounded-full ${agent.inConversation ? "bg-green-500" : "bg-gray-300"}`}
                    title={
                      agent.inConversation
                        ? "In conversation"
                        : "Not in conversation"
                    }
                  />
                </div>
              </div>
              <div className="mt-2 flex justify-between">
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteAgent(agent.id);
                  }}
                >
                  <Trash size={14} />
                </Button>

                {/* Autonomy toggle button */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleAutonomy(agent.id, !agent.autonomyEnabled);
                  }}
                  className={`${
                    agent.autonomyEnabled
                      ? "bg-green-900/50 border-green-500 text-white hover:bg-green-800 hover:text-white"
                      : "bg-gray-900/50 border-gray-500 text-white hover:bg-gray-800 hover:text-white"
                  }`}
                >
                  {agent.autonomyEnabled ? (
                    <>
                      <Power size={14} className="mr-1" />
                      Auto On
                    </>
                  ) : (
                    <>
                      <PowerOff size={14} className="mr-1" />
                      Auto Off
                    </>
                  )}
                </Button>

                {agent.inConversation ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveFromConversation(agent.id);
                    }}
                    className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <UserMinus size={14} className="mr-1" />
                    Remove
                  </Button>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={
                      !activeConversation &&
                      agents.some((a) => a.inConversation)
                    }
                    onClick={(e) => {
                      e.stopPropagation();
                      onAddToConversation(agent.id);
                    }}
                    className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
                  >
                    <UserPlus size={14} className="mr-1" />
                    Add
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="p-4 border-t border-border">
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <Button onClick={() => setIsDialogOpen(true)} className="w-full">
            <Plus size={16} className="mr-2" />
            Create Agent
          </Button>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Agent</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <label
                htmlFor="new-agent-name"
                className="text-sm font-medium block mb-2"
              >
                Agent Name
              </label>
              <Input
                id="new-agent-name"
                value={newAgentName}
                onChange={(e) => setNewAgentName(e.target.value)}
                placeholder="Enter agent name..."
                autoFocus
              />
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">Cancel</Button>
              </DialogClose>
              <Button
                onClick={() => {
                  if (newAgentName.trim()) {
                    onCreateAgentWithName(newAgentName.trim());
                    setNewAgentName("");
                    setIsDialogOpen(false);
                  }
                }}
              >
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Export Dialog */}
      <Dialog open={isExportDialogOpen} onOpenChange={setIsExportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export Agents</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <div className="mb-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="export-all"
                  checked={exportAllAgents}
                  onCheckedChange={(checked) => {
                    setExportAllAgents(checked === true);
                  }}
                />
                <Label htmlFor="export-all">Export all agents</Label>
              </div>
            </div>

            {!exportAllAgents && (
              <div className="space-y-2">
                <p className="text-sm font-medium mb-2">
                  Select agents to export:
                </p>
                {agents.map((agent) => (
                  <div key={agent.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-agent-${agent.id}`}
                      checked={selectedAgentsForExport[agent.id] || false}
                      onCheckedChange={() => toggleAgentSelection(agent.id)}
                    />
                    <Label
                      htmlFor={`export-agent-${agent.id}`}
                      className="flex items-center"
                    >
                      <div
                        className="w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: agent.color }}
                      />
                      {agent.name}
                    </Label>
                  </div>
                ))}
              </div>
            )}

            {/* Export options */}
            <div className="mt-6 pt-4 border-t">
              {/* Settings export option */}
              <div className="flex items-center space-x-2 mb-3">
                <Checkbox
                  id="include-settings"
                  checked={includeSettings}
                  onCheckedChange={(checked) => {
                    setIncludeSettings(checked === true);
                    if (checked === false) {
                      setIncludeApiKeys(false);
                    }
                  }}
                />
                <Label htmlFor="include-settings">Include settings</Label>
              </div>

              {includeSettings && (
                <div className="ml-6 mb-3">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="include-api-keys"
                      checked={includeApiKeys}
                      onCheckedChange={(checked) =>
                        setIncludeApiKeys(checked === true)
                      }
                    />
                    <Label
                      htmlFor="include-api-keys"
                      className="flex items-center"
                    >
                      Include API keys
                      <span className="text-red-500 ml-1">*</span>
                    </Label>
                  </div>
                  {includeApiKeys && (
                    <p className="text-xs text-red-500 mt-1">
                      Warning: API keys are sensitive information. Only export
                      them if you&apos;re sure about security.
                    </p>
                  )}
                </div>
              )}

              {/* Conversations export option */}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="include-conversations"
                  checked={includeConversations}
                  onCheckedChange={(checked) =>
                    setIncludeConversations(checked === true)
                  }
                />
                <Label htmlFor="include-conversations">
                  Include conversation histories
                </Label>
              </div>
            </div>
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <Button
              onClick={handleExportConfirm}
              disabled={
                !exportAllAgents &&
                Object.values(selectedAgentsForExport).filter(Boolean)
                  .length === 0
              }
            >
              <Download size={16} className="mr-2" />
              Export
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Import Dialog */}
      <Dialog
        open={isImportDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            setImportFile(null);
            setImportError(null);
          }
          setIsImportDialogOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Import Agents</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            {importFile && (
              <div className="mb-4">
                <p className="text-sm font-medium">Selected file:</p>
                <p className="text-sm">
                  {importFile.name} ({(importFile.size / 1024).toFixed(1)} KB)
                </p>
              </div>
            )}

            {/* Settings import options */}
            {hasSettingsInImport && (
              <div className="mb-4 pb-4 border-b">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="import-settings"
                    checked={importSettings}
                    onCheckedChange={(checked) => {
                      setImportSettings(checked === true);
                      if (checked === false) {
                        setImportApiKeys(false);
                      }
                    }}
                  />
                  <Label htmlFor="import-settings">Import settings</Label>
                </div>

                {importSettings && (
                  <div className="ml-6 mt-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="import-api-keys"
                        checked={importApiKeys}
                        onCheckedChange={(checked) =>
                          setImportApiKeys(checked === true)
                        }
                      />
                      <Label htmlFor="import-api-keys">Import API keys</Label>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Conversations import option */}
            <div className="mb-4 pb-4 border-b">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="import-conversations"
                  checked={importConversations}
                  onCheckedChange={(checked) =>
                    setImportConversations(checked === true)
                  }
                  disabled={!hasConversationsInImport}
                />
                <Label
                  htmlFor="import-conversations"
                  className={
                    !hasConversationsInImport ? "text-muted-foreground" : ""
                  }
                >
                  Import conversation histories
                  {!hasConversationsInImport && (
                    <span className="ml-2 text-xs text-muted-foreground">
                      (No conversations found in import file)
                    </span>
                  )}
                </Label>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium mb-2">Import mode:</p>
                <RadioGroup
                  value={importMode}
                  onValueChange={(value) =>
                    setImportMode(
                      value as "replace" | "new" | "merge" | "settings-only",
                    )
                  }
                >
                  {hasSettingsInImport && (
                    <div className="flex items-center space-x-2 mb-2">
                      <RadioGroupItem
                        value="settings-only"
                        id="import-settings-only"
                      />
                      <Label htmlFor="import-settings-only">
                        Import settings only (no agents)
                      </Label>
                    </div>
                  )}
                  <div className="flex items-center space-x-2 mb-2">
                    <RadioGroupItem value="new" id="import-new" />
                    <Label htmlFor="import-new">Import as new agents</Label>
                  </div>
                  <div className="flex items-center space-x-2 mb-2">
                    <RadioGroupItem value="replace" id="import-replace" />
                    <Label htmlFor="import-replace">
                      Replace existing agents with same ID
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="merge" id="import-merge" />
                    <Label htmlFor="import-merge">
                      Merge knowledge with existing agents
                    </Label>
                  </div>
                </RadioGroup>
              </div>
            </div>

            {importError && (
              <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded-md flex items-start">
                <AlertCircle
                  className="text-red-500 mr-2 mt-0.5 flex-shrink-0"
                  size={16}
                />
                <p className="text-sm text-red-500">{importError}</p>
              </div>
            )}
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <Button
              onClick={handleImportConfirm}
              disabled={!importFile || isImporting}
            >
              {isImporting ? (
                <>
                  <Spinner size={16} className="mr-2" />
                  Importing...
                </>
              ) : (
                <>
                  <Upload size={16} className="mr-2" />
                  Import
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
