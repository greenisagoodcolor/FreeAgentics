"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { ScrollArea } from "./ui/scroll-area";
import { Separator } from "./ui/separator";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Slider } from "./ui/slider";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Checkbox } from "./ui/checkbox";
import { Calendar } from "./ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import {
  Settings,
  Save,
  Download,
  Upload,
  History,
  AlertTriangle,
  CheckCircle2,
  X,
  Filter,
  Search,
  Calendar as CalendarIcon,
  FileText,
  Database,
  Shield,
  Eye,
  Edit,
  Trash2,
  RotateCcw,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

// Import existing components and utilities
import { AgentTemplate, AGENT_TEMPLATES } from "./ui/agent-template-selector";
import {
  auditLogger,
  AuditLogEntry,
  AuditLogFilter,
  AuditLogStats,
  ExportOptions as AuditExportOptions,
  logBoundaryEdit,
  logTemplateSelection,
  logThresholdChange,
} from "@/lib/audit-logger";

// Boundary configuration interfaces
export interface BoundaryConfiguration {
  agentId: string;
  templateId: string;
  dimensions: {
    internal: {
      threshold: number;
      precision: number;
      adaptiveScaling: boolean;
      mathematicalConstraints: string[];
    };
    sensory: {
      threshold: number;
      precision: number;
      modalityWeights: Record<string, number>;
      noiseFiltering: boolean;
    };
    active: {
      threshold: number;
      precision: number;
      actionSpaceSize: number;
      policyConstraints: string[];
    };
    external: {
      threshold: number;
      precision: number;
      environmentComplexity: number;
      boundaryRigidity: number;
    };
  };
  monitoring: {
    enabled: boolean;
    alertThresholds: {
      warning: number;
      critical: number;
    };
    violationHandling: {
      autoMitigation: boolean;
      escalationRules: string[];
      notificationChannels: string[];
    };
  };
  compliance: {
    framework: string;
    auditingEnabled: boolean;
    retentionPeriod: number;
    encryptionRequired: boolean;
  };
  metadata: {
    createdAt: string;
    updatedAt: string;
    version: number;
    description?: string;
    tags: string[];
  };
}

export interface ConfigurationTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  configuration: Partial<BoundaryConfiguration>;
  isValidated: boolean;
  expertApproved: boolean;
  usageCount: number;
  lastUsed?: string;
}

interface MarkovBlanketConfigurationUIProps {
  agentId?: string;
  initialConfiguration?: BoundaryConfiguration;
  onConfigurationChange?: (config: BoundaryConfiguration) => void;
  onSave?: (config: BoundaryConfiguration) => Promise<void>;
  className?: string;
  readOnly?: boolean;
  showAuditLog?: boolean;
  showTemplateSelector?: boolean;
  enableExport?: boolean;
}

export const MarkovBlanketConfigurationUI: React.FC<
  MarkovBlanketConfigurationUIProps
> = ({
  agentId = "default",
  initialConfiguration,
  onConfigurationChange,
  onSave,
  className,
  readOnly = false,
  showAuditLog = true,
  showTemplateSelector = true,
  enableExport = true,
}) => {
  // Configuration state
  const [configuration, setConfiguration] = useState<BoundaryConfiguration>(
    initialConfiguration || getDefaultConfiguration(agentId),
  );
  const [selectedTemplate, setSelectedTemplate] =
    useState<AgentTemplate | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Audit log state
  const [auditEntries, setAuditEntries] = useState<AuditLogEntry[]>([]);
  const [auditStats, setAuditStats] = useState<AuditLogStats | null>(null);
  const [auditFilter, setAuditFilter] = useState<AuditLogFilter>({
    agentId,
    limit: 50,
  });
  const [isLoadingAudit, setIsLoadingAudit] = useState(false);

  // UI state
  const [activeTab, setActiveTab] = useState("boundaries");
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const [selectedAuditEntries, setSelectedAuditEntries] = useState<string[]>(
    [],
  );
  const [auditSearchText, setAuditSearchText] = useState("");
  const [dateRange, setDateRange] = useState<{ start?: Date; end?: Date }>({});

  // Load audit log data
  const loadAuditData = useCallback(async () => {
    if (!showAuditLog) return;

    setIsLoadingAudit(true);
    try {
      const [entries, stats] = await Promise.all([
        auditLogger.getEntries(auditFilter),
        auditLogger.getStats(),
      ]);
      setAuditEntries(entries);
      setAuditStats(stats);
    } catch (error) {
      console.error("Failed to load audit data:", error);
    } finally {
      setIsLoadingAudit(false);
    }
  }, [auditFilter, showAuditLog]);

  useEffect(() => {
    loadAuditData();
  }, [loadAuditData]);

  // Handle configuration changes
  const handleConfigurationChange = useCallback(
    (path: string, value: any, logChange: boolean = true) => {
      if (readOnly) return;

      setConfiguration((prev) => {
        const newConfig = { ...prev };
        const keys = path.split(".");
        let current: any = newConfig;

        // Navigate to the parent of the target property
        for (let i = 0; i < keys.length - 1; i++) {
          if (!current[keys[i]]) {
            current[keys[i]] = {};
          }
          current = current[keys[i]];
        }

        const oldValue = current[keys[keys.length - 1]];
        current[keys[keys.length - 1]] = value;

        // Update metadata
        newConfig.metadata.updatedAt = new Date().toISOString();
        newConfig.metadata.version += 1;

        // Log the change if requested
        if (logChange && oldValue !== value) {
          const description = `Updated ${path} from ${JSON.stringify(oldValue)} to ${JSON.stringify(value)}`;
          logBoundaryEdit(agentId, path, oldValue, value, description)
            .then(() => loadAuditData())
            .catch(console.error);
        }

        setHasUnsavedChanges(true);
        onConfigurationChange?.(newConfig);
        return newConfig;
      });
    },
    [agentId, readOnly, onConfigurationChange, loadAuditData],
  );

  // Handle template selection
  const handleTemplateSelection = useCallback(
    async (template: AgentTemplate) => {
      if (readOnly) return;

      setSelectedTemplate(template);

      // Apply template configuration
      const templateConfig = createConfigurationFromTemplate(template, agentId);
      setConfiguration(templateConfig);
      setHasUnsavedChanges(true);

      // Log template selection
      try {
        await logTemplateSelection(
          agentId,
          template.id,
          template,
          `Applied ${template.name} template to agent ${agentId}`,
        );
        await loadAuditData();
      } catch (error) {
        console.error("Failed to log template selection:", error);
      }

      onConfigurationChange?.(templateConfig);
    },
    [agentId, readOnly, onConfigurationChange, loadAuditData],
  );

  // Handle save
  const handleSave = useCallback(async () => {
    if (readOnly || !onSave) return;

    setIsSaving(true);
    try {
      await onSave(configuration);
      setHasUnsavedChanges(false);

      // Log save operation
      await auditLogger.logChange(
        "configuration_update",
        "agent",
        agentId,
        `Saved configuration for agent ${agentId}`,
        { after: configuration },
        { agentId },
        { riskLevel: "medium", requiresApproval: true },
      );

      await loadAuditData();
    } catch (error) {
      console.error("Failed to save configuration:", error);
    } finally {
      setIsSaving(false);
    }
  }, [configuration, agentId, onSave, readOnly, loadAuditData]);

  // Handle audit log export
  const handleExportAuditLog = useCallback(
    async (options: AuditExportOptions) => {
      try {
        const blob = await auditLogger.exportData({
          ...options,
          filters: auditFilter,
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `audit-log-${agentId}-${format(new Date(), "yyyy-MM-dd")}.${options.format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Log export operation
        await auditLogger.logChange(
          "export_data",
          "system",
          `audit_log_${agentId}`,
          `Exported audit log for agent ${agentId} in ${options.format} format`,
          {
            after: { format: options.format, entryCount: auditEntries.length },
          },
          { agentId },
          { riskLevel: "medium" },
        );

        setIsExportDialogOpen(false);
      } catch (error) {
        console.error("Failed to export audit log:", error);
      }
    },
    [auditFilter, agentId, auditEntries.length],
  );

  // Filtered audit entries for display
  const filteredAuditEntries = useMemo(() => {
    let filtered = auditEntries;

    if (auditSearchText) {
      const searchLower = auditSearchText.toLowerCase();
      filtered = filtered.filter(
        (entry) =>
          entry.description.toLowerCase().includes(searchLower) ||
          entry.operationType.toLowerCase().includes(searchLower) ||
          entry.entityId.toLowerCase().includes(searchLower),
      );
    }

    return filtered;
  }, [auditEntries, auditSearchText]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Markov Blanket Configuration</h2>
          <p className="text-muted-foreground">
            Configure agent boundaries, select templates, and review audit logs
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {hasUnsavedChanges && (
            <Badge
              variant="outline"
              className="text-orange-600 border-orange-200"
            >
              Unsaved Changes
            </Badge>
          )}

          {!readOnly && (
            <Button
              onClick={handleSave}
              disabled={!hasUnsavedChanges || isSaving}
              className="flex items-center space-x-2"
            >
              <Save className="h-4 w-4" />
              <span>{isSaving ? "Saving..." : "Save Configuration"}</span>
            </Button>
          )}
        </div>
      </div>

      {/* Main tabs */}
      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="space-y-4"
      >
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger
            value="boundaries"
            className="flex items-center space-x-2"
          >
            <Settings className="h-4 w-4" />
            <span>Boundaries</span>
          </TabsTrigger>
          {showTemplateSelector && (
            <TabsTrigger
              value="templates"
              className="flex items-center space-x-2"
            >
              <FileText className="h-4 w-4" />
              <span>Templates</span>
            </TabsTrigger>
          )}
          {showAuditLog && (
            <TabsTrigger value="audit" className="flex items-center space-x-2">
              <History className="h-4 w-4" />
              <span>Audit Log</span>
            </TabsTrigger>
          )}
          <TabsTrigger
            value="compliance"
            className="flex items-center space-x-2"
          >
            <Shield className="h-4 w-4" />
            <span>Compliance</span>
          </TabsTrigger>
        </TabsList>

        {/* Boundary Configuration Tab */}
        <TabsContent value="boundaries" className="space-y-6">
          <BoundaryConfigurationPanel
            configuration={configuration}
            onChange={handleConfigurationChange}
            readOnly={readOnly}
            showAdvanced={showAdvancedSettings}
            onShowAdvancedChange={setShowAdvancedSettings}
          />
        </TabsContent>

        {/* Template Selection Tab */}
        {showTemplateSelector && (
          <TabsContent value="templates" className="space-y-6">
            <TemplateSelectionPanel
              selectedTemplate={selectedTemplate}
              onTemplateSelect={handleTemplateSelection}
              readOnly={readOnly}
            />
          </TabsContent>
        )}

        {/* Audit Log Tab */}
        {showAuditLog && (
          <TabsContent value="audit" className="space-y-6">
            <AuditLogPanel
              entries={filteredAuditEntries}
              stats={auditStats}
              filter={auditFilter}
              onFilterChange={setAuditFilter}
              searchText={auditSearchText}
              onSearchTextChange={setAuditSearchText}
              selectedEntries={selectedAuditEntries}
              onSelectedEntriesChange={setSelectedAuditEntries}
              isLoading={isLoadingAudit}
              onExport={enableExport ? handleExportAuditLog : undefined}
              onRefresh={loadAuditData}
            />
          </TabsContent>
        )}

        {/* Compliance Tab */}
        <TabsContent value="compliance" className="space-y-6">
          <CompliancePanel
            configuration={configuration}
            onChange={handleConfigurationChange}
            auditStats={auditStats}
            readOnly={readOnly}
          />
        </TabsContent>
      </Tabs>

      {/* Export Dialog */}
      <ExportDialog
        isOpen={isExportDialogOpen}
        onClose={() => setIsExportDialogOpen(false)}
        onExport={handleExportAuditLog}
        entryCount={filteredAuditEntries.length}
      />
    </div>
  );
};

// Helper functions
function getDefaultConfiguration(agentId: string): BoundaryConfiguration {
  return {
    agentId,
    templateId: "default",
    dimensions: {
      internal: {
        threshold: 0.8,
        precision: 16.0,
        adaptiveScaling: true,
        mathematicalConstraints: ["stochastic_matrix", "probability_simplex"],
      },
      sensory: {
        threshold: 0.8,
        precision: 16.0,
        modalityWeights: { visual: 0.4, auditory: 0.3, tactile: 0.3 },
        noiseFiltering: true,
      },
      active: {
        threshold: 0.8,
        precision: 16.0,
        actionSpaceSize: 8,
        policyConstraints: ["action_bounds", "energy_conservation"],
      },
      external: {
        threshold: 0.8,
        precision: 16.0,
        environmentComplexity: 0.5,
        boundaryRigidity: 0.7,
      },
    },
    monitoring: {
      enabled: true,
      alertThresholds: {
        warning: 0.7,
        critical: 0.5,
      },
      violationHandling: {
        autoMitigation: false,
        escalationRules: ["notify_admin", "log_violation"],
        notificationChannels: ["dashboard", "email"],
      },
    },
    compliance: {
      framework: "ADR-011",
      auditingEnabled: true,
      retentionPeriod: 365,
      encryptionRequired: true,
    },
    metadata: {
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      version: 1,
      tags: [],
    },
  };
}

function createConfigurationFromTemplate(
  template: AgentTemplate,
  agentId: string,
): BoundaryConfiguration {
  const defaultConfig = getDefaultConfiguration(agentId);

  return {
    ...defaultConfig,
    templateId: template.id,
    dimensions: {
      ...defaultConfig.dimensions,
      internal: {
        ...defaultConfig.dimensions.internal,
        precision: template.mathematicalFoundation.defaultPrecision.state,
      },
      sensory: {
        ...defaultConfig.dimensions.sensory,
        precision: template.mathematicalFoundation.defaultPrecision.sensory,
      },
      active: {
        ...defaultConfig.dimensions.active,
        precision: template.mathematicalFoundation.defaultPrecision.policy,
        actionSpaceSize: template.mathematicalFoundation.actionSpaces,
      },
    },
    metadata: {
      ...defaultConfig.metadata,
      description: `Configuration based on ${template.name} template`,
    },
  };
}

// Sub-components (to be implemented in separate files or as part of this component)
const BoundaryConfigurationPanel: React.FC<{
  configuration: BoundaryConfiguration;
  onChange: (path: string, value: any) => void;
  readOnly: boolean;
  showAdvanced: boolean;
  onShowAdvancedChange: (show: boolean) => void;
}> = ({
  configuration,
  onChange,
  readOnly,
  showAdvanced,
  onShowAdvancedChange,
}) => {
  return (
    <div className="space-y-6">
      {/* Advanced settings toggle */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Boundary Dimensions</h3>
        <div className="flex items-center space-x-2">
          <Switch
            checked={showAdvanced}
            onCheckedChange={onShowAdvancedChange}
            disabled={readOnly}
          />
          <Label>Advanced Settings</Label>
        </div>
      </div>

      {/* Dimension configurations */}
      {Object.entries(configuration.dimensions).map(([dimension, config]) => (
        <Card key={dimension}>
          <CardHeader>
            <CardTitle className="capitalize">{dimension} Boundary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Threshold</Label>
                <Slider
                  value={[config.threshold]}
                  onValueChange={([value]) =>
                    onChange(`dimensions.${dimension}.threshold`, value)
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={readOnly}
                  className="w-full"
                />
                <div className="text-sm text-muted-foreground">
                  Current: {config.threshold.toFixed(2)}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Precision</Label>
                <Input
                  type="number"
                  value={config.precision}
                  onChange={(e) =>
                    onChange(
                      `dimensions.${dimension}.precision`,
                      parseFloat(e.target.value),
                    )
                  }
                  disabled={readOnly}
                  min={0.1}
                  max={1000}
                  step={0.1}
                />
              </div>
            </div>

            {showAdvanced && (
              <div className="space-y-4 pt-4 border-t">
                {/* Dimension-specific advanced settings */}
                {dimension === "sensory" && (
                  <div className="space-y-2">
                    <Label>Modality Weights</Label>
                    {Object.entries(
                      "modalityWeights" in config
                        ? config.modalityWeights || {}
                        : {},
                    ).map(([modality, weight]) => (
                      <div
                        key={modality}
                        className="flex items-center space-x-2"
                      >
                        <Label className="capitalize w-20">{modality}</Label>
                        <Slider
                          value={[weight as number]}
                          onValueChange={([value]) =>
                            onChange(
                              `dimensions.${dimension}.modalityWeights.${modality}`,
                              value,
                            )
                          }
                          min={0}
                          max={1}
                          step={0.01}
                          disabled={readOnly}
                          className="flex-1"
                        />
                        <span className="text-sm w-12">
                          {(weight as number).toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {dimension === "active" && (
                  <div className="space-y-2">
                    <Label>Action Space Size</Label>
                    <Input
                      type="number"
                      value={
                        "actionSpaceSize" in config ? config.actionSpaceSize : 8
                      }
                      onChange={(e) =>
                        onChange(
                          `dimensions.${dimension}.actionSpaceSize`,
                          parseInt(e.target.value),
                        )
                      }
                      disabled={readOnly}
                      min={1}
                      max={64}
                    />
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

const TemplateSelectionPanel: React.FC<{
  selectedTemplate: AgentTemplate | null;
  onTemplateSelect: (template: AgentTemplate) => void;
  readOnly: boolean;
}> = ({ selectedTemplate, onTemplateSelect, readOnly }) => {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold">Available Templates</h3>
        <p className="text-muted-foreground">
          Select a validated template to apply predefined boundary
          configurations
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {AGENT_TEMPLATES.map((template) => (
          <Card
            key={template.id}
            className={cn(
              "cursor-pointer transition-colors",
              selectedTemplate?.id === template.id && "ring-2 ring-primary",
              readOnly && "opacity-50 cursor-not-allowed",
            )}
            onClick={() => !readOnly && onTemplateSelect(template)}
          >
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  {template.icon}
                  <span>{template.name}</span>
                </CardTitle>
                <Badge variant="outline">{template.complexity}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                {template.description}
              </p>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="font-medium">States:</span>{" "}
                  {template.mathematicalFoundation.beliefsStates}
                </div>
                <div>
                  <span className="font-medium">Actions:</span>{" "}
                  {template.mathematicalFoundation.actionSpaces}
                </div>
                <div>
                  <span className="font-medium">Modalities:</span>{" "}
                  {template.mathematicalFoundation.observationModalities}
                </div>
                <div>
                  <span className="font-medium">Precision:</span>{" "}
                  {template.mathematicalFoundation.defaultPrecision.sensory}
                </div>
              </div>

              {template.expertRecommendation && (
                <div className="mt-4 p-2 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-xs text-blue-800">
                    <span className="font-semibold">Expert:</span>{" "}
                    {template.expertRecommendation}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

const AuditLogPanel: React.FC<{
  entries: AuditLogEntry[];
  stats: AuditLogStats | null;
  filter: AuditLogFilter;
  onFilterChange: (filter: AuditLogFilter) => void;
  searchText: string;
  onSearchTextChange: (text: string) => void;
  selectedEntries: string[];
  onSelectedEntriesChange: (entries: string[]) => void;
  isLoading: boolean;
  onExport?: (options: AuditExportOptions) => void;
  onRefresh: () => void;
}> = ({
  entries,
  stats,
  filter,
  onFilterChange,
  searchText,
  onSearchTextChange,
  selectedEntries,
  onSelectedEntriesChange,
  isLoading,
  onExport,
  onRefresh,
}) => {
  return (
    <div className="space-y-6">
      {/* Stats overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{stats.totalEntries}</div>
              <div className="text-sm text-muted-foreground">Total Entries</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-orange-600">
                {stats.complianceMetrics.totalHighRiskOperations}
              </div>
              <div className="text-sm text-muted-foreground">
                High Risk Operations
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-red-600">
                {stats.complianceMetrics.pendingApprovals}
              </div>
              <div className="text-sm text-muted-foreground">
                Pending Approvals
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-green-600">
                {stats.complianceMetrics.integrityViolations}
              </div>
              <div className="text-sm text-muted-foreground">
                Integrity Violations
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters and search */}
      <div className="flex items-center space-x-4">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search audit entries..."
              value={searchText}
              onChange={(e) => onSearchTextChange(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        <Button variant="outline" onClick={onRefresh} disabled={isLoading}>
          <RotateCcw className={cn("h-4 w-4", isLoading && "animate-spin")} />
        </Button>

        {onExport && (
          <Button
            variant="outline"
            onClick={() =>
              onExport({
                format: "csv",
                includeMetadata: true,
                includeIntegrityData: true,
              })
            }
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        )}
      </div>

      {/* Audit entries table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">
                  <Checkbox
                    checked={
                      selectedEntries.length === entries.length &&
                      entries.length > 0
                    }
                    onCheckedChange={(checked) => {
                      if (checked) {
                        onSelectedEntriesChange(entries.map((e) => e.id));
                      } else {
                        onSelectedEntriesChange([]);
                      }
                    }}
                  />
                </TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Operation</TableHead>
                <TableHead>Entity</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Risk Level</TableHead>
                <TableHead>Success</TableHead>
                <TableHead className="w-12">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {entries.map((entry) => (
                <TableRow key={entry.id}>
                  <TableCell>
                    <Checkbox
                      checked={selectedEntries.includes(entry.id)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          onSelectedEntriesChange([
                            ...selectedEntries,
                            entry.id,
                          ]);
                        } else {
                          onSelectedEntriesChange(
                            selectedEntries.filter((id) => id !== entry.id),
                          );
                        }
                      }}
                    />
                  </TableCell>
                  <TableCell className="text-sm">
                    {format(new Date(entry.timestamp), "MMM dd, yyyy HH:mm:ss")}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{entry.operationType}</Badge>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm">
                      <div className="font-medium">{entry.entityType}</div>
                      <div className="text-muted-foreground">
                        {entry.entityId}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell
                    className="max-w-xs truncate"
                    title={entry.description}
                  >
                    {entry.description}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        entry.compliance.riskLevel === "critical"
                          ? "destructive"
                          : entry.compliance.riskLevel === "high"
                            ? "destructive"
                            : entry.compliance.riskLevel === "medium"
                              ? "secondary"
                              : "outline"
                      }
                    >
                      {entry.compliance.riskLevel}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {entry.metadata.success ? (
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                    ) : (
                      <X className="h-4 w-4 text-red-600" />
                    )}
                  </TableCell>
                  <TableCell>
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

const CompliancePanel: React.FC<{
  configuration: BoundaryConfiguration;
  onChange: (path: string, value: any) => void;
  auditStats: AuditLogStats | null;
  readOnly: boolean;
}> = ({ configuration, onChange, auditStats, readOnly }) => {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold">Compliance Configuration</h3>
        <p className="text-muted-foreground">
          Configure compliance frameworks and audit settings
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Framework Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Compliance Framework</Label>
              <Select
                value={configuration.compliance.framework}
                onValueChange={(value) =>
                  onChange("compliance.framework", value)
                }
                disabled={readOnly}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ADR-011">ADR-011</SelectItem>
                  <SelectItem value="GDPR">GDPR</SelectItem>
                  <SelectItem value="HIPAA">HIPAA</SelectItem>
                  <SelectItem value="SOX">SOX</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Retention Period (days)</Label>
              <Input
                type="number"
                value={configuration.compliance.retentionPeriod}
                onChange={(e) =>
                  onChange(
                    "compliance.retentionPeriod",
                    parseInt(e.target.value),
                  )
                }
                disabled={readOnly}
                min={30}
                max={2555} // 7 years
              />
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Switch
                checked={configuration.compliance.auditingEnabled}
                onCheckedChange={(checked) =>
                  onChange("compliance.auditingEnabled", checked)
                }
                disabled={readOnly}
              />
              <Label>Enable Audit Logging</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                checked={configuration.compliance.encryptionRequired}
                onCheckedChange={(checked) =>
                  onChange("compliance.encryptionRequired", checked)
                }
                disabled={readOnly}
              />
              <Label>Require Encryption</Label>
            </div>
          </div>
        </CardContent>
      </Card>

      {auditStats && (
        <Card>
          <CardHeader>
            <CardTitle>Compliance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">
                  High Risk Operations
                </div>
                <div className="text-2xl font-bold text-orange-600">
                  {auditStats.complianceMetrics.totalHighRiskOperations}
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">
                  Pending Approvals
                </div>
                <div className="text-2xl font-bold text-red-600">
                  {auditStats.complianceMetrics.pendingApprovals}
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">
                  Integrity Violations
                </div>
                <div className="text-2xl font-bold text-red-600">
                  {auditStats.complianceMetrics.integrityViolations}
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">
                  Avg Operation Duration
                </div>
                <div className="text-2xl font-bold">
                  {auditStats.complianceMetrics.averageOperationDuration.toFixed(
                    1,
                  )}
                  ms
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const ExportDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onExport: (options: AuditExportOptions) => void;
  entryCount: number;
}> = ({ isOpen, onClose, onExport, entryCount }) => {
  const [format, setFormat] = useState<"json" | "csv" | "pdf" | "xlsx">("csv");
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeIntegrityData, setIncludeIntegrityData] = useState(false);
  const [reportTitle, setReportTitle] = useState("Audit Log Report");

  const handleExport = () => {
    onExport({
      format,
      includeMetadata,
      includeIntegrityData,
      reportTitle,
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Export Audit Log</DialogTitle>
          <DialogDescription>
            Export {entryCount} audit log entries in your preferred format
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Export Format</Label>
            <Select
              value={format}
              onValueChange={(value: any) => setFormat(value)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">CSV</SelectItem>
                <SelectItem value="json">JSON</SelectItem>
                <SelectItem value="xlsx">Excel (XLSX)</SelectItem>
                <SelectItem value="pdf">PDF</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Report Title</Label>
            <Input
              value={reportTitle}
              onChange={(e) => setReportTitle(e.target.value)}
              placeholder="Enter report title"
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                checked={includeMetadata}
                onCheckedChange={(checked) =>
                  setIncludeMetadata(checked === true)
                }
              />
              <Label>Include Metadata</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                checked={includeIntegrityData}
                onCheckedChange={(checked) =>
                  setIncludeIntegrityData(checked === true)
                }
              />
              <Label>Include Integrity Data</Label>
            </div>
          </div>

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleExport}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default MarkovBlanketConfigurationUI;
