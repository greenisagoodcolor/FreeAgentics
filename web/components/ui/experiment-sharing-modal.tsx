"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "./dialog";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./card";
import { Button } from "./button";
import { Input } from "./input";
import { Label } from "./label";
import { Textarea } from "./textarea";
import { Checkbox } from "./checkbox";
import { Badge } from "./badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./select";
import { useToast } from "./use-toast";
import {
  Link,
  Copy,
  Eye,
  EyeOff,
  Clock,
  Users,
  Share2,
  CheckCircle,
  AlertCircle,
  History,
  GitBranch,
  FileText,
  Calendar,
  Lock,
  Globe,
  User,
  Download,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SharedLink {
  id: string;
  url: string;
  name: string;
  description?: string;
  createdAt: string;
  expiresAt?: string;
  accessLevel: "view" | "comment" | "edit";
  isActive: boolean;
  accessCount: number;
  lastAccessed?: string;
}

interface ExperimentVersion {
  id: string;
  name: string;
  createdAt: string;
  createdBy: string;
  changesSummary: string;
  statistics: {
    totalAgents: number;
    totalConversations: number;
    totalMessages: number;
    totalKnowledgeNodes: number;
  };
}

interface ChangeLogEntry {
  id: string;
  timestamp: string;
  author: string;
  action: "created" | "modified" | "deleted" | "shared" | "version_created";
  component: string;
  description: string;
  details?: string;
}

interface SharingModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  exportId: string;
  exportName: string;
  exportDescription?: string;
}

export function ExperimentSharingModal({
  open,
  onOpenChange,
  exportId,
  exportName,
  exportDescription,
}: SharingModalProps) {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<"link" | "versions" | "changelog">(
    "link",
  );
  const [sharedLinks, setSharedLinks] = useState<SharedLink[]>([]);
  const [versions, setVersions] = useState<ExperimentVersion[]>([]);
  const [changeLog, setChangeLog] = useState<ChangeLogEntry[]>([]);
  const [isCreatingLink, setIsCreatingLink] = useState(false);
  const [newLinkConfig, setNewLinkConfig] = useState({
    name: "",
    description: "",
    accessLevel: "view" as "view" | "comment" | "edit",
    expiresIn: "7d" as string,
    requireAuth: false,
  });

  const loadSharingData = useCallback(async () => {
    // Simulate loading data
    setSharedLinks([
      {
        id: "link_001",
        url: `https://freeagentics.com/shared/${exportId}/view?token=abc123def456`,
        name: "Research Team Access",
        description: "Shared with the research team for collaborative analysis",
        createdAt: "2024-01-15T10:30:00Z",
        expiresAt: "2024-01-22T10:30:00Z",
        accessLevel: "view",
        isActive: true,
        accessCount: 12,
        lastAccessed: "2024-01-16T14:20:00Z",
      },
    ]);

    setVersions([
      {
        id: "v1.0.0",
        name: "Initial Export",
        createdAt: "2024-01-15T10:30:00Z",
        createdBy: "researcher@example.com",
        changesSummary: "Initial export with all components",
        statistics: {
          totalAgents: 5,
          totalConversations: 12,
          totalMessages: 89,
          totalKnowledgeNodes: 234,
        },
      },
      {
        id: "v1.1.0",
        name: "Updated Knowledge Graphs",
        createdAt: "2024-01-16T14:20:00Z",
        createdBy: "analyst@example.com",
        changesSummary:
          "Added new knowledge graph connections and updated agent behaviors",
        statistics: {
          totalAgents: 5,
          totalConversations: 15,
          totalMessages: 127,
          totalKnowledgeNodes: 298,
        },
      },
    ]);

    setChangeLog([
      {
        id: "change_001",
        timestamp: "2024-01-16T14:20:00Z",
        author: "analyst@example.com",
        action: "modified",
        component: "Knowledge Graphs",
        description: "Added 64 new nodes and 12 edges",
        details:
          "Enhanced agent decision-making pathways with additional concept relationships",
      },
      {
        id: "change_002",
        timestamp: "2024-01-16T09:15:00Z",
        author: "researcher@example.com",
        action: "shared",
        component: "Export",
        description: "Shared experiment with research team",
        details: "Created read-only access link for collaborative analysis",
      },
      {
        id: "change_003",
        timestamp: "2024-01-15T10:30:00Z",
        author: "researcher@example.com",
        action: "created",
        component: "Export",
        description: "Created initial experiment export",
        details:
          "Exported complete experiment state including all agents, conversations, and knowledge graphs",
      },
    ]);
  }, [exportId]);

  useEffect(() => {
    if (open) {
      loadSharingData();
    }
  }, [open, exportId, loadSharingData]);

  const handleCreateLink = async () => {
    if (!newLinkConfig.name.trim()) {
      toast({
        title: "Link name required",
        description: "Please provide a name for this shared link",
      });
      return;
    }

    setIsCreatingLink(true);

    // Simulate API call
    setTimeout(() => {
      const newLink: SharedLink = {
        id: `link_${Date.now()}`,
        url: `https://freeagentics.com/shared/${exportId}/view?token=${Math.random().toString(36).substring(2, 10)}`,
        name: newLinkConfig.name,
        description: newLinkConfig.description,
        createdAt: new Date().toISOString(),
        expiresAt: getExpirationDate(newLinkConfig.expiresIn),
        accessLevel: newLinkConfig.accessLevel,
        isActive: true,
        accessCount: 0,
      };

      setSharedLinks((prev) => [newLink, ...prev]);
      setNewLinkConfig({
        name: "",
        description: "",
        accessLevel: "view",
        expiresIn: "7d",
        requireAuth: false,
      });
      setIsCreatingLink(false);

      toast({
        title: "Link created successfully",
        description:
          "Your shareable link has been generated and copied to clipboard",
      });

      // Copy to clipboard
      navigator.clipboard.writeText(newLink.url);
    }, 1000);
  };

  const getExpirationDate = (expiresIn: string): string => {
    const now = new Date();
    const expirationMap: Record<string, number> = {
      "1h": 1000 * 60 * 60,
      "1d": 1000 * 60 * 60 * 24,
      "7d": 1000 * 60 * 60 * 24 * 7,
      "30d": 1000 * 60 * 60 * 24 * 30,
      never: 0,
    };

    const duration = expirationMap[expiresIn];
    if (duration === 0) return "";

    return new Date(now.getTime() + duration).toISOString();
  };

  const handleCopyLink = (url: string) => {
    navigator.clipboard.writeText(url);
    toast({
      title: "Link copied",
      description: "The shareable link has been copied to your clipboard",
    });
  };

  const handleToggleLink = (linkId: string) => {
    setSharedLinks((prev) =>
      prev.map((link) =>
        link.id === linkId ? { ...link, isActive: !link.isActive } : link,
      ),
    );
  };

  const handleDeleteLink = (linkId: string) => {
    setSharedLinks((prev) => prev.filter((link) => link.id !== linkId));
    toast({
      title: "Link deleted",
      description: "The shareable link has been removed",
    });
  };

  const renderLinkManagement = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Create Shareable Link</CardTitle>
          <CardDescription>
            Generate a secure link to share this experiment with others
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2">
            <Label htmlFor="link-name">Link Name</Label>
            <Input
              id="link-name"
              value={newLinkConfig.name}
              onChange={(e) =>
                setNewLinkConfig({ ...newLinkConfig, name: e.target.value })
              }
              placeholder="Research Team Access"
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="link-description">Description (Optional)</Label>
            <Textarea
              id="link-description"
              value={newLinkConfig.description}
              onChange={(e) =>
                setNewLinkConfig({
                  ...newLinkConfig,
                  description: e.target.value,
                })
              }
              placeholder="Describe who this link is for and its purpose"
              rows={2}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-2">
              <Label>Access Level</Label>
              <Select
                value={newLinkConfig.accessLevel}
                onValueChange={(value) =>
                  setNewLinkConfig({
                    ...newLinkConfig,
                    accessLevel: value as "view" | "comment" | "edit",
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="view">
                    <div className="flex items-center gap-2">
                      <Eye className="h-4 w-4" />
                      <span>View Only</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="comment">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      <span>Comment</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="edit">
                    <div className="flex items-center gap-2">
                      <Settings className="h-4 w-4" />
                      <span>Edit</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label>Expires In</Label>
              <Select
                value={newLinkConfig.expiresIn}
                onValueChange={(value) =>
                  setNewLinkConfig({ ...newLinkConfig, expiresIn: value })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1h">1 Hour</SelectItem>
                  <SelectItem value="1d">1 Day</SelectItem>
                  <SelectItem value="7d">7 Days</SelectItem>
                  <SelectItem value="30d">30 Days</SelectItem>
                  <SelectItem value="never">Never</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="require-auth"
              checked={newLinkConfig.requireAuth}
              onCheckedChange={(checked) =>
                setNewLinkConfig({ ...newLinkConfig, requireAuth: !!checked })
              }
            />
            <Label htmlFor="require-auth" className="text-sm">
              Require authentication to access
            </Label>
          </div>

          <Button
            onClick={handleCreateLink}
            disabled={isCreatingLink}
            className="w-full"
          >
            {isCreatingLink ? (
              <>
                <Clock className="mr-2 h-4 w-4 animate-spin" />
                Creating Link...
              </>
            ) : (
              <>
                <Share2 className="mr-2 h-4 w-4" />
                Create Shareable Link
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium">Active Links</h3>
          <Badge variant="secondary">
            {sharedLinks.filter((l) => l.isActive).length} active
          </Badge>
        </div>

        {sharedLinks.length === 0 ? (
          <Card>
            <CardContent className="text-center py-8">
              <Share2 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">
                No shared links created yet
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {sharedLinks.map((link) => (
              <Card
                key={link.id}
                className={cn(!link.isActive && "opacity-60")}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium">{link.name}</h4>
                        <Badge
                          variant={
                            link.accessLevel === "view"
                              ? "secondary"
                              : link.accessLevel === "comment"
                                ? "default"
                                : "destructive"
                          }
                        >
                          {link.accessLevel}
                        </Badge>
                        {!link.isActive && (
                          <Badge variant="outline">Inactive</Badge>
                        )}
                      </div>
                      {link.description && (
                        <p className="text-sm text-muted-foreground mb-2">
                          {link.description}
                        </p>
                      )}
                      <div className="text-xs text-muted-foreground space-y-1">
                        <div>
                          Created: {new Date(link.createdAt).toLocaleString()}
                        </div>
                        {link.expiresAt && (
                          <div>
                            Expires: {new Date(link.expiresAt).toLocaleString()}
                          </div>
                        )}
                        <div>Access count: {link.accessCount}</div>
                        {link.lastAccessed && (
                          <div>
                            Last accessed:{" "}
                            {new Date(link.lastAccessed).toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-2 ml-4">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleCopyLink(link.url)}
                        title="Copy link"
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleToggleLink(link.id)}
                        title={
                          link.isActive ? "Deactivate link" : "Activate link"
                        }
                      >
                        {link.isActive ? (
                          <EyeOff className="h-4 w-4" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDeleteLink(link.id)}
                        title="Delete link"
                        className="text-destructive hover:text-destructive"
                      >
                        <AlertCircle className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const renderVersionComparison = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Version History</h3>
        <Badge variant="secondary">{versions.length} versions</Badge>
      </div>

      {versions.length === 0 ? (
        <Card>
          <CardContent className="text-center py-8">
            <GitBranch className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No versions available</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {versions.map((version, index) => (
            <Card key={version.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-medium">{version.name}</h4>
                      <Badge variant="outline">{version.id}</Badge>
                      {index === 0 && <Badge>Latest</Badge>}
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">
                      {version.changesSummary}
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-xs">
                      <div>
                        <span className="text-muted-foreground">Created:</span>
                        <span className="ml-1">
                          {new Date(version.createdAt).toLocaleString()}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">By:</span>
                        <span className="ml-1">{version.createdBy}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Agents:</span>
                        <span className="ml-1">
                          {version.statistics.totalAgents}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">
                          Conversations:
                        </span>
                        <span className="ml-1">
                          {version.statistics.totalConversations}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Messages:</span>
                        <span className="ml-1">
                          {version.statistics.totalMessages}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">
                          Knowledge Nodes:
                        </span>
                        <span className="ml-1">
                          {version.statistics.totalKnowledgeNodes}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 ml-4">
                    <Button variant="outline" size="sm">
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </Button>
                    {index > 0 && (
                      <Button variant="ghost" size="sm">
                        <GitBranch className="mr-2 h-4 w-4" />
                        Compare
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );

  const renderChangeLog = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Change Log</h3>
        <Badge variant="secondary">{changeLog.length} entries</Badge>
      </div>

      {changeLog.length === 0 ? (
        <Card>
          <CardContent className="text-center py-8">
            <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No changes recorded yet</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {changeLog.map((entry) => (
            <Card key={entry.id}>
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-1">
                    {entry.action === "created" && (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    )}
                    {entry.action === "modified" && (
                      <Settings className="h-4 w-4 text-blue-600" />
                    )}
                    {entry.action === "deleted" && (
                      <AlertCircle className="h-4 w-4 text-red-600" />
                    )}
                    {entry.action === "shared" && (
                      <Share2 className="h-4 w-4 text-purple-600" />
                    )}
                    {entry.action === "version_created" && (
                      <GitBranch className="h-4 w-4 text-orange-600" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium capitalize">
                        {entry.action.replace("_", " ")}
                      </span>
                      <Badge variant="outline">{entry.component}</Badge>
                    </div>
                    <p className="text-sm mb-2">{entry.description}</p>
                    {entry.details && (
                      <p className="text-xs text-muted-foreground mb-2">
                        {entry.details}
                      </p>
                    )}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <User className="h-3 w-3" />
                        <span>{entry.author}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        <span>
                          {new Date(entry.timestamp).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[700px] max-h-[80vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>Share Experiment</DialogTitle>
          <DialogDescription>
            Collaborate on &quot;{exportName}&quot; with secure sharing, version
            control, and change tracking.
          </DialogDescription>
        </DialogHeader>

        <Tabs
          value={activeTab}
          onValueChange={(value) =>
            setActiveTab(value as "link" | "versions" | "changelog")
          }
          className="flex-1"
        >
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="link">
              <Share2 className="mr-2 h-4 w-4" />
              Sharing
            </TabsTrigger>
            <TabsTrigger value="versions">
              <GitBranch className="mr-2 h-4 w-4" />
              Versions
            </TabsTrigger>
            <TabsTrigger value="changelog">
              <History className="mr-2 h-4 w-4" />
              Changes
            </TabsTrigger>
          </TabsList>

          <div className="mt-4 max-h-[60vh] overflow-y-auto">
            <TabsContent value="link" className="space-y-4">
              {renderLinkManagement()}
            </TabsContent>

            <TabsContent value="versions" className="space-y-4">
              {renderVersionComparison()}
            </TabsContent>

            <TabsContent value="changelog" className="space-y-4">
              {renderChangeLog()}
            </TabsContent>
          </div>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
