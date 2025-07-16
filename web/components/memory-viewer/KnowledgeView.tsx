import React, { useState, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Brain,
  Plus,
  Search,
  Download,
  Trash2,
  Edit3,
  Save,
  X,
  BookOpen,
  Lightbulb,
  Tag,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { exportAgentKnowledge } from "@/lib/knowledge-export";
import { formatTimestamp, extractTagsFromMarkdown } from "@/lib/utils";
import { renderMarkdownWithTags, calculateKnowledgeStats } from "@/lib/memory-viewer-utils";
import type { KnowledgeEntry, SelectedKnowledgeNode } from "@/types/memory-viewer";
import type { Agent } from "@/lib/types";

interface KnowledgeViewProps {
  selectedAgent: Agent | null;
  knowledge: KnowledgeEntry[];
  selectedKnowledgeNode?: SelectedKnowledgeNode | null;
  onAddKnowledge: (knowledge: KnowledgeEntry) => void;
  onDeleteKnowledge: (id: string) => void;
  onUpdateKnowledge: (id: string, updates: Partial<KnowledgeEntry>) => void;
  onExtractBeliefs?: (content: string) => Promise<void>;
}

export function KnowledgeView({
  selectedAgent,
  knowledge = [],
  selectedKnowledgeNode,
  onAddKnowledge,
  onDeleteKnowledge,
  onUpdateKnowledge,
  onExtractBeliefs,
}: KnowledgeViewProps) {
  const [knowledgeView, setKnowledgeView] = useState<"browse" | "add" | "insights">("browse");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<"lastUpdated" | "importance" | "title">("lastUpdated");
  const [filterConfidence, setFilterConfidence] = useState([0, 1]);
  const [showOnlyRelated, setShowOnlyRelated] = useState(false);

  // Add knowledge form state
  const [newKnowledgeTitle, setNewKnowledgeTitle] = useState("");
  const [newKnowledgeContent, setNewKnowledgeContent] = useState("");
  const [newKnowledgeType, setNewKnowledgeType] = useState<
    "fact" | "experience" | "insight" | "goal" | "preference"
  >("fact");
  const [newKnowledgeTags, setNewKnowledgeTags] = useState<string[]>([]);
  const [newKnowledgeConfidence, setNewKnowledgeConfidence] = useState(0.8);
  const [newKnowledgeImportance, setNewKnowledgeImportance] = useState(5);

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [editTitle, setEditTitle] = useState("");

  // Delete confirmation
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Belief extraction state
  const [extractionText, setExtractionText] = useState("");
  const [isExtracting, setIsExtracting] = useState(false);

  const { toast } = useToast();

  // Calculate stats
  const stats = useMemo(() => calculateKnowledgeStats(knowledge), [knowledge]);

  // Filter and sort knowledge
  const filteredKnowledge = useMemo(() => {
    let filtered = [...knowledge];

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(
        (k) =>
          k.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          k.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
          k.tags.some((tag) => tag.toLowerCase().includes(searchTerm.toLowerCase())),
      );
    }

    // Tag filter
    if (selectedTags.length > 0) {
      filtered = filtered.filter((k) => selectedTags.every((tag) => k.tags.includes(tag)));
    }

    // Confidence filter
    filtered = filtered.filter(
      (k) =>
        (k.confidence || 0) >= filterConfidence[0] && (k.confidence || 0) <= filterConfidence[1],
    );

    // Related knowledge filter
    if (showOnlyRelated && selectedKnowledgeNode?.type === "entry") {
      const relatedIds =
        knowledge.find((k) => k.id === selectedKnowledgeNode.id)?.relatedKnowledge || [];
      filtered = filtered.filter((k) => relatedIds.includes(k.id));
    }

    // Sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "importance":
          return (b.importance || 0) - (a.importance || 0);
        case "title":
          return a.title.localeCompare(b.title);
        case "lastUpdated":
        default:
          return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime();
      }
    });

    return filtered;
  }, [
    knowledge,
    searchTerm,
    selectedTags,
    filterConfidence,
    sortBy,
    showOnlyRelated,
    selectedKnowledgeNode,
  ]);

  // Handle knowledge addition
  const handleAddKnowledge = () => {
    if (!selectedAgent || !newKnowledgeTitle || !newKnowledgeContent) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    const extractedTags = extractTagsFromMarkdown(newKnowledgeContent);
    const allTags = Array.from(new Set([...newKnowledgeTags, ...extractedTags]));

    const newKnowledge: KnowledgeEntry = {
      id: `knowledge-${Date.now()}`,
      title: newKnowledgeTitle,
      content: newKnowledgeContent,
      type: newKnowledgeType,
      source: "user",
      timestamp: new Date().toISOString(),
      lastUpdated: new Date().toISOString(),
      tags: allTags,
      confidence: newKnowledgeConfidence,
      importance: newKnowledgeImportance,
      agentId: selectedAgent.id,
    };

    onAddKnowledge(newKnowledge);

    // Reset form
    setNewKnowledgeTitle("");
    setNewKnowledgeContent("");
    setNewKnowledgeType("fact");
    setNewKnowledgeTags([]);
    setNewKnowledgeConfidence(0.8);
    setNewKnowledgeImportance(5);
    setKnowledgeView("browse");

    toast({
      title: "Success",
      description: "Knowledge entry added successfully",
    });
  };

  // Handle knowledge update
  const handleUpdateKnowledge = (id: string) => {
    const entry = knowledge.find((k) => k.id === id);
    if (!entry) return;

    const extractedTags = extractTagsFromMarkdown(editContent);

    onUpdateKnowledge(id, {
      title: editTitle,
      content: editContent,
      tags: extractedTags,
    });

    setEditingId(null);
    setEditContent("");
    setEditTitle("");

    toast({
      title: "Success",
      description: "Knowledge entry updated successfully",
    });
  };

  // Handle knowledge deletion
  const handleDeleteKnowledge = (id: string) => {
    onDeleteKnowledge(id);
    setDeleteId(null);

    toast({
      title: "Success",
      description: "Knowledge entry deleted successfully",
    });
  };

  // Handle export
  const handleExport = async () => {
    if (!selectedAgent) return;

    try {
      await exportAgentKnowledge(selectedAgent, knowledge);
      toast({
        title: "Success",
        description: "Knowledge exported successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to export knowledge",
        variant: "destructive",
      });
    }
  };

  // Handle belief extraction
  const handleExtractBeliefs = async () => {
    if (!extractionText || !onExtractBeliefs) return;

    setIsExtracting(true);
    try {
      await onExtractBeliefs(extractionText);
      setExtractionText("");
      toast({
        title: "Success",
        description: "Beliefs extracted successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to extract beliefs",
        variant: "destructive",
      });
    } finally {
      setIsExtracting(false);
    }
  };

  // Start editing
  const startEditing = (entry: KnowledgeEntry) => {
    setEditingId(entry.id);
    setEditTitle(entry.title);
    setEditContent(entry.content);
  };

  return (
    <div className="h-full space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Knowledge Base
          </h3>
          <div className="flex gap-2">
            <Badge variant="outline">{stats.totalEntries} entries</Badge>
            <Badge variant="outline">
              {Math.round(stats.averageConfidence * 100)}% avg confidence
            </Badge>
          </div>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      <Tabs
        value={knowledgeView}
        onValueChange={(v) => setKnowledgeView(v as "browse" | "add" | "insights")}
      >
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="browse" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            Browse
          </TabsTrigger>
          <TabsTrigger value="add" className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            Add Knowledge
          </TabsTrigger>
          <TabsTrigger value="extract" className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4" />
            Extract Beliefs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="browse" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Search</Label>
                  <div className="relative">
                    <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search knowledge..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-8"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Sort by</Label>
                  <Select
                    value={sortBy}
                    onValueChange={(v) => setSortBy(v as "lastUpdated" | "importance" | "title")}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="date">Date</SelectItem>
                      <SelectItem value="confidence">Confidence</SelectItem>
                      <SelectItem value="title">Title</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label>
                  Confidence Range: {Math.round(filterConfidence[0] * 100)}% -{" "}
                  {Math.round(filterConfidence[1] * 100)}%
                </Label>
                <Slider
                  value={filterConfidence}
                  onValueChange={setFilterConfidence}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-full"
                />
              </div>

              {selectedKnowledgeNode?.type === "entry" && (
                <div className="flex items-center justify-between">
                  <Label htmlFor="related-only">Show only related knowledge</Label>
                  <Switch
                    id="related-only"
                    checked={showOnlyRelated}
                    onCheckedChange={setShowOnlyRelated}
                  />
                </div>
              )}

              {stats.topTags.length > 0 && (
                <div className="space-y-2">
                  <Label>Filter by tags</Label>
                  <div className="flex flex-wrap gap-2">
                    {stats.topTags.map((tag) => (
                      <Badge
                        key={tag}
                        variant={selectedTags.includes(tag) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => {
                          setSelectedTags((prev) =>
                            prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag],
                          );
                        }}
                      >
                        {tag} ({stats.tagFrequency[tag]})
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Knowledge entries */}
          <div className="space-y-4 max-h-[400px] overflow-y-auto">
            {filteredKnowledge.map((entry) => (
              <Card
                key={entry.id}
                className={selectedKnowledgeNode?.id === entry.id ? "ring-2 ring-primary" : ""}
              >
                <CardContent className="p-4">
                  {editingId === entry.id ? (
                    <div className="space-y-4">
                      <Input
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        placeholder="Title"
                      />
                      <Textarea
                        value={editContent}
                        onChange={(e) => setEditContent(e.target.value)}
                        placeholder="Content"
                        rows={4}
                      />
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setEditingId(null);
                            setEditContent("");
                            setEditTitle("");
                          }}
                        >
                          <X className="h-4 w-4 mr-2" />
                          Cancel
                        </Button>
                        <Button size="sm" onClick={() => handleUpdateKnowledge(entry.id)}>
                          <Save className="h-4 w-4 mr-2" />
                          Save
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start justify-between mb-2">
                        <div className="space-y-1">
                          <h4 className="font-medium">{entry.title}</h4>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span>{formatTimestamp(entry.timestamp)}</span>
                            <span>Source: {entry.source}</span>
                            {entry.confidence !== undefined && (
                              <span>Confidence: {Math.round(entry.confidence * 100)}%</span>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button size="sm" variant="ghost" onClick={() => startEditing(entry)}>
                            <Edit3 className="h-4 w-4" />
                          </Button>
                          <Button size="sm" variant="ghost" onClick={() => setDeleteId(entry.id)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>

                      <div
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdownWithTags(entry.content),
                        }}
                      />
                    </>
                  )}
                </CardContent>
              </Card>
            ))}

            {filteredKnowledge.length === 0 && (
              <Card>
                <CardContent className="p-8 text-center text-muted-foreground">
                  {knowledge.length === 0
                    ? "No knowledge entries yet"
                    : "No matching entries found"}
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="add" className="space-y-4">
          <Card>
            <CardContent className="p-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="knowledge-title">Title</Label>
                <Input
                  id="knowledge-title"
                  placeholder="Enter knowledge title..."
                  value={newKnowledgeTitle}
                  onChange={(e) => setNewKnowledgeTitle(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="knowledge-content">Content</Label>
                <Textarea
                  id="knowledge-content"
                  placeholder="Enter knowledge content... Use #hashtags for automatic tagging"
                  value={newKnowledgeContent}
                  onChange={(e) => setNewKnowledgeContent(e.target.value)}
                  rows={6}
                />
              </div>

              <div className="space-y-2">
                <Label>Confidence: {Math.round(newKnowledgeConfidence * 100)}%</Label>
                <Slider
                  value={[newKnowledgeConfidence]}
                  onValueChange={([v]) => setNewKnowledgeConfidence(v)}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label>Tags</Label>
                <div className="flex flex-wrap gap-2">
                  {extractTagsFromMarkdown(newKnowledgeContent).map((tag) => (
                    <Badge key={tag} variant="secondary">
                      <Tag className="h-3 w-3 mr-1" />
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>

              <Button
                onClick={handleAddKnowledge}
                disabled={!newKnowledgeTitle || !newKnowledgeContent}
                className="w-full"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Knowledge Entry
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="extract" className="space-y-4">
          <Card>
            <CardContent className="p-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="extraction-text">Text to Extract Beliefs From</Label>
                <Textarea
                  id="extraction-text"
                  placeholder="Paste or type text here to extract beliefs and knowledge..."
                  value={extractionText}
                  onChange={(e) => setExtractionText(e.target.value)}
                  rows={10}
                />
              </div>

              <Button
                onClick={handleExtractBeliefs}
                disabled={!extractionText || isExtracting || !onExtractBeliefs}
                className="w-full"
              >
                <Lightbulb className="h-4 w-4 mr-2" />
                {isExtracting ? "Extracting..." : "Extract Beliefs"}
              </Button>

              {!onExtractBeliefs && (
                <p className="text-sm text-muted-foreground text-center">
                  Belief extraction requires LLM configuration
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Delete confirmation dialog */}
      <AlertDialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Knowledge Entry</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this knowledge entry? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => deleteId && handleDeleteKnowledge(deleteId)}>
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
