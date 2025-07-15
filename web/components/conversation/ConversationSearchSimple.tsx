/**
 * ConversationSearch - Simplified search and filtering functionality
 */

import React, { useState, useCallback, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Search, Filter, Download, X, ChevronDown, ChevronUp } from "lucide-react";
import type { ConversationMessage } from "./ConversationPanel";
import type { Agent } from "@/lib/types";

export interface SearchFilters {
  searchQuery: string;
  dateFrom?: string;
  dateTo?: string;
  userIds: string[];
  agentIds: string[];
  messageTypes: string[];
  contentTypes: string[];
}

export type ExportFormat = "json" | "csv" | "pdf" | "txt";

interface ConversationSearchProps {
  messages: ConversationMessage[];
  agents: Agent[];
  users: Array<{ id: string; name: string }>;
  onFilterChange?: (filters: SearchFilters) => void;
  onExport?: (format: ExportFormat, data: any) => void;
  className?: string;
}

export function ConversationSearch({
  messages,
  agents,
  users,
  onFilterChange,
  onExport,
  className = "",
}: ConversationSearchProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("json");
  const [filters, setFilters] = useState<SearchFilters>({
    searchQuery: "",
    userIds: [],
    agentIds: [],
    messageTypes: [],
    contentTypes: [],
  });

  // Get unique message types
  const messageTypes = useMemo(() => {
    const types = new Set<string>();
    messages.forEach((msg) => types.add(msg.message_type));
    return Array.from(types);
  }, [messages]);

  // Filter messages based on current filters
  const filteredMessages = useMemo(() => {
    let filtered = messages;

    // Search query filter
    if (filters.searchQuery) {
      filtered = filtered.filter((msg) =>
        msg.content.toLowerCase().includes(filters.searchQuery.toLowerCase()),
      );
    }

    // Date filters
    if (filters.dateFrom) {
      filtered = filtered.filter((msg) => msg.timestamp >= filters.dateFrom!);
    }
    if (filters.dateTo) {
      filtered = filtered.filter((msg) => msg.timestamp <= filters.dateTo!);
    }

    // User filter
    if (filters.userIds.length > 0) {
      filtered = filtered.filter((msg) => msg.user_id && filters.userIds.includes(msg.user_id));
    }

    // Agent filter
    if (filters.agentIds.length > 0) {
      filtered = filtered.filter((msg) => msg.agent_id && filters.agentIds.includes(msg.agent_id));
    }

    // Message type filter
    if (filters.messageTypes.length > 0) {
      filtered = filtered.filter((msg) => filters.messageTypes.includes(msg.message_type));
    }

    return filtered;
  }, [messages, filters]);

  // Active filter count
  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.searchQuery) count++;
    if (filters.dateFrom || filters.dateTo) count++;
    if (filters.userIds.length > 0) count++;
    if (filters.agentIds.length > 0) count++;
    if (filters.messageTypes.length > 0) count++;
    return count;
  }, [filters]);

  // Update filters and notify parent
  const updateFilters = useCallback(
    (newFilters: Partial<SearchFilters>) => {
      const updated = { ...filters, ...newFilters };
      setFilters(updated);
      onFilterChange?.(updated);
    },
    [filters, onFilterChange],
  );

  // Handle search query change
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      updateFilters({ searchQuery });
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, updateFilters]);

  // Toggle user filter
  const toggleUserFilter = useCallback(
    (userId: string) => {
      const newIds = filters.userIds.includes(userId)
        ? filters.userIds.filter((id) => id !== userId)
        : [...filters.userIds, userId];
      updateFilters({ userIds: newIds });
    },
    [filters.userIds, updateFilters],
  );

  // Toggle agent filter
  const toggleAgentFilter = useCallback(
    (agentId: string) => {
      const newIds = filters.agentIds.includes(agentId)
        ? filters.agentIds.filter((id) => id !== agentId)
        : [...filters.agentIds, agentId];
      updateFilters({ agentIds: newIds });
    },
    [filters.agentIds, updateFilters],
  );

  // Toggle message type filter
  const toggleMessageTypeFilter = useCallback(
    (type: string) => {
      const newTypes = filters.messageTypes.includes(type)
        ? filters.messageTypes.filter((t) => t !== type)
        : [...filters.messageTypes, type];
      updateFilters({ messageTypes: newTypes });
    },
    [filters.messageTypes, updateFilters],
  );

  // Clear all filters
  const clearAllFilters = useCallback(() => {
    setSearchQuery("");
    updateFilters({
      searchQuery: "",
      dateFrom: undefined,
      dateTo: undefined,
      userIds: [],
      agentIds: [],
      messageTypes: [],
      contentTypes: [],
    });
  }, [updateFilters]);

  // Handle export
  const handleExport = useCallback(() => {
    if (onExport) {
      onExport(exportFormat, filteredMessages);
    }
  }, [exportFormat, filteredMessages, onExport]);

  // Format message preview
  const formatMessagePreview = (message: ConversationMessage) => {
    const maxLength = 100;
    const content = message.content;

    if (!filters.searchQuery) {
      return content.length > maxLength ? content.substring(0, maxLength) + "..." : content;
    }

    // Highlight search term
    const searchIndex = content.toLowerCase().indexOf(filters.searchQuery.toLowerCase());
    if (searchIndex === -1) {
      return content.length > maxLength ? content.substring(0, maxLength) + "..." : content;
    }

    const start = Math.max(0, searchIndex - 30);
    const end = Math.min(content.length, searchIndex + filters.searchQuery.length + 30);
    const preview = content.substring(start, end);

    const parts = preview.split(new RegExp(`(${filters.searchQuery})`, "gi"));
    return (
      <>
        {start > 0 && "..."}
        {parts.map((part: string, i: number) =>
          part.toLowerCase() === filters.searchQuery.toLowerCase() ? (
            <mark key={i}>{part}</mark>
          ) : (
            part
          ),
        )}
        {end < content.length && "..."}
      </>
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Search & Export
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Search Input */}
        <div className="space-y-2">
          <Label>Search Messages</Label>
          <div className="relative">
            <Input
              type="text"
              placeholder="Search messages..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pr-8"
            />
            {searchQuery && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 h-6 w-6 p-0"
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>
        </div>

        {/* Filter Toggle */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowFilters(!showFilters)}
          className="w-full justify-between"
        >
          <span className="flex items-center gap-2">
            <Filter className="h-4 w-4" />
            Filters
            {activeFilterCount > 0 && (
              <Badge variant="secondary">
                {activeFilterCount} filter{activeFilterCount > 1 ? "s" : ""} active
              </Badge>
            )}
          </span>
          {showFilters ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>

        {/* Advanced Filters */}
        {showFilters && (
          <div className="space-y-4 p-4 border rounded-lg bg-muted/50">
            {/* Date Range */}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-2">
                <Label htmlFor="date-from">From Date</Label>
                <Input
                  id="date-from"
                  type="date"
                  value={filters.dateFrom || ""}
                  onChange={(e) => updateFilters({ dateFrom: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="date-to">To Date</Label>
                <Input
                  id="date-to"
                  type="date"
                  value={filters.dateTo || ""}
                  onChange={(e) => updateFilters({ dateTo: e.target.value })}
                />
              </div>
            </div>

            {/* Users */}
            <div className="space-y-2">
              <Label>Filter by Users</Label>
              <div className="flex flex-wrap gap-2">
                {users.map((user) => (
                  <Badge
                    key={user.id}
                    variant={filters.userIds.includes(user.id) ? "default" : "outline"}
                    className="cursor-pointer"
                    onClick={() => toggleUserFilter(user.id)}
                  >
                    {user.name}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Agents */}
            <div className="space-y-2">
              <Label>Filter by Agents</Label>
              <div className="flex flex-wrap gap-2">
                {agents.map((agent) => (
                  <Badge
                    key={agent.id}
                    variant={filters.agentIds.includes(agent.id) ? "default" : "outline"}
                    className="cursor-pointer"
                    onClick={() => toggleAgentFilter(agent.id)}
                  >
                    {agent.name}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Message Types */}
            <div className="space-y-2">
              <Label>Message Types</Label>
              <div className="flex flex-wrap gap-2">
                {messageTypes.map((type) => (
                  <Badge
                    key={type}
                    variant={filters.messageTypes.includes(type) ? "default" : "outline"}
                    className="cursor-pointer"
                    onClick={() => toggleMessageTypeFilter(type)}
                  >
                    {type}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Clear Filters */}
            <Button variant="outline" size="sm" onClick={clearAllFilters} className="w-full">
              Clear All Filters
            </Button>
          </div>
        )}

        {/* Export Section */}
        <div className="space-y-2">
          <Label>Export Format</Label>
          <div className="flex gap-2">
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value as ExportFormat)}
              className="flex-1 px-3 py-2 border rounded-md"
            >
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
              <option value="pdf">PDF</option>
              <option value="txt">TXT</option>
            </select>
            <Button onClick={handleExport} disabled={filteredMessages.length === 0}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>

        {/* Results Summary */}
        <div className="text-sm text-muted-foreground">
          Found {filteredMessages.length} message{filteredMessages.length !== 1 ? "s" : ""}
          {messages.length > 0 && ` of ${messages.length} total`}
        </div>

        {/* Preview */}
        {filteredMessages.length > 0 && (
          <div className="space-y-2">
            <Label>Preview (First 5 messages)</Label>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {filteredMessages.slice(0, 5).map((msg) => (
                <div key={msg.id} className="p-2 border rounded text-sm">
                  <div className="font-medium text-xs text-muted-foreground mb-1">
                    {msg.user_id
                      ? users.find((u) => u.id === msg.user_id)?.name
                      : msg.agent_id
                        ? agents.find((a) => a.id === msg.agent_id)?.name
                        : "Unknown"}{" "}
                    • {new Date(msg.timestamp).toLocaleString()}
                  </div>
                  <div className="text-sm">{formatMessagePreview(msg)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
