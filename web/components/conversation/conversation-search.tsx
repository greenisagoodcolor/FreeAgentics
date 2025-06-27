"use client";

import React, { useState, useCallback, useMemo, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { DatePickerWithRange } from "@/components/ui/date-range-picker";
import type { Message, Agent, Conversation } from "@/lib/types";
import {
  Search,
  Filter,
  X,
  Calendar,
  Users,
  MessageSquare,
  Clock,
  Hash,
  Bot,
  User,
  AlertCircle,
  CheckCircle,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { DateRange } from "react-day-picker";

export interface ConversationFilters {
  searchQuery: string;
  status: string[];
  participants: string[];
  messageTypes: string[];
  dateRange: DateRange | undefined;
  messageCountRange: [number, number];
  durationRange: [number, number];
  hasErrors: boolean;
  isLive: boolean;
  threadCount: [number, number];
  agentTypes: string[];
}

export interface ConversationSearchProps {
  conversations: Conversation[];
  agents: Agent[];
  filters: ConversationFilters;
  onFiltersChange: (filters: ConversationFilters) => void;
  onSearch: (query: string) => void;
  searchResults?: {
    conversations: string[];
    messages: { conversationId: string; messageId: string; snippet: string }[];
    totalResults: number;
  };
  className?: string;
}

const defaultFilters: ConversationFilters = {
  searchQuery: "",
  status: [],
  participants: [],
  messageTypes: [],
  dateRange: undefined,
  messageCountRange: [0, 1000],
  durationRange: [0, 120], // minutes
  hasErrors: false,
  isLive: false,
  threadCount: [0, 10],
  agentTypes: [],
};

export function ConversationSearch({
  conversations,
  agents,
  filters,
  onFiltersChange,
  onSearch,
  searchResults,
  className,
}: ConversationSearchProps) {
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [searchValue, setSearchValue] = useState(filters.searchQuery);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchValue !== filters.searchQuery) {
        onSearch(searchValue);
        onFiltersChange({ ...filters, searchQuery: searchValue });
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchValue, filters, onFiltersChange, onSearch]);

  // Calculate filter statistics
  const filterStats = useMemo(() => {
    const statusCounts = conversations.reduce(
      (acc, conv) => {
        const status = conv.endTime ? "completed" : "active";
        acc[status] = (acc[status] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );

    const participantCounts = agents.reduce(
      (acc, agent) => {
        const count = conversations.filter((conv) =>
          conv.participants?.includes(agent.id),
        ).length;
        if (count > 0) acc[agent.id] = count;
        return acc;
      },
      {} as Record<string, number>,
    );

    const messageTypeCounts = conversations.reduce(
      (acc, conv) => {
        conv.messages?.forEach((msg) => {
          const type = msg.metadata?.type || "regular";
          acc[type] = (acc[type] || 0) + 1;
        });
        return acc;
      },
      {} as Record<string, number>,
    );

    return {
      statusCounts,
      participantCounts,
      messageTypeCounts,
      totalConversations: conversations.length,
      activeConversations: statusCounts.active || 0,
      completedConversations: statusCounts.completed || 0,
    };
  }, [conversations, agents]);

  // Handle filter updates
  const updateFilter = useCallback(
    (key: keyof ConversationFilters, value: any) => {
      onFiltersChange({ ...filters, [key]: value });
    },
    [filters, onFiltersChange],
  );

  // Clear all filters
  const clearAllFilters = useCallback(() => {
    setSearchValue("");
    onFiltersChange(defaultFilters);
  }, [onFiltersChange]);

  // Get active filter count
  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.searchQuery) count++;
    if (filters.status.length > 0) count++;
    if (filters.participants.length > 0) count++;
    if (filters.messageTypes.length > 0) count++;
    if (filters.dateRange) count++;
    if (filters.messageCountRange[0] > 0 || filters.messageCountRange[1] < 1000)
      count++;
    if (filters.durationRange[0] > 0 || filters.durationRange[1] < 120) count++;
    if (filters.hasErrors) count++;
    if (filters.isLive) count++;
    if (filters.threadCount[0] > 0 || filters.threadCount[1] < 10) count++;
    if (filters.agentTypes.length > 0) count++;
    return count;
  }, [filters]);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
        <Input
          placeholder="Search conversations, messages, participants..."
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
          className="pl-10 pr-12"
        />
        {searchValue && (
          <Button
            variant="ghost"
            size="sm"
            className="absolute right-1 top-1/2 transform -translate-y-1/2 w-8 h-8 p-0"
            onClick={() => setSearchValue("")}
          >
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Filter Controls and Stats */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Popover open={isFilterOpen} onOpenChange={setIsFilterOpen}>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <Filter className="w-4 h-4" />
                Filters
                {activeFilterCount > 0 && (
                  <Badge
                    variant="secondary"
                    className="ml-1 px-1.5 py-0 text-xs"
                  >
                    {activeFilterCount}
                  </Badge>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-96 p-4" align="start">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">Conversation Filters</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearAllFilters}
                    className="text-xs"
                  >
                    Clear All
                  </Button>
                </div>

                {/* Status Filter */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Status</Label>
                  <div className="flex gap-2">
                    {Object.entries(filterStats.statusCounts).map(
                      ([status, count]) => (
                        <div
                          key={status}
                          className="flex items-center space-x-2"
                        >
                          <Checkbox
                            id={`status-${status}`}
                            checked={filters.status.includes(status)}
                            onCheckedChange={(checked) => {
                              const newStatus = checked
                                ? [...filters.status, status]
                                : filters.status.filter((s) => s !== status);
                              updateFilter("status", newStatus);
                            }}
                          />
                          <Label
                            htmlFor={`status-${status}`}
                            className="text-sm"
                          >
                            {status} ({count})
                          </Label>
                        </div>
                      ),
                    )}
                  </div>
                </div>

                {/* Participants Filter */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Participants</Label>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {agents
                      .filter(
                        (agent) => filterStats.participantCounts[agent.id],
                      )
                      .map((agent) => {
                        const count = filterStats.participantCounts[agent.id];
                        return (
                          <div
                            key={agent.id}
                            className="flex items-center space-x-2"
                          >
                            <Checkbox
                              id={`participant-${agent.id}`}
                              checked={filters.participants.includes(agent.id)}
                              onCheckedChange={(checked) => {
                                const newParticipants = checked
                                  ? [...filters.participants, agent.id]
                                  : filters.participants.filter(
                                      (p) => p !== agent.id,
                                    );
                                updateFilter("participants", newParticipants);
                              }}
                            />
                            <Label
                              htmlFor={`participant-${agent.id}`}
                              className="text-sm flex items-center gap-2"
                            >
                              <div
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: agent.color }}
                              />
                              {agent.name} ({count})
                            </Label>
                          </div>
                        );
                      })}
                  </div>
                </div>

                {/* Message Types Filter */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Message Types</Label>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(filterStats.messageTypeCounts).map(
                      ([type, count]) => (
                        <div key={type} className="flex items-center space-x-2">
                          <Checkbox
                            id={`type-${type}`}
                            checked={filters.messageTypes.includes(type)}
                            onCheckedChange={(checked) => {
                              const newTypes = checked
                                ? [...filters.messageTypes, type]
                                : filters.messageTypes.filter(
                                    (t) => t !== type,
                                  );
                              updateFilter("messageTypes", newTypes);
                            }}
                          />
                          <Label htmlFor={`type-${type}`} className="text-sm">
                            {type} ({count})
                          </Label>
                        </div>
                      ),
                    )}
                  </div>
                </div>

                {/* Date Range Filter */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Date Range</Label>
                  <DatePickerWithRange
                    date={filters.dateRange}
                    onDateChange={(dateRange) =>
                      updateFilter("dateRange", dateRange)
                    }
                  />
                </div>

                {/* Message Count Range */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">
                    Message Count: {filters.messageCountRange[0]} -{" "}
                    {filters.messageCountRange[1]}
                  </Label>
                  <Slider
                    value={filters.messageCountRange}
                    onValueChange={(value) =>
                      updateFilter("messageCountRange", value)
                    }
                    min={0}
                    max={1000}
                    step={10}
                    className="w-full"
                  />
                </div>

                {/* Duration Range */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">
                    Duration (minutes): {filters.durationRange[0]} -{" "}
                    {filters.durationRange[1]}
                  </Label>
                  <Slider
                    value={filters.durationRange}
                    onValueChange={(value) =>
                      updateFilter("durationRange", value)
                    }
                    min={0}
                    max={120}
                    step={5}
                    className="w-full"
                  />
                </div>

                {/* Thread Count Range */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">
                    Thread Count: {filters.threadCount[0]} -{" "}
                    {filters.threadCount[1]}
                  </Label>
                  <Slider
                    value={filters.threadCount}
                    onValueChange={(value) =>
                      updateFilter("threadCount", value)
                    }
                    min={0}
                    max={10}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Boolean Filters */}
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="has-errors"
                      checked={filters.hasErrors}
                      onCheckedChange={(checked) =>
                        updateFilter("hasErrors", checked)
                      }
                    />
                    <Label htmlFor="has-errors" className="text-sm">
                      Has errors or issues
                    </Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="is-live"
                      checked={filters.isLive}
                      onCheckedChange={(checked) =>
                        updateFilter("isLive", checked)
                      }
                    />
                    <Label htmlFor="is-live" className="text-sm">
                      Live conversations only
                    </Label>
                  </div>
                </div>
              </div>
            </PopoverContent>
          </Popover>

          {/* Quick Filter Badges */}
          {filters.status.length > 0 && (
            <Badge variant="outline" className="gap-1">
              Status: {filters.status.join(", ")}
              <X
                className="w-3 h-3 cursor-pointer"
                onClick={() => updateFilter("status", [])}
              />
            </Badge>
          )}

          {filters.participants.length > 0 && (
            <Badge variant="outline" className="gap-1">
              <Users className="w-3 h-3" />
              {filters.participants.length} participants
              <X
                className="w-3 h-3 cursor-pointer"
                onClick={() => updateFilter("participants", [])}
              />
            </Badge>
          )}
        </div>

        {/* Filter Stats */}
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-1">
            <MessageSquare className="w-4 h-4" />
            {filterStats.totalConversations} total
          </span>
          <span className="flex items-center gap-1">
            <CheckCircle className="w-4 h-4 text-green-500" />
            {filterStats.activeConversations} active
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            {filterStats.completedConversations} completed
          </span>
        </div>
      </div>

      {/* Search Results */}
      {searchResults && searchResults.totalResults > 0 && (
        <div className="border rounded-lg p-4 bg-muted/50">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold">
              Search Results ({searchResults.totalResults})
            </h4>
            <Badge variant="secondary">
              {searchResults.conversations.length} conversations,{" "}
              {searchResults.messages.length} messages
            </Badge>
          </div>

          {/* Message results preview */}
          {searchResults.messages.length > 0 && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Message matches:</Label>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {searchResults.messages.slice(0, 5).map((result, index) => (
                  <div
                    key={index}
                    className="text-sm p-2 bg-background rounded border"
                  >
                    <div className="font-medium text-xs text-muted-foreground mb-1">
                      Conversation {result.conversationId.substring(0, 8)}
                    </div>
                    <div className="line-clamp-2">{result.snippet}</div>
                  </div>
                ))}
                {searchResults.messages.length > 5 && (
                  <div className="text-xs text-muted-foreground text-center py-1">
                    +{searchResults.messages.length - 5} more matches
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
