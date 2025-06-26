"use client";

import React, { useState, useCallback, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ConversationSearch, ConversationFilters } from "./conversation-search";
import { MessageQueueVisualization, QueuedMessage, QueueMetrics } from "./message-queue-visualization";
import { VirtualizedMessageList } from "./virtualized-message-list";
import { useAutoScroll } from "@/hooks/useAutoScroll";
import { useConversationWebSocket } from "@/hooks/useConversationWebSocket";
import type { Message, Agent, Conversation, ConversationThread } from "@/lib/types";
import { 
  MessageSquare, 
  Users, 
  Settings, 
  Search, 
  Filter,
  Activity,
  BarChart3,
  Play,
  Pause,
  ArrowDown,
  ArrowUp,
  Maximize2,
  Minimize2,
  RefreshCw,
  Eye,
  EyeOff,
  Hash,
  Clock,
  Zap,
  AlertTriangle,
  TrendingUp,
  ChevronRight,
  ChevronDown
} from "lucide-react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

export interface ConversationDashboardProps {
  conversations: Conversation[];
  agents: Agent[];
  selectedConversationId?: string;
  onConversationSelect: (conversationId: string) => void;
  onSendMessage?: (conversationId: string, content: string, senderId: string) => void;
  className?: string;
}

// Mock data generators for demonstration
const generateMockQueue = (): QueuedMessage[] => {
  return Array.from({ length: 12 }, (_, i) => ({
    id: `queue-${i}`,
    conversationId: `conv-${Math.floor(i / 3)}`,
    agentId: `agent-${i % 4}`,
    type: ['response', 'autonomous', 'tool_call', 'retry'][i % 4] as any,
    priority: ['low', 'normal', 'high', 'urgent'][i % 4] as any,
    status: ['queued', 'processing', 'completed', 'failed'][i % 4] as any,
    queuedAt: new Date(Date.now() - Math.random() * 300000),
    progress: i % 4 === 1 ? Math.random() * 100 : undefined,
    estimatedDuration: 2000 + Math.random() * 8000,
    metadata: {
      messageContent: `Sample message content ${i}...`,
      modelUsed: ['gpt-4', 'claude-3', 'llama-2'][i % 3],
      confidence: Math.random()
    }
  }));
};

const generateMockMetrics = (): QueueMetrics => ({
  totalQueued: 5,
  totalProcessing: 3,
  totalCompleted: 42,
  totalFailed: 2,
  averageProcessingTime: 3500,
  averageQueueTime: 1200,
  throughputPerMinute: 8.5,
  errorRate: 0.04,
  queuedByPriority: { urgent: 1, high: 2, normal: 2, low: 0 },
  processingByAgent: { 'agent-1': 2, 'agent-2': 1 },
  conversationLoad: { 'conv-1': 3, 'conv-2': 2 }
});

export function ConversationDashboard({
  conversations,
  agents,
  selectedConversationId,
  onConversationSelect,
  onSendMessage,
  className
}: ConversationDashboardProps) {
  const [filters, setFilters] = useState<ConversationFilters>({
    searchQuery: "",
    status: [],
    participants: [],
    messageTypes: [],
    dateRange: undefined,
    messageCountRange: [0, 1000],
    durationRange: [0, 120],
    hasErrors: false,
    isLive: false,
    threadCount: [0, 10],
    agentTypes: []
  });

  const [searchResults, setSearchResults] = useState<any>(null);
  const [selectedTab, setSelectedTab] = useState<'conversations' | 'queue' | 'analytics'>('conversations');
  const [showAdvancedControls, setShowAdvancedControls] = useState(false);
  const [isQueuePaused, setIsQueuePaused] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    search: true,
    queue: true,
    controls: false
  });

  // Mock data
  const [mockQueue] = useState(generateMockQueue());
  const [mockMetrics] = useState(generateMockMetrics());

  // Refs for auto-scroll
  const conversationListRef = useRef<HTMLDivElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);

  // Auto-scroll for conversations list
  const conversationAutoScroll = useAutoScroll(
    conversationListRef,
    [conversations],
    {
      threshold: 100,
      enableUserOverride: true,
      overrideTimeout: 10000,
      onScrollStateChange: (enabled, atBottom) => {
        console.log('Conversation scroll state:', { enabled, atBottom });
      }
    }
  );

  // Auto-scroll for message list
  const messageAutoScroll = useAutoScroll(
    messageListRef,
    [selectedConversationId],
    {
      threshold: 50,
      enableUserOverride: true,
      overrideTimeout: 5000
    }
  );

  // WebSocket connection
  const {
    isConnected,
    isConnecting,
    error: wsError,
    connectionStats
  } = useConversationWebSocket({
    autoConnect: true,
    onEvent: (event) => {
      console.log('Dashboard received WebSocket event:', event);
    }
  });

  // Filter conversations
  const filteredConversations = useMemo(() => {
    return conversations.filter(conversation => {
      // Search query
      if (filters.searchQuery) {
        const query = filters.searchQuery.toLowerCase();
        const matchesContent = conversation.messages?.some(msg => 
          msg.content.toLowerCase().includes(query)
        );
        const matchesParticipants = conversation.participants?.some(pid => {
          const agent = agents.find(a => a.id === pid);
          return agent?.name.toLowerCase().includes(query);
        });
        if (!matchesContent && !matchesParticipants) return false;
      }

      // Status filter
      if (filters.status.length > 0) {
        const status = conversation.endTime ? 'completed' : 'active';
        if (!filters.status.includes(status)) return false;
      }

      // Participants filter
      if (filters.participants.length > 0) {
        const hasMatchingParticipant = filters.participants.some(pid =>
          conversation.participants?.includes(pid)
        );
        if (!hasMatchingParticipant) return false;
      }

      // Message count filter
      const messageCount = conversation.messages?.length || 0;
      if (messageCount < filters.messageCountRange[0] || 
          messageCount > filters.messageCountRange[1]) {
        return false;
      }

      // Date range filter
      if (filters.dateRange?.from || filters.dateRange?.to) {
        const startTime = new Date(conversation.startTime);
        if (filters.dateRange.from && startTime < filters.dateRange.from) return false;
        if (filters.dateRange.to && startTime > filters.dateRange.to) return false;
      }

      return true;
    });
  }, [conversations, agents, filters]);

  // Get selected conversation
  const selectedConversation = conversations.find(
    conv => conv.id === selectedConversationId
  );

  // Handle search
  const handleSearch = useCallback((query: string) => {
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }

    // Mock search results
    const mockResults = {
      conversations: conversations.slice(0, 3).map(c => c.id),
      messages: conversations.slice(0, 2).flatMap(conv =>
        conv.messages?.slice(0, 2).map(msg => ({
          conversationId: conv.id,
          messageId: msg.id,
          snippet: msg.content.substring(0, 100) + '...'
        })) || []
      ),
      totalResults: 8
    };

    setSearchResults(mockResults);
  }, [conversations]);

  // Handle queue actions
  const handleCancelMessage = useCallback((messageId: string) => {
    console.log('Canceling message:', messageId);
  }, []);

  const handleRetryMessage = useCallback((messageId: string) => {
    console.log('Retrying message:', messageId);
  }, []);

  const handlePauseQueue = useCallback(() => {
    setIsQueuePaused(true);
  }, []);

  const handleResumeQueue = useCallback(() => {
    setIsQueuePaused(false);
  }, []);

  // Toggle section expansion
  const toggleSection = useCallback((section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  }, []);

  return (
    <TooltipProvider>
      <div className={cn("h-full flex flex-col", className)}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold">Conversation Dashboard</h1>
            
            {/* Connection Status */}
            <div className="flex items-center gap-2">
              <div className={cn(
                "w-2 h-2 rounded-full",
                isConnected ? "bg-green-500" : isConnecting ? "bg-yellow-500" : "bg-red-500"
              )} />
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Connected' : isConnecting ? 'Connecting...' : 'Disconnected'}
              </span>
              {connectionStats && (
                <Badge variant="outline" className="text-xs">
                  {connectionStats.total_connections} clients
                </Badge>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAdvancedControls(!showAdvancedControls)}
                >
                  <Settings className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Advanced Controls</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Refresh Data</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex">
          <ResizablePanelGroup direction="horizontal">
            {/* Left Panel - Controls and Conversations */}
            <ResizablePanel defaultSize={30} minSize={25}>
              <div className="h-full flex flex-col">
                {/* Search and Filters */}
                <div className="border-b">
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold">Search & Filter</h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleSection('search')}
                      >
                        {expandedSections.search ? 
                          <ChevronDown className="w-4 h-4" /> : 
                          <ChevronRight className="w-4 h-4" />
                        }
                      </Button>
                    </div>
                    
                    {expandedSections.search && (
                      <ConversationSearch
                        conversations={conversations}
                        agents={agents}
                        filters={filters}
                        onFiltersChange={setFilters}
                        onSearch={handleSearch}
                        searchResults={searchResults}
                      />
                    )}
                  </div>
                </div>

                {/* Conversation List */}
                <div className="flex-1 min-h-0">
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold">
                        Conversations ({filteredConversations.length})
                      </h3>
                      
                      {/* Auto-scroll controls */}
                      <div className="flex items-center gap-1">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => conversationAutoScroll.scrollToTop()}
                            >
                              <ArrowUp className="w-4 h-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Scroll to top</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => conversationAutoScroll.scrollToBottom()}
                            >
                              <ArrowDown className="w-4 h-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Scroll to bottom</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={conversationAutoScroll.toggleAutoScroll}
                              className={cn(
                                conversationAutoScroll.state.isAutoScrollEnabled ? 
                                "text-blue-500" : "text-muted-foreground"
                              )}
                            >
                              {conversationAutoScroll.state.isAutoScrollEnabled ? 
                                <Eye className="w-4 h-4" /> : 
                                <EyeOff className="w-4 h-4" />
                              }
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>
                              {conversationAutoScroll.state.isAutoScrollEnabled ? 
                                'Disable' : 'Enable'} auto-scroll
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </div>

                    {/* Conversation List */}
                    <div 
                      ref={conversationListRef}
                      className="space-y-2 max-h-96 overflow-y-auto"
                    >
                      {filteredConversations.map(conversation => (
                        <Card
                          key={conversation.id}
                          className={cn(
                            "cursor-pointer transition-all hover:shadow-md",
                            selectedConversationId === conversation.id && 
                            "border-primary bg-primary/5"
                          )}
                          onClick={() => onConversationSelect(conversation.id)}
                        >
                          <CardContent className="p-3">
                            <div className="flex justify-between items-start mb-2">
                              <div className="flex items-center gap-2">
                                <MessageSquare className="w-4 h-4" />
                                <span className="font-medium text-sm">
                                  {conversation.id.substring(0, 8)}
                                </span>
                              </div>
                              <Badge 
                                variant={conversation.endTime ? "secondary" : "default"}
                                className="text-xs"
                              >
                                {conversation.endTime ? 'Completed' : 'Active'}
                              </Badge>
                            </div>

                            <div className="text-xs text-muted-foreground space-y-1">
                              <div className="flex items-center gap-1">
                                <Users className="w-3 h-3" />
                                <span>{conversation.participants?.length || 0} participants</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <MessageSquare className="w-3 h-3" />
                                <span>{conversation.messages?.length || 0} messages</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                <span>{format(new Date(conversation.startTime), 'HH:mm')}</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}

                      {filteredConversations.length === 0 && (
                        <div className="text-center py-8 text-muted-foreground">
                          <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                          <p>No conversations match your filters</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </ResizablePanel>

            <ResizableHandle withHandle />

            {/* Center Panel - Message View */}
            <ResizablePanel defaultSize={45} minSize={30}>
              <div className="h-full flex flex-col">
                {selectedConversation ? (
                  <>
                    {/* Message Header */}
                    <div className="border-b p-4">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold">
                          Conversation {selectedConversation.id.substring(0, 8)}
                        </h3>
                        
                        {/* Message controls */}
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            Progress: {Math.round(messageAutoScroll.state.scrollProgress * 100)}%
                          </Badge>
                          
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={messageAutoScroll.jumpToLatest}
                                disabled={messageAutoScroll.state.isAtBottom}
                              >
                                <ArrowDown className="w-4 h-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Jump to latest</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                      </div>
                    </div>

                    {/* Message List */}
                    <div ref={messageListRef} className="flex-1 min-h-0">
                      <VirtualizedMessageList
                        messages={selectedConversation.messages || []}
                        agents={agents}
                        height={500}
                        onMessageClick={(message) => console.log('Message clicked:', message)}
                        onReply={(message) => console.log('Reply to:', message)}
                        onReaction={(messageId, type) => console.log('Reaction:', type, messageId)}
                        className="h-full"
                      />
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-muted-foreground">
                      <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <h3 className="text-lg font-semibold mb-2">No Conversation Selected</h3>
                      <p>Select a conversation from the list to view messages</p>
                    </div>
                  </div>
                )}
              </div>
            </ResizablePanel>

            <ResizableHandle withHandle />

            {/* Right Panel - Analytics and Queue */}
            <ResizablePanel defaultSize={25} minSize={20}>
              <Tabs value={selectedTab} onValueChange={(value: any) => setSelectedTab(value)}>
                <div className="border-b p-4">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="conversations" className="text-xs">Stats</TabsTrigger>
                    <TabsTrigger value="queue" className="text-xs">Queue</TabsTrigger>
                    <TabsTrigger value="analytics" className="text-xs">Analytics</TabsTrigger>
                  </TabsList>
                </div>

                <div className="p-4">
                  <TabsContent value="conversations" className="mt-0 space-y-4">
                    <div className="grid grid-cols-2 gap-2">
                      <Card className="p-3">
                        <div className="text-center">
                          <div className="text-2xl font-bold">{conversations.length}</div>
                          <div className="text-xs text-muted-foreground">Total</div>
                        </div>
                      </Card>
                      <Card className="p-3">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-500">
                            {conversations.filter(c => !c.endTime).length}
                          </div>
                          <div className="text-xs text-muted-foreground">Active</div>
                        </div>
                      </Card>
                    </div>
                  </TabsContent>

                  <TabsContent value="queue" className="mt-0">
                    <MessageQueueVisualization
                      queue={mockQueue}
                      agents={agents}
                      conversations={conversations}
                      metrics={mockMetrics}
                      onCancelMessage={handleCancelMessage}
                      onRetryMessage={handleRetryMessage}
                      onPauseQueue={handlePauseQueue}
                      onResumeQueue={handleResumeQueue}
                      isPaused={isQueuePaused}
                    />
                  </TabsContent>

                  <TabsContent value="analytics" className="mt-0">
                    <div className="space-y-4">
                      <Card className="p-4">
                        <div className="text-center">
                          <TrendingUp className="w-8 h-8 mx-auto mb-2 text-blue-500" />
                          <div className="text-lg font-semibold">Analytics</div>
                          <div className="text-sm text-muted-foreground">
                            Advanced metrics coming soon
                          </div>
                        </div>
                      </Card>
                    </div>
                  </TabsContent>
                </div>
              </Tabs>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>

        {/* Status Bar */}
        <div className="border-t px-4 py-2 text-xs text-muted-foreground flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span>
              {filteredConversations.length} of {conversations.length} conversations shown
            </span>
            {wsError && (
              <span className="text-red-500 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Connection error
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            <span>
              Auto-scroll: {messageAutoScroll.state.isAutoScrollEnabled ? 'ON' : 'OFF'}
            </span>
            <span>
              Queue: {isQueuePaused ? 'PAUSED' : 'RUNNING'}
            </span>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
} 