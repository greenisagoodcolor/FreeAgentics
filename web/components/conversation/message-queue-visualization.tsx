"use client";

import React, { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Agent, Message, Conversation } from "@/lib/types";
import { 
  Clock, 
  Loader2, 
  AlertTriangle, 
  CheckCircle, 
  Pause,
  Play,
  X,
  Users,
  MessageSquare,
  Zap,
  TrendingUp,
  Activity,
  Timer,
  Bot,
  User,
  Hash,
  BarChart3
} from "lucide-react";
import { cn } from "@/lib/utils";
import { format, formatDistanceToNow } from "date-fns";

export interface QueuedMessage {
  id: string;
  conversationId: string;
  messageId?: string; // Message being responded to
  agentId: string;
  type: 'response' | 'autonomous' | 'tool_call' | 'retry';
  priority: 'low' | 'normal' | 'high' | 'urgent';
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  queuedAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  estimatedDuration?: number; // in milliseconds
  actualDuration?: number;
  progress?: number; // 0-100
  errorMessage?: string;
  retryCount?: number;
  metadata?: {
    messageContent?: string;
    responseLength?: number;
    modelUsed?: string;
    toolsUsed?: string[];
    confidence?: number;
  };
}

export interface QueueMetrics {
  totalQueued: number;
  totalProcessing: number;
  totalCompleted: number;
  totalFailed: number;
  averageProcessingTime: number;
  averageQueueTime: number;
  throughputPerMinute: number;
  errorRate: number;
  queuedByPriority: Record<string, number>;
  processingByAgent: Record<string, number>;
  conversationLoad: Record<string, number>;
}

export interface MessageQueueVisualizationProps {
  queue: QueuedMessage[];
  agents: Agent[];
  conversations: Conversation[];
  metrics: QueueMetrics;
  onCancelMessage?: (messageId: string) => void;
  onRetryMessage?: (messageId: string) => void;
  onPauseQueue?: () => void;
  onResumeQueue?: () => void;
  isPaused?: boolean;
  className?: string;
}

export function MessageQueueVisualization({
  queue,
  agents,
  conversations,
  metrics,
  onCancelMessage,
  onRetryMessage,
  onPauseQueue,
  onResumeQueue,
  isPaused = false,
  className
}: MessageQueueVisualizationProps) {
  const [selectedTab, setSelectedTab] = useState<'queue' | 'processing' | 'completed' | 'failed'>('queue');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Filter messages by status
  const filteredMessages = useMemo(() => {
    switch (selectedTab) {
      case 'queue':
        return queue.filter(msg => msg.status === 'queued');
      case 'processing':
        return queue.filter(msg => msg.status === 'processing');
      case 'completed':
        return queue.filter(msg => msg.status === 'completed');
      case 'failed':
        return queue.filter(msg => msg.status === 'failed');
      default:
        return queue;
    }
  }, [queue, selectedTab]);

  // Get agent info
  const getAgent = (agentId: string) => {
    return agents.find(agent => agent.id === agentId);
  };

  // Get conversation info
  const getConversation = (conversationId: string) => {
    return conversations.find(conv => conv.id === conversationId);
  };

  // Calculate queue position
  const getQueuePosition = (messageId: string) => {
    const queuedMessages = queue.filter(msg => msg.status === 'queued');
    return queuedMessages.findIndex(msg => msg.id === messageId) + 1;
  };

  // Format duration
  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  // Get priority color
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'normal': return 'text-blue-500';
      case 'low': return 'text-gray-500';
      default: return 'text-gray-500';
    }
  };

  // Get status icon
  const getStatusIcon = (status: string, progress?: number) => {
    switch (status) {
      case 'queued':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'processing':
        return progress !== undefined ? 
          <div className="relative w-4 h-4">
            <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
            <div className="absolute inset-0 flex items-center justify-center text-xs text-blue-500">
              {Math.round(progress)}
            </div>
          </div> :
          <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'cancelled':
        return <X className="w-4 h-4 text-gray-500" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  // Auto refresh effect
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Trigger refresh - in real app this would call an API
      console.log('Auto-refreshing queue visualization');
    }, 2000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Queue Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Queued</p>
                <p className="text-2xl font-bold">{metrics.totalQueued}</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Processing</p>
                <p className="text-2xl font-bold">{metrics.totalProcessing}</p>
              </div>
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold">{metrics.totalCompleted}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold">{metrics.totalFailed}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Metrics */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Timer className="w-5 h-5 text-blue-500 mr-1" />
                <span className="text-sm font-medium">Avg Processing</span>
              </div>
              <p className="text-xl font-semibold">
                {formatDuration(metrics.averageProcessingTime)}
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Clock className="w-5 h-5 text-yellow-500 mr-1" />
                <span className="text-sm font-medium">Avg Queue Time</span>
              </div>
              <p className="text-xl font-semibold">
                {formatDuration(metrics.averageQueueTime)}
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <TrendingUp className="w-5 h-5 text-green-500 mr-1" />
                <span className="text-sm font-medium">Throughput</span>
              </div>
              <p className="text-xl font-semibold">
                {metrics.throughputPerMinute.toFixed(1)}/min
              </p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <BarChart3 className="w-5 h-5 text-red-500 mr-1" />
                <span className="text-sm font-medium">Error Rate</span>
              </div>
              <p className="text-xl font-semibold">
                {(metrics.errorRate * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Queue Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={isPaused ? onResumeQueue : onPauseQueue}
            className="gap-2"
          >
            {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            {isPaused ? 'Resume Queue' : 'Pause Queue'}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className="gap-2"
          >
            <Activity className={cn("w-4 h-4", autoRefresh && "animate-pulse")} />
            Auto Refresh
          </Button>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-1">
          {[
            { id: 'queue', label: 'Queued', count: metrics.totalQueued },
            { id: 'processing', label: 'Processing', count: metrics.totalProcessing },
            { id: 'completed', label: 'Completed', count: metrics.totalCompleted },
            { id: 'failed', label: 'Failed', count: metrics.totalFailed }
          ].map(tab => (
            <Button
              key={tab.id}
              variant={selectedTab === tab.id ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedTab(tab.id as any)}
              className="gap-2"
            >
              {tab.label}
              <Badge variant="secondary" className="ml-1">
                {tab.count}
              </Badge>
            </Button>
          ))}
        </div>
      </div>

      {/* Message Queue List */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg capitalize">
            {selectedTab} Messages
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-96">
            <div className="space-y-2 p-4">
              {filteredMessages.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No {selectedTab} messages</p>
                </div>
              ) : (
                filteredMessages.map((message) => {
                  const agent = getAgent(message.agentId);
                  const conversation = getConversation(message.conversationId);
                  const queuePosition = message.status === 'queued' ? getQueuePosition(message.id) : null;

                  return (
                    <Card key={message.id} className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1">
                          {/* Status Icon */}
                          <div className="flex-shrink-0 mt-1">
                            {getStatusIcon(message.status, message.progress)}
                          </div>

                          {/* Message Info */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-2">
                              {/* Agent */}
                              <div className="flex items-center gap-1">
                                {agent ? (
                                  <>
                                    <div 
                                      className="w-3 h-3 rounded-full" 
                                      style={{ backgroundColor: agent.color }}
                                    />
                                    <span className="font-medium">{agent.name}</span>
                                  </>
                                ) : (
                                  <Bot className="w-4 h-4" />
                                )}
                              </div>

                              {/* Priority */}
                              <Badge 
                                variant="outline" 
                                className={cn("text-xs", getPriorityColor(message.priority))}
                              >
                                {message.priority}
                              </Badge>

                              {/* Type */}
                              <Badge variant="secondary" className="text-xs">
                                {message.type}
                              </Badge>

                              {/* Queue Position */}
                              {queuePosition && (
                                <Badge variant="outline" className="text-xs">
                                  #{queuePosition} in queue
                                </Badge>
                              )}
                            </div>

                            {/* Conversation */}
                            <div className="text-sm text-muted-foreground mb-2">
                              <Hash className="w-3 h-3 inline mr-1" />
                              {conversation?.id.substring(0, 8) || message.conversationId.substring(0, 8)}
                            </div>

                            {/* Message Content Preview */}
                            {message.metadata?.messageContent && (
                              <div className="text-sm text-muted-foreground mb-2 line-clamp-2">
                                {message.metadata.messageContent}
                              </div>
                            )}

                            {/* Progress Bar */}
                            {message.status === 'processing' && message.progress !== undefined && (
                              <div className="mb-2">
                                <Progress value={message.progress} className="h-2" />
                                <div className="text-xs text-muted-foreground mt-1">
                                  {message.progress}% complete
                                </div>
                              </div>
                            )}

                            {/* Timing Info */}
                            <div className="flex items-center gap-4 text-xs text-muted-foreground">
                              <span>
                                Queued {formatDistanceToNow(message.queuedAt, { addSuffix: true })}
                              </span>
                              
                              {message.startedAt && (
                                <span>
                                  Started {formatDistanceToNow(message.startedAt, { addSuffix: true })}
                                </span>
                              )}
                              
                              {message.completedAt && (
                                <span>
                                  Completed {formatDistanceToNow(message.completedAt, { addSuffix: true })}
                                </span>
                              )}
                              
                              {message.estimatedDuration && (
                                <span>
                                  ETA: {formatDuration(message.estimatedDuration)}
                                </span>
                              )}
                              
                              {message.actualDuration && (
                                <span>
                                  Duration: {formatDuration(message.actualDuration)}
                                </span>
                              )}
                            </div>

                            {/* Error Message */}
                            {message.errorMessage && (
                              <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                                {message.errorMessage}
                              </div>
                            )}

                            {/* Metadata */}
                            {message.metadata && (
                              <div className="mt-2 flex gap-2 flex-wrap">
                                {message.metadata.modelUsed && (
                                  <Badge variant="outline" className="text-xs">
                                    Model: {message.metadata.modelUsed}
                                  </Badge>
                                )}
                                {message.metadata.toolsUsed && message.metadata.toolsUsed.length > 0 && (
                                  <Badge variant="outline" className="text-xs">
                                    Tools: {message.metadata.toolsUsed.join(', ')}
                                  </Badge>
                                )}
                                {message.metadata.confidence && (
                                  <Badge variant="outline" className="text-xs">
                                    Confidence: {Math.round(message.metadata.confidence * 100)}%
                                  </Badge>
                                )}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex gap-1">
                          {message.status === 'queued' && onCancelMessage && (
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => onCancelMessage(message.id)}
                                  >
                                    <X className="w-4 h-4" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>Cancel message</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          )}

                          {message.status === 'failed' && onRetryMessage && (
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => onRetryMessage(message.id)}
                                  >
                                    <Zap className="w-4 h-4" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>Retry message</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          )}
                        </div>
                      </div>
                    </Card>
                  );
                })
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
} 