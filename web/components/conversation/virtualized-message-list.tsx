"use client";

import React, { memo, useCallback, useMemo, useRef, useEffect, useState } from "react";
import { FixedSizeList as List, ListChildComponentProps } from "react-window";
import { format } from "date-fns";
import { 
  Avatar, 
  AvatarFallback, 
  AvatarImage 
} from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Message, Agent, ConversationThread } from "@/lib/types";
import { 
  CornerDownRight, 
  MessageSquare, 
  Clock, 
  User, 
  Bot,
  AlertCircle,
  CheckCircle,
  XCircle,
  Loader2,
  Hash,
  ThumbsUp,
  ThumbsDown,
  Reply,
  MoreHorizontal
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface VirtualizedMessageListProps {
  messages: Message[];
  agents: Agent[];
  threads?: ConversationThread[];
  height: number;
  onMessageClick?: (message: Message) => void;
  onReply?: (message: Message) => void;
  onReaction?: (messageId: string, type: string) => void;
  showMetadata?: boolean;
  showThreads?: boolean;
  showTypingIndicators?: boolean;
  typingAgents?: Record<string, { text?: string; messageId?: string }>;
  className?: string;
}

interface MessageItemProps extends ListChildComponentProps {
  data: {
    messages: Message[];
    agents: Agent[];
    threads?: ConversationThread[];
    onMessageClick?: (message: Message) => void;
    onReply?: (message: Message) => void;
    onReaction?: (messageId: string, type: string) => void;
    showMetadata: boolean;
    showThreads: boolean;
    typingAgents?: Record<string, { text?: string; messageId?: string }>;
  };
}

const MessageItem = memo<MessageItemProps>(({ index, style, data }) => {
  const {
    messages,
    agents,
    threads,
    onMessageClick,
    onReply,
    onReaction,
    showMetadata,
    showThreads,
    typingAgents = {}
  } = data;

  const message = messages[index];
  if (!message) return null;

  // Find the agent who sent this message
  const sender = agents.find(agent => agent.id === message.senderId);
  
  // Determine if this is a system message
  const isSystemMessage = message.metadata?.isSystemMessage || message.senderId === 'system';
  
  // Get thread information
  const messageThread = threads?.find(thread => 
    thread.id === message.metadata?.threadId
  );
  
  // Check if this message is being responded to
  const isBeingRespondedTo = message.metadata?.respondingTo && 
    Object.values(typingAgents).some(agent => agent.messageId === message.id);

  // Get the parent message if this is a reply
  const parentMessage = message.metadata?.respondingTo ? 
    messages.find(m => m.id === message.metadata?.respondingTo) : null;

  // Calculate message depth for thread visualization
  const threadDepth = calculateThreadDepth(message, messages);

  const handleMessageClick = () => {
    onMessageClick?.(message);
  };

  const handleReply = (e: React.MouseEvent) => {
    e.stopPropagation();
    onReply?.(message);
  };

  const handleReaction = (type: string) => {
    onReaction?.(message.id, type);
  };

  if (isSystemMessage) {
    return (
      <div style={style} className="px-4 py-2">
        <div className="flex justify-center">
          <div className="max-w-2xl">
            <div className="bg-muted/50 rounded-lg px-4 py-2 text-center text-sm text-muted-foreground">
              <AlertCircle className="inline-block w-4 h-4 mr-2" />
              {message.content}
              {showMetadata && message.timestamp && (
                <span className="ml-2 text-xs">
                  {format(new Date(message.timestamp), 'HH:mm:ss')}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={style} className="px-4 py-2 hover:bg-muted/20 transition-colors">
      <div 
        className={cn(
          "group cursor-pointer",
          threadDepth > 0 && "ml-8 border-l-2 border-muted pl-4"
        )}
        onClick={handleMessageClick}
      >
        {/* Thread connection line */}
        {showThreads && parentMessage && (
          <div className="flex items-center text-xs text-muted-foreground mb-2">
            <CornerDownRight className="w-3 h-3 mr-1" />
            <span>Replying to: {parentMessage.content.substring(0, 30)}...</span>
          </div>
        )}

        <div className="flex gap-3">
          {/* Avatar */}
          <Avatar className="w-8 h-8 flex-shrink-0">
            <AvatarImage src={sender?.avatar} />
            <AvatarFallback style={{ backgroundColor: sender?.color || '#666' }}>
              {sender ? sender.name.charAt(0).toUpperCase() : 
               message.senderId === 'user' ? <User className="w-4 h-4" /> : 
               <Bot className="w-4 h-4" />}
            </AvatarFallback>
          </Avatar>

          <div className="flex-1 min-w-0">
            {/* Message header */}
            <div className="flex items-center gap-2 mb-1">
              <span className="font-semibold text-sm">
                {sender?.name || 
                 (message.senderId === 'user' ? 'You' : message.senderId)}
              </span>
              
              {/* Agent type badge */}
              {message.metadata?.agentType && (
                <Badge variant="outline" className="text-xs">
                  {message.metadata.agentType}
                </Badge>
              )}
              
              {/* AI generated badge */}
              {message.metadata?.isGeneratedByLLM && (
                <Badge variant="secondary" className="text-xs">
                  <Bot className="w-3 h-3 mr-1" />
                  AI
                </Badge>
              )}
              
              {/* Message type badge */}
              {message.metadata?.type && message.metadata.type !== 'agent' && (
                <Badge variant="outline" className="text-xs">
                  {message.metadata.type}
                </Badge>
              )}
              
              {/* Timestamp */}
              <span className="text-xs text-muted-foreground">
                {format(new Date(message.timestamp), 'HH:mm:ss')}
              </span>
              
              {/* Thread indicator */}
              {showThreads && messageThread && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Hash className="w-3 h-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Thread: {messageThread.topic || 'Untitled'}</p>
                      <p className="text-xs">{messageThread.messageCount} messages</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>

            {/* Message content */}
            <div className="text-sm mb-2">
              {message.content}
            </div>

            {/* Message metadata */}
            {showMetadata && (
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {/* Confidence */}
                {message.metadata?.confidence && (
                  <span>
                    Confidence: {Math.round(message.metadata.confidence * 100)}%
                  </span>
                )}
                
                {/* Processing time */}
                {message.metadata?.processingTime && (
                  <span>
                    <Clock className="inline w-3 h-3 mr-1" />
                    {message.metadata.processingTime}ms
                  </span>
                )}
                
                {/* Delivery status */}
                {message.metadata?.deliveryStatus && (
                  <span className="flex items-center gap-1">
                    {message.metadata.deliveryStatus === 'delivered' && (
                      <CheckCircle className="w-3 h-3 text-green-500" />
                    )}
                    {message.metadata.deliveryStatus === 'failed' && (
                      <XCircle className="w-3 h-3 text-red-500" />
                    )}
                    {message.metadata.deliveryStatus === 'pending' && (
                      <Loader2 className="w-3 h-3 animate-spin text-yellow-500" />
                    )}
                    {message.metadata.deliveryStatus}
                  </span>
                )}
                
                {/* Topics */}
                {message.metadata?.topics && message.metadata.topics.length > 0 && (
                  <div className="flex gap-1">
                    {message.metadata.topics.slice(0, 3).map((topic: string) => (
                      <Badge key={topic} variant="outline" className="text-xs">
                        {topic}
                      </Badge>
                    ))}
                    {message.metadata.topics.length > 3 && (
                      <span>+{message.metadata.topics.length - 3} more</span>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Reactions */}
            {message.metadata?.reactions && message.metadata.reactions.length > 0 && (
              <div className="flex gap-1 mt-2">
                {message.metadata.reactions.map((reaction: any) => (
                  <Badge 
                    key={`${reaction.agentId}-${reaction.type}`} 
                    variant="outline" 
                    className="text-xs cursor-pointer hover:bg-muted"
                    onClick={() => handleReaction(reaction.type)}
                  >
                    {reaction.type} {reaction.agentId}
                  </Badge>
                ))}
              </div>
            )}

            {/* Response indicator */}
            {isBeingRespondedTo && (
              <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Agents are responding...</span>
              </div>
            )}

            {/* Action buttons (shown on hover) */}
            <div className="flex gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={handleReply}
              >
                <Reply className="w-3 h-3 mr-1" />
                Reply
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleReaction('👍')}
              >
                <ThumbsUp className="w-3 h-3" />
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleReaction('👎')}
              >
                <ThumbsDown className="w-3 h-3" />
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
              >
                <MoreHorizontal className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

MessageItem.displayName = 'MessageItem';

// Helper function to calculate thread depth
function calculateThreadDepth(message: Message, messages: Message[]): number {
  let depth = 0;
  let currentMessage = message;
  
  while (currentMessage.metadata?.parentMessageId) {
    const parent = messages.find(m => m.id === currentMessage.metadata?.parentMessageId);
    if (!parent) break;
    depth++;
    currentMessage = parent;
    if (depth > 10) break; // Prevent infinite loops
  }
  
  return depth;
}

export const VirtualizedMessageList = memo<VirtualizedMessageListProps>(({
  messages,
  agents,
  threads,
  height,
  onMessageClick,
  onReply,
  onReaction,
  showMetadata = true,
  showThreads = true,
  showTypingIndicators = true,
  typingAgents = {},
  className
}) => {
  const listRef = useRef<List>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (shouldAutoScroll && listRef.current) {
      listRef.current.scrollToItem(messages.length - 1, 'end');
    }
  }, [messages.length, shouldAutoScroll]);

  // Handle scroll to detect if user is at bottom
  const handleScroll = useCallback(({ scrollTop, clientHeight, scrollHeight }: any) => {
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - 50;
    setShouldAutoScroll(isNearBottom);
  }, []);

  // Create typing indicator messages
  const typingMessages = useMemo(() => {
    return Object.entries(typingAgents).map(([agentId, info]) => ({
      id: `typing-${agentId}`,
      content: info.text || '...',
      senderId: agentId,
      timestamp: new Date(),
      metadata: {
        isTyping: true,
        type: 'typing' as const,
        respondingTo: info.messageId
      }
    })) as Message[];
  }, [typingAgents]);

  // Combine messages with typing indicators
  const allMessages = useMemo(() => {
    const combined = [...messages];
    if (showTypingIndicators) {
      combined.push(...typingMessages);
    }
    return combined;
  }, [messages, typingMessages, showTypingIndicators]);

  const itemData = useMemo(() => ({
    messages: allMessages,
    agents,
    threads,
    onMessageClick,
    onReply,
    onReaction,
    showMetadata,
    showThreads,
    typingAgents
  }), [
    allMessages,
    agents,
    threads,
    onMessageClick,
    onReply,
    onReaction,
    showMetadata,
    showThreads,
    typingAgents
  ]);

  // Estimate item size based on content
  const estimateItemSize = useCallback(() => {
    // Base size for avatar and padding
    let baseSize = 60;
    
    // Add size for metadata
    if (showMetadata) baseSize += 20;
    
    // Add size for thread indicators
    if (showThreads) baseSize += 10;
    
    return baseSize;
  }, [showMetadata, showThreads]);

  const itemSize = estimateItemSize();

  if (allMessages.length === 0) {
    return (
      <div className={cn("flex items-center justify-center h-full", className)}>
        <div className="text-center text-muted-foreground">
          <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No messages yet</p>
          <p className="text-sm">Start a conversation to see messages here</p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("relative", className)}>
      <List
        ref={listRef}
        height={height}
        width="100%"
        itemCount={allMessages.length}
        itemSize={itemSize}
        itemData={itemData}
        onScroll={handleScroll}
        overscanCount={5}
        className="scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent"
      >
        {MessageItem}
      </List>
      
      {/* Scroll to bottom button */}
      {!shouldAutoScroll && (
        <div className="absolute bottom-4 right-4">
          <Button
            variant="outline"
            size="sm"
            className="shadow-lg"
            onClick={() => {
              setShouldAutoScroll(true);
              listRef.current?.scrollToItem(allMessages.length - 1, 'end');
            }}
          >
            <CornerDownRight className="w-4 h-4 mr-1" />
            Jump to bottom
          </Button>
        </div>
      )}
    </div>
  );
});

VirtualizedMessageList.displayName = 'VirtualizedMessageList'; 