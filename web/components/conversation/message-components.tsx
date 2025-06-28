"use client";

import React, { memo } from "react";
import { format } from "date-fns";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Message, Agent } from "@/lib/types";
import {
  Bot,
  User,
  AlertCircle,
  Loader2,
  Clock,
  CheckCircle,
  XCircle,
  Hash,
  Reply,
  ThumbsUp,
  ThumbsDown,
  MoreHorizontal,
  Zap,
  Brain,
  Eye,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface MessageComponentProps {
  message: Message;
  sender?: Agent;
  showMetadata?: boolean;
  showActions?: boolean;
  onReply?: (message: Message) => void;
  onReaction?: (messageId: string, type: string) => void;
  className?: string;
}

// System Message Component
export const SystemMessage = memo<MessageComponentProps>(
  ({ message, showMetadata = true, className }) => (
    <div className={cn("flex justify-center py-2", className)}>
      <div className="max-w-2xl">
        <div className="bg-muted/50 rounded-lg px-4 py-2 text-center text-sm text-muted-foreground system-message">
          <AlertCircle className="inline-block w-4 h-4 mr-2" />
          {message.content}
          {showMetadata && message.timestamp && (
            <span className="ml-2 text-xs opacity-70">
              {format(new Date(message.timestamp), "HH:mm:ss")}
            </span>
          )}
        </div>
      </div>
    </div>
  ),
);

SystemMessage.displayName = "SystemMessage";

// Typing Indicator Component
export const TypingIndicator = memo<{
  agent: Agent;
  text?: string;
  className?: string;
}>(({ agent, text = "...", className }) => (
  <div className={cn("flex gap-3 px-4 py-2 opacity-75", className)}>
    <Avatar className="w-8 h-8 flex-shrink-0">
      <AvatarImage src={agent.avatar} />
      <AvatarFallback style={{ backgroundColor: agent.color }}>
        {agent.name.charAt(0).toUpperCase()}
      </AvatarFallback>
    </Avatar>

    <div className="flex-1 min-w-0">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-semibold text-sm">{agent.name}</span>
        <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />
        <span className="text-xs text-muted-foreground">typing...</span>
      </div>
      <div className="text-sm text-muted-foreground italic">{text}</div>
    </div>
  </div>
));

TypingIndicator.displayName = "TypingIndicator";

// Message Header Component
export const MessageHeader = memo<{
  message: Message;
  sender?: Agent;
  showMetadata?: boolean;
}>(({ message, sender, showMetadata = true }) => (
  <div className="flex items-center gap-2 mb-1 flex-wrap">
    {/* Sender name */}
    <span className="font-semibold text-sm">
      {sender?.name || (message.senderId === "user" ? "You" : message.senderId)}
    </span>

    {/* Agent type badge */}
    {message.metadata?.agentType && (
      <Badge variant="outline" className="text-xs">
        {message.metadata.agentType}
      </Badge>
    )}

    {/* Agent role badge */}
    {message.metadata?.agentRole && (
      <Badge variant="secondary" className="text-xs">
        {message.metadata.agentRole}
      </Badge>
    )}

    {/* AI generated badge */}
    {message.metadata?.isGeneratedByLLM && (
      <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-800">
        <Bot className="w-3 h-3 mr-1" />
        AI
      </Badge>
    )}

    {/* Message type badge */}
    {message.metadata?.type && message.metadata.type !== "agent" && (
      <Badge variant="outline" className="text-xs">
        {getMessageTypeIcon(message.metadata.type)}
        {message.metadata.type}
      </Badge>
    )}

    {/* Priority indicator */}
    {message.metadata?.priority && message.metadata.priority !== "normal" && (
      <Badge
        variant={
          message.metadata.priority === "urgent" ? "destructive" : "default"
        }
        className="text-xs"
      >
        {message.metadata.priority}
      </Badge>
    )}

    {/* Timestamp */}
    <span className="text-xs text-muted-foreground">
      {format(new Date(message.timestamp), "HH:mm:ss")}
    </span>

    {/* Thread indicator */}
    {message.metadata?.threadId && (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger>
            <Hash className="w-3 h-3 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent>
            <p>Thread ID: {message.metadata.threadId}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )}
  </div>
));

MessageHeader.displayName = "MessageHeader";

// Message Metadata Component
export const MessageMetadata = memo<{
  message: Message;
}>(({ message }) => {
  const metadata = message.metadata;
  if (!metadata) return null;

  return (
    <div className="flex flex-wrap gap-2 mt-2 text-xs text-muted-foreground">
      {/* Confidence */}
      {metadata.confidence && (
        <div className="flex items-center gap-1">
          <Brain className="w-3 h-3" />
          <span>Confidence: {Math.round(metadata.confidence * 100)}%</span>
        </div>
      )}

      {/* Processing time */}
      {metadata.processingTime && (
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          <span>{metadata.processingTime}ms</span>
        </div>
      )}

      {/* Performance metrics */}
      {metadata.performanceMetrics?.generationTime && (
        <div className="flex items-center gap-1">
          <Zap className="w-3 h-3" />
          <span>Generated: {metadata.performanceMetrics.generationTime}ms</span>
        </div>
      )}

      {/* Token usage */}
      {metadata.performanceMetrics?.tokens && (
        <div className="flex items-center gap-1">
          <span>
            Tokens: {metadata.performanceMetrics.tokens.input}/
            {metadata.performanceMetrics.tokens.output}
          </span>
        </div>
      )}

      {/* Model used */}
      {metadata.performanceMetrics?.modelUsed && (
        <Badge variant="outline" className="text-xs">
          {metadata.performanceMetrics.modelUsed}
        </Badge>
      )}

      {/* Delivery status */}
      {metadata.deliveryStatus && (
        <div className="flex items-center gap-1">
          {metadata.deliveryStatus === "delivered" && (
            <CheckCircle className="w-3 h-3 text-green-500" />
          )}
          {metadata.deliveryStatus === "failed" && (
            <XCircle className="w-3 h-3 text-red-500" />
          )}
          {metadata.deliveryStatus === "pending" && (
            <Loader2 className="w-3 h-3 animate-spin text-yellow-500" />
          )}
          <span>{metadata.deliveryStatus}</span>
        </div>
      )}

      {/* Retry count */}
      {metadata.retryCount && metadata.retryCount > 0 && (
        <span>Retries: {metadata.retryCount}</span>
      )}

      {/* Topics */}
      {metadata.topics && metadata.topics.length > 0 && (
        <div className="flex gap-1 flex-wrap">
          <span>Topics:</span>
          {metadata.topics.slice(0, 3).map((topic) => (
            <Badge key={topic} variant="outline" className="text-xs">
              {topic}
            </Badge>
          ))}
          {metadata.topics.length > 3 && (
            <span>+{metadata.topics.length - 3} more</span>
          )}
        </div>
      )}

      {/* Entities */}
      {metadata.entities && metadata.entities.length > 0 && (
        <div className="flex gap-1 flex-wrap">
          <span>Entities:</span>
          {metadata.entities.slice(0, 2).map((entity) => (
            <Badge key={entity.value} variant="outline" className="text-xs">
              {entity.type}: {entity.value}
            </Badge>
          ))}
          {metadata.entities.length > 2 && (
            <span>+{metadata.entities.length - 2} more</span>
          )}
        </div>
      )}

      {/* Knowledge sources */}
      {metadata.knowledgeSources && metadata.knowledgeSources.length > 0 && (
        <div className="flex gap-1 flex-wrap">
          <span>Sources:</span>
          {metadata.knowledgeSources.slice(0, 2).map((source) => (
            <Badge key={source.id} variant="outline" className="text-xs">
              {source.title}
            </Badge>
          ))}
          {metadata.knowledgeSources.length > 2 && (
            <span>+{metadata.knowledgeSources.length - 2} more</span>
          )}
        </div>
      )}

      {/* Conversation turn */}
      {metadata.conversationTurn && (
        <span>Turn: {metadata.conversationTurn}</span>
      )}

      {/* Read by indicators */}
      {metadata.readBy && metadata.readBy.length > 0 && (
        <div className="flex items-center gap-1">
          <Eye className="w-3 h-3" />
          <span>Read by {metadata.readBy.length}</span>
        </div>
      )}
    </div>
  );
});

MessageMetadata.displayName = "MessageMetadata";

// Message Reactions Component
export const MessageReactions = memo<{
  message: Message;
  onReaction?: (messageId: string, type: string) => void;
}>(({ message, onReaction }) => {
  const reactions = message.metadata?.reactions;
  if (!reactions || reactions.length === 0) return null;

  // Group reactions by type
  const reactionGroups = reactions.reduce(
    (acc, reaction) => {
      if (!acc[reaction.type]) {
        acc[reaction.type] = [];
      }
      acc[reaction.type].push(reaction);
      return acc;
    },
    {} as Record<string, typeof reactions>,
  );

  return (
    <div className="flex gap-1 mt-2 flex-wrap">
      {Object.entries(reactionGroups).map(([type, reactionList]) => (
        <Button
          key={type}
          variant="outline"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={() => onReaction?.(message.id, type)}
        >
          {type} {reactionList.length}
        </Button>
      ))}
    </div>
  );
});

MessageReactions.displayName = "MessageReactions";

// Message Actions Component
export const MessageActions = memo<{
  message: Message;
  onReply?: (message: Message) => void;
  onReaction?: (messageId: string, type: string) => void;
  showActions?: boolean;
}>(({ message, onReply, onReaction, showActions = true }) => {
  if (!showActions) return null;

  return (
    <div className="flex gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs"
        onClick={(e) => {
          e.stopPropagation();
          onReply?.(message);
        }}
      >
        <Reply className="w-3 h-3 mr-1" />
        Reply
      </Button>

      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs"
        onClick={(e) => {
          e.stopPropagation();
          onReaction?.(message.id, "👍");
        }}
      >
        <ThumbsUp className="w-3 h-3" />
      </Button>

      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs"
        onClick={(e) => {
          e.stopPropagation();
          onReaction?.(message.id, "👎");
        }}
      >
        <ThumbsDown className="w-3 h-3" />
      </Button>

      <Button variant="ghost" size="sm" className="h-6 px-2 text-xs">
        <MoreHorizontal className="w-3 h-3" />
      </Button>
    </div>
  );
});

MessageActions.displayName = "MessageActions";

// Regular Message Component
export const RegularMessage = memo<MessageComponentProps>(
  ({
    message,
    sender,
    showMetadata = true,
    showActions = true,
    onReply,
    onReaction,
    className,
  }) => {
    return (
      <div
        className={cn(
          "group flex gap-3 hover:bg-muted/20 transition-colors",
          className,
        )}
      >
        {/* Avatar */}
        <Avatar className="w-8 h-8 flex-shrink-0">
          <AvatarImage src={sender?.avatar} />
          <AvatarFallback style={{ backgroundColor: sender?.color || "#666" }}>
            {sender ? (
              sender.name.charAt(0).toUpperCase()
            ) : message.senderId === "user" ? (
              <User className="w-4 h-4" />
            ) : (
              <Bot className="w-4 h-4" />
            )}
          </AvatarFallback>
        </Avatar>

        <div className="flex-1 min-w-0">
          {/* Message header */}
          <MessageHeader
            message={message}
            sender={sender}
            showMetadata={showMetadata}
          />

          {/* Message content */}
          <div className="text-sm mb-2 break-words">{message.content}</div>

          {/* Attachments */}
          {message.metadata?.attachments &&
            message.metadata.attachments.length > 0 && (
              <div className="mb-2">
                {message.metadata.attachments.map((attachment, index) => (
                  <div
                    key={index}
                    className="text-xs text-muted-foreground border rounded p-2 mb-1"
                  >
                    📎 {attachment.type}: {attachment.url}
                  </div>
                ))}
              </div>
            )}

          {/* Metadata */}
          {showMetadata && <MessageMetadata message={message} />}

          {/* Reactions */}
          <MessageReactions message={message} onReaction={onReaction} />

          {/* Actions */}
          <MessageActions
            message={message}
            onReply={onReply}
            onReaction={onReaction}
            showActions={showActions}
          />
        </div>
      </div>
    );
  },
);

RegularMessage.displayName = "RegularMessage";

// Text Message Component (alias for RegularMessage)
export const TextMessage = RegularMessage;

// Code Message Component
export const CodeMessage = memo<MessageComponentProps>(
  ({
    message,
    sender,
    showMetadata = true,
    showActions = true,
    onReply,
    onReaction,
    className,
  }) => {
    const language = message.metadata?.language || 'text';
    
    return (
      <div
        className={cn(
          "group flex gap-3 hover:bg-muted/20 transition-colors",
          className,
        )}
      >
        {/* Avatar */}
        <Avatar className="w-8 h-8 flex-shrink-0">
          <AvatarImage src={sender?.avatar} />
          <AvatarFallback style={{ backgroundColor: sender?.color || "#666" }}>
            {sender ? (
              sender.name.charAt(0).toUpperCase()
            ) : message.senderId === "user" ? (
              <User className="w-4 h-4" />
            ) : (
              <Bot className="w-4 h-4" />
            )}
          </AvatarFallback>
        </Avatar>

        <div className="flex-1 min-w-0">
          {/* Message header */}
          <MessageHeader
            message={message}
            sender={sender}
            showMetadata={showMetadata}
          />

          {/* Code content */}
          <div className="mb-2">
            <div className="text-xs text-muted-foreground mb-1 flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {language}
              </Badge>
            </div>
            <pre role="code" className="bg-muted/50 rounded-lg p-3 text-sm overflow-x-auto">
              <code>{message.content}</code>
            </pre>
          </div>

          {/* Metadata */}
          {showMetadata && <MessageMetadata message={message} />}

          {/* Reactions */}
          <MessageReactions message={message} onReaction={onReaction} />

          {/* Actions */}
          <MessageActions
            message={message}
            onReply={onReply}
            onReaction={onReaction}
            showActions={showActions}
          />
        </div>
      </div>
    );
  },
);

CodeMessage.displayName = "CodeMessage";

// Helper function to get message type icon
function getMessageTypeIcon(type: string) {
  const icons: Record<string, React.ReactNode> = {
    conversation_starter: <Hash className="w-3 h-3 mr-1" />,
    action: <Zap className="w-3 h-3 mr-1" />,
    tool_result: <Brain className="w-3 h-3 mr-1" />,
  };

  return icons[type] || null;
}
