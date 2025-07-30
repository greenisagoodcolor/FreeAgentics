"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Loader2, AlertCircle, Trash2, Sparkles } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { useConversation, type MessageRole } from "@/hooks/use-conversation";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  role: MessageRole;
  content: string;
  timestamp: string;
  isStreaming?: boolean;
}

function MessageBubble({ role, content, timestamp, isStreaming }: MessageBubbleProps) {
  const isUser = role === "user";
  const isSystem = role === "system";

  const formatTime = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div
      className={cn("flex gap-3 mb-4", isUser && "flex-row-reverse", isSystem && "justify-center")}
    >
      {!isSystem && (
        <div
          className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted",
          )}
        >
          {isUser ? "U" : "AI"}
        </div>
      )}

      <div
        className={cn(
          "flex flex-col gap-1",
          isUser && "items-end",
          isSystem && "items-center w-full",
        )}
      >
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{isUser ? "You" : isSystem ? "System" : "Assistant"}</span>
          <span>{formatTime(timestamp)}</span>
          {isStreaming && (
            <div data-testid="streaming-indicator" className="flex gap-1">
              <span className="w-1 h-1 bg-primary rounded-full animate-bounce" />
              <span className="w-1 h-1 bg-primary rounded-full animate-bounce [animation-delay:0.2s]" />
              <span className="w-1 h-1 bg-primary rounded-full animate-bounce [animation-delay:0.4s]" />
            </div>
          )}
        </div>

        <div
          className={cn(
            "rounded-lg px-4 py-2 max-w-[80%]",
            isUser
              ? "bg-primary text-primary-foreground"
              : isSystem
                ? "bg-muted text-center text-sm italic"
                : "bg-muted",
          )}
        >
          <div className="prose prose-sm dark:prose-invert">{content}</div>
        </div>
      </div>
    </div>
  );
}

export function ConversationWindow() {
  const { messages, sendMessage, isLoading, error, conversationId, clearConversation, goalPrompt } =
    useConversation();
  const { suggestions } = usePromptProcessor();

  const [input, setInput] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]",
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!input.trim() || isLoading) return;

    sendMessage({ content: input.trim() });
    setInput("");

    // Focus back on textarea
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    textareaRef.current?.focus();
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Conversation</CardTitle>
            <CardDescription>
              {conversationId
                ? `Session: ${conversationId.slice(0, 8)}...`
                : "Chat with your agents"}
            </CardDescription>
          </div>
          {messages.length > 0 && (
            <Button variant="outline" size="sm" onClick={clearConversation} className="gap-2">
              <Trash2 className="h-4 w-4" />
              Clear conversation
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Goal Prompt Banner */}
        {goalPrompt && (
          <div className="bg-primary/10 border border-primary/20 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <Sparkles className="h-4 w-4 text-primary mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-primary mb-1">Active Goal</p>
                <p className="text-sm text-primary/80">{goalPrompt}</p>
              </div>
            </div>
          </div>
        )}

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-center">
              <div className="text-muted-foreground">
                <p className="text-sm font-medium">Start a conversation</p>
                <p className="text-xs mt-1">Type a message below to begin</p>
              </div>
            </div>
          ) : (
            <ScrollArea
              ref={scrollAreaRef}
              data-testid="messages-scroll-area"
              className="h-full pr-4"
            >
              <div className="space-y-2">
                {messages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    role={message.role}
                    content={message.content}
                    timestamp={message.timestamp}
                    isStreaming={message.isStreaming}
                  />
                ))}
              </div>
            </ScrollArea>
          )}
        </div>

        {/* Suggestions */}
        {suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <Sparkles className="h-3 w-3" />
              Suggestions:
            </span>
            {suggestions.map((suggestion, idx) => (
              <Badge
                key={idx}
                variant="secondary"
                className="cursor-pointer hover:bg-secondary/80"
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion}
              </Badge>
            ))}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error.message}</AlertDescription>
          </Alert>
        )}

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Textarea
            ref={textareaRef}
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            className="flex-1 min-h-[60px] max-h-[120px] resize-none"
          />
          <Button
            type="submit"
            disabled={!input.trim() || isLoading}
            size="icon"
            className="h-[60px] w-[60px]"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            <span className="sr-only">Send message</span>
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
