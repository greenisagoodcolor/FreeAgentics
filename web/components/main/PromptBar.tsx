"use client";

import React, { useState, useRef, KeyboardEvent } from "react";
import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { usePromptProcessor } from "@/hooks/use-prompt-processor";
import { SettingsDrawer } from "./SettingsDrawer";
import { cn } from "@/lib/utils";

export function PromptBar() {
  const [prompt, setPrompt] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { submitPrompt, isLoading, error, iterationContext, conversationId } =
    usePromptProcessor();

  const handleSubmit = async () => {
    if (!prompt.trim() || isLoading) return;

    await submitPrompt(prompt.trim());
    setPrompt("");
    setIsExpanded(false);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.key === "Enter" && !e.shiftKey) || (e.key === "Enter" && e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex items-center gap-2 p-2 h-full">
      {/* History snippets */}
      {conversationId &&
        iterationContext?.conversation_summary?.prompt_themes &&
        iterationContext.conversation_summary.prompt_themes.length > 0 && (
          <ScrollArea className="flex-1 max-w-xs">
            <div className="flex gap-2 text-xs text-muted-foreground">
              {iterationContext.conversation_summary.prompt_themes.map((theme, idx) => (
                <div key={idx} className="whitespace-nowrap">
                  {theme}
                </div>
              ))}
            </div>
          </ScrollArea>
        )}

      {/* Main prompt input */}
      <div className="flex-1 relative">
        <Textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsExpanded(true)}
          onBlur={() => !prompt && setIsExpanded(false)}
          placeholder="Enter your goal or prompt..."
          disabled={isLoading}
          className={cn(
            "resize-none transition-all duration-200 pr-20",
            isExpanded ? "h-20" : "h-10",
            "min-h-10",
          )}
        />

        {isLoading && (
          <div className="absolute right-2 top-2 text-sm text-muted-foreground">Processing...</div>
        )}
      </div>

      {/* Settings button */}
      <Button
        variant="outline"
        size="icon"
        onClick={() => setShowSettings(true)}
        aria-label="Settings"
      >
        <Settings className="h-4 w-4" />
      </Button>

      {/* Error display */}
      {error && (
        <Alert variant="destructive" className="absolute top-full left-0 right-0 mt-2">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Settings drawer */}
      <SettingsDrawer open={showSettings} onOpenChange={setShowSettings} />
    </div>
  );
}
