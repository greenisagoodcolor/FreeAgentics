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
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full">
      {/* History snippets - moved to top */}
      {conversationId &&
        iterationContext?.conversation_summary?.prompt_themes &&
        iterationContext.conversation_summary.prompt_themes.length > 0 && (
          <div className="mb-3">
            <ScrollArea className="max-w-full">
              <div className="flex gap-2 text-xs text-muted-foreground">
                {iterationContext.conversation_summary.prompt_themes.map((theme, idx) => (
                  <div key={idx} className="whitespace-nowrap">
                    {theme}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

      {/* Main prompt input - now full width and bigger */}
      <div className="flex gap-3 items-start">
        <div className="flex-1 relative">
          <Textarea
            ref={textareaRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsExpanded(true)}
            onBlur={() => !prompt && setIsExpanded(false)}
            placeholder="Describe your goal or task in detail. You can write multiple paragraphs here to explain what you want the agents to do..."
            disabled={isLoading}
            className={cn(
              "resize-none transition-all duration-200 text-base",
              isExpanded || prompt ? "h-32" : "h-20",
              "min-h-20",
            )}
          />

          {isLoading && (
            <div className="absolute right-3 top-3 text-sm text-muted-foreground">Processing...</div>
          )}
        </div>

        {/* Settings button */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => setShowSettings(true)}
          aria-label="Settings"
          className="mt-1"
        >
          <Settings className="h-4 w-4" />
        </Button>
      </div>

      {/* Submit hint */}
      <div className="mt-2 text-xs text-muted-foreground">
        Press Enter to submit, or Shift+Enter for new line
      </div>

      {/* Error display */}
      {error && (
        <Alert variant="destructive" className="mt-3">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Settings drawer */}
      <SettingsDrawer open={showSettings} onOpenChange={setShowSettings} />
    </div>
  );
}
