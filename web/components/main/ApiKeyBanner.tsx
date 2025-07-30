"use client";

import React, { useEffect, useState } from "react";
import { AlertCircle, Settings, X } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useSettings } from "@/hooks/use-settings";

interface ApiKeyBannerProps {
  onSettingsClick: () => void;
}

export function ApiKeyBanner({ onSettingsClick }: ApiKeyBannerProps) {
  const { settings, isLoading } = useSettings();
  const [isDismissed, setIsDismissed] = useState(false);

  // Check if any API key is configured
  const hasApiKey = settings.openaiApiKey || settings.anthropicApiKey;

  // Reset dismissed state when API key status changes
  useEffect(() => {
    setIsDismissed(false);
  }, [hasApiKey]);

  if (isLoading || hasApiKey || isDismissed) {
    return null;
  }

  return (
    <Alert className="mb-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20">
      <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-500" />
      <AlertTitle className="text-yellow-800 dark:text-yellow-400">
        API Key Required for Agent Conversations
      </AlertTitle>
      <AlertDescription className="mt-2 text-yellow-700 dark:text-yellow-500">
        <div className="space-y-2">
          <p>
            To enable AI agents to converse with each other, you need to add your OpenAI API key.
            Without it, agents cannot communicate or collaborate.
          </p>
          <div className="flex gap-2 mt-3">
            <Button
              size="sm"
              onClick={onSettingsClick}
              className="gap-2"
            >
              <Settings className="h-4 w-4" />
              Add API Key in Settings
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setIsDismissed(true)}
              className="gap-2"
            >
              <X className="h-4 w-4" />
              Dismiss
            </Button>
          </div>
        </div>
      </AlertDescription>
    </Alert>
  );
}