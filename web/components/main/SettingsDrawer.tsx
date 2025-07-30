"use client";

import React from "react";
import { Settings2, LogOut, LogIn, RotateCcw } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/simple-select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/hooks/use-auth";
import { useSettings, LLM_MODELS, type LLMProvider } from "@/hooks/use-settings";

interface SettingsDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// Removed unused PROVIDER_DISPLAY_NAMES constant

export function SettingsDrawer({ open, onOpenChange }: SettingsDrawerProps) {
  const { user, isAuthenticated, logout } = useAuth();
  const { settings, updateSettings, resetSettings } = useSettings();

  const handleProviderChange = (provider: string) => {
    updateSettings({ llmProvider: provider as LLMProvider });
  };

  const handleModelChange = (model: string) => {
    updateSettings({ llmModel: model });
  };

  const availableModels = LLM_MODELS[settings.llmProvider] || [];

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[400px] sm:w-[540px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Settings2 className="h-5 w-5" />
            Settings
          </SheetTitle>
          <SheetDescription>Configure your FreeAgentics experience</SheetDescription>
        </SheetHeader>

        <div className="mt-8 space-y-8 pb-6">
          {/* LLM Configuration */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Language Model</h3>

            <div className="space-y-3">
              <div className="space-y-2">
                <Label htmlFor="llm-provider">LLM Provider</Label>
                <Select value={settings.llmProvider} onValueChange={handleProviderChange}>
                  <SelectTrigger id="llm-provider">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="ollama">Ollama</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="llm-model">Model</Label>
                <Select value={settings.llmModel} onValueChange={handleModelChange}>
                  <SelectTrigger id="llm-model">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* API Key Configuration */}
              {settings.llmProvider !== "ollama" && (
                <div className="space-y-2">
                  <Label htmlFor="api-key">
                    {settings.llmProvider === "openai" ? "OpenAI API Key" : "Anthropic API Key"}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      id="api-key"
                      type="password"
                      placeholder={`Enter your ${settings.llmProvider === "openai" ? "OpenAI" : "Anthropic"} API key`}
                      value={settings.llmProvider === "openai" ? settings.openaiApiKey : settings.anthropicApiKey}
                      onChange={(e) => {
                        const key = settings.llmProvider === "openai" ? "openaiApiKey" : "anthropicApiKey";
                        updateSettings({ [key]: e.target.value });
                      }}
                      className="flex-1"
                    />
                    <Button
                      type="button"
                      size="sm"
                      onClick={() => {
                        // API key is already saved on change
                        const key = settings.llmProvider === "openai" ? settings.openaiApiKey : settings.anthropicApiKey;
                        if (key) {
                          // Visual feedback that key is saved
                          const input = document.getElementById("api-key") as HTMLInputElement;
                          if (input) {
                            const originalPlaceholder = input.placeholder;
                            input.placeholder = "API key saved!";
                            setTimeout(() => {
                              input.placeholder = originalPlaceholder;
                            }, 2000);
                          }
                        }
                      }}
                    >
                      Save
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Your API key is stored locally and never sent to our servers
                  </p>
                </div>
              )}
            </div>
          </div>

          <Separator />

          {/* Authentication */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Authentication</h3>

            <div className="flex items-center justify-between rounded-lg border p-4">
              <div className="space-y-0.5">
                <div className="text-sm font-medium">
                  {isAuthenticated ? user?.email : "Not logged in"}
                </div>
                {isAuthenticated && user?.name && (
                  <div className="text-xs text-muted-foreground">{user.name}</div>
                )}
              </div>

              {isAuthenticated ? (
                <Button variant="outline" size="sm" onClick={() => logout()} className="gap-2">
                  <LogOut className="h-4 w-4" />
                  Log out
                </Button>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onOpenChange(false)}
                  className="gap-2"
                >
                  <LogIn className="h-4 w-4" />
                  Log in
                </Button>
              )}
            </div>
          </div>

          <Separator />

          {/* Feature Flags */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Features</h3>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="gnn-enabled">GNN Enabled</Label>
                  <p className="text-xs text-muted-foreground">
                    Enable Graph Neural Network features
                  </p>
                </div>
                <Switch
                  id="gnn-enabled"
                  checked={settings.gnnEnabled}
                  onCheckedChange={(checked) => updateSettings({ gnnEnabled: checked })}
                  aria-label="GNN Enabled"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="debug-logs">Debug Logs</Label>
                  <p className="text-xs text-muted-foreground">Show detailed debug information</p>
                </div>
                <Switch
                  id="debug-logs"
                  checked={settings.debugLogs}
                  onCheckedChange={(checked) => updateSettings({ debugLogs: checked })}
                  aria-label="Debug Logs"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="auto-suggest">Auto-Suggest</Label>
                  <p className="text-xs text-muted-foreground">
                    Show automatic suggestions during prompts
                  </p>
                </div>
                <Switch
                  id="auto-suggest"
                  checked={settings.autoSuggest}
                  onCheckedChange={(checked) => updateSettings({ autoSuggest: checked })}
                  aria-label="Auto-Suggest"
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Reset Settings */}
          <div className="pt-4">
            <Button
              variant="outline"
              onClick={() => {
                // Use window.confirm with fallback for tests
                const shouldReset =
                  typeof window !== "undefined" && window.confirm
                    ? window.confirm("Are you sure you want to reset all settings to defaults?")
                    : true; // In tests, always proceed

                if (shouldReset) {
                  resetSettings();
                }
              }}
              className="w-full gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Reset to Defaults
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
