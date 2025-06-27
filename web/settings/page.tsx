"use client";

import type React from "react";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { useLLM } from "@/contexts/llm-context";
import { useToast } from "@/hooks/use-toast";
import LLMTest from "@/components/llmtest";
import { validateStoredSession } from "@/lib/session-management";
import { Shield, ShieldAlert, ShieldCheck } from "lucide-react";
import type { LLMSettings } from "@/lib/llm-settings";
import { createLogger } from "@/lib/debug-logger";

const logger = createLogger("SETTINGS");

export default function SettingsPage() {
  const {
    settings,
    updateSettings,
    saveSettings,
    isProcessing,
    setIsProcessing,
    client,
  } = useLLM();
  const { toast } = useToast();

  // Initialize with hardcoded default values to prevent errors
  const [provider, setProvider] = useState<string>("openai");
  const [model, setModel] = useState<string>("gpt-4o");
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(1024);
  const [topP, setTopP] = useState<number>(0.9);
  const [frequencyPenalty, setFrequencyPenalty] = useState<number>(0);
  const [presencePenalty, setPresencePenalty] = useState<number>(0);
  const [maxAutonomousMessages, setMaxAutonomousMessages] = useState<number>(4);
  const [conversationCooldown, setConversationCooldown] =
    useState<number>(5000);
  const [systemFingerprint, setSystemFingerprint] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>("");
  const [hasStoredApiKey, setHasStoredApiKey] = useState<boolean>(false);
  const [isValidatingApiKey, setIsValidatingApiKey] = useState<boolean>(false);

  // Update local state when settings change
  useEffect(() => {
    if (settings) {
      setProvider(settings.provider || "openai");
      setModel(settings.model || "gpt-4o");

      // Ensure numeric parameters have proper default values if undefined
      setTemperature(
        typeof settings.temperature === "number" ? settings.temperature : 0.7,
      );
      setMaxTokens(
        typeof settings.maxTokens === "number" ? settings.maxTokens : 1024,
      );
      setTopP(typeof settings.topP === "number" ? settings.topP : 0.9);
      setFrequencyPenalty(
        typeof settings.frequencyPenalty === "number"
          ? settings.frequencyPenalty
          : 0,
      );
      setPresencePenalty(
        typeof settings.presencePenalty === "number"
          ? settings.presencePenalty
          : 0,
      );
      setMaxAutonomousMessages(
        typeof settings.maxAutonomousMessages === "number"
          ? settings.maxAutonomousMessages
          : 4,
      );
      setConversationCooldown(
        typeof settings.conversationCooldown === "number"
          ? settings.conversationCooldown
          : 5000,
      );

      setSystemFingerprint(
        typeof settings.systemFingerprint === "boolean"
          ? settings.systemFingerprint
          : false,
      );

      // Check if we have a stored API key
      const checkApiKeyValidity = async () => {
        if (settings.apiKeySessionId) {
          setIsValidatingApiKey(true);
          try {
            const isValid = await validateStoredSession(
              settings.provider,
              settings.apiKeySessionId,
            );
            setHasStoredApiKey(isValid);
          } catch (error) {
            console.error("Error validating API key session:", error);
            setHasStoredApiKey(false);
          } finally {
            setIsValidatingApiKey(false);
          }
        } else {
          setHasStoredApiKey(false);
        }
      };

      checkApiKeyValidity();

      // Clear the API key input field - we don't display the actual API key for security
      setApiKey("");

      logger.info("Settings page updated with settings:", {
        provider: settings.provider,
        model: settings.model,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        topP: settings.topP,
        frequencyPenalty: settings.frequencyPenalty,
        presencePenalty: settings.presencePenalty,
        systemFingerprint: settings.systemFingerprint,
        maxAutonomousMessages: settings.maxAutonomousMessages,
        conversationCooldown: settings.conversationCooldown,
        hasApiKeySessionId: !!settings.apiKeySessionId,
        apiKeySessionId: settings.apiKeySessionId,
      });
    }
  }, [settings]);

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      setIsProcessing(true);

      // Clean the API key (trim whitespace)
      const cleanApiKey = apiKey.trim();
      console.log("API key length after trimming:", cleanApiKey.length);

      // Create a new settings object with all the form values (except API key)
      const newSettings: LLMSettings = {
        provider: provider as "openai" | "openrouter",
        model,
        temperature,
        maxTokens,
        topP,
        frequencyPenalty,
        presencePenalty,
        systemFingerprint,
        maxAutonomousMessages,
        conversationCooldown,
      };

      console.log("Submitting settings:", {
        ...newSettings,
        provider: newSettings.provider,
      });

      // If a new API key was provided, store it securely
      let apiKeySuccess = true;
      let sessionId = settings?.apiKeySessionId;

      if (cleanApiKey) {
        logger.info("Setting new API key");
        try {
          // Store the API key securely
          const response = await fetch("/api/api-key/store", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              provider: newSettings.provider,
              apiKey: cleanApiKey,
            }),
          });

          if (!response.ok) {
            logger.error(`Error storing API key: HTTP ${response.status}`);
            apiKeySuccess = false;
          } else {
            const data = await response.json();

            if (!data.success) {
              logger.error("Failed to store API key:", data.message);
              apiKeySuccess = false;
            } else {
              // Get the session ID
              sessionId = data.sessionId;
              logger.info("Received session ID:", sessionId);

              // Store the session ID in localStorage only if it exists
              if (sessionId) {
                localStorage.setItem(
                  `api_session_${newSettings.provider}`,
                  sessionId,
                );
                logger.info(
                  `Stored session ID in localStorage with key: api_session_${newSettings.provider}`,
                );
              }

              setHasStoredApiKey(true);
              setApiKey(""); // Clear the input field after successful storage
            }
          }
        } catch (error) {
          logger.error("Error setting API key:", error);
          apiKeySuccess = false;
        }
      }

      if (!apiKeySuccess) {
        toast({
          title: "Error saving API key",
          description: "Failed to securely store the API key.",
          variant: "destructive",
          duration: 5000,
        });
      }

      // Add the session ID to the settings if we have one
      if (sessionId) {
        newSettings.apiKeySessionId = sessionId;
        logger.info("Adding session ID to settings:", sessionId);
      }

      // Update settings in context
      updateSettings(newSettings);

      // Force update the client settings
      if (client) {
        client.updateSettings(newSettings);
      }

      // Save settings
      const success = await saveSettings();

      if (success && apiKeySuccess) {
        toast({
          title: "Settings saved",
          description: "Your LLM settings have been updated successfully.",
          duration: 3000,
        });
      } else if (!success) {
        toast({
          title: "Error saving settings",
          description: "Failed to save LLM settings.",
          variant: "destructive",
          duration: 5000,
        });
      }
    } catch (error) {
      console.error("Error saving settings:", error);
      toast({
        title: "Error saving settings",
        description:
          error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Available models by provider
  const modelsByProvider: Record<
    string,
    Array<{ id: string; name: string }>
  > = {
    openai: [
      { id: "gpt-4o", name: "GPT-4o" },
      { id: "gpt-4-turbo", name: "GPT-4 Turbo" },
      { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo" },
    ],
    openrouter: [
      { id: "openai/gpt-4o", name: "OpenAI GPT-4o" },
      { id: "openai/gpt-4-turbo", name: "OpenAI GPT-4 Turbo" },
      { id: "openai/gpt-3.5-turbo", name: "OpenAI GPT-3.5 Turbo" },
      { id: "anthropic/claude-3-opus", name: "Anthropic Claude 3 Opus" },
      { id: "anthropic/claude-3-sonnet", name: "Anthropic Claude 3 Sonnet" },
      { id: "meta-llama/llama-3-70b-instruct", name: "Meta Llama 3 70B" },
    ],
  };

  // Format number safely with fallback
  const formatNumber = (
    value: number | undefined,
    decimals: number,
    fallback: number,
  ): string => {
    if (typeof value !== "number") return fallback.toFixed(decimals);
    return value.toFixed(decimals);
  };

  // Convert milliseconds to seconds for display
  const msToSeconds = (ms: number): number => ms / 1000;

  // Convert seconds to milliseconds for storage
  const secondsToMs = (seconds: number): number => seconds * 1000;

  return (
    <div className="min-h-screen p-4 text-white">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">LLM Settings</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <Card className="w-full">
              <CardHeader>
                <CardTitle>Model Configuration</CardTitle>
                <CardDescription>
                  Configure the LLM model and parameters
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="space-y-2">
                    <label htmlFor="provider" className="text-sm font-medium">
                      Provider
                    </label>
                    <Select
                      value={provider}
                      onValueChange={(value) => {
                        logger.info("Provider changed to:", value);
                        setProvider(value);
                        // Reset model when provider changes
                        if (modelsByProvider[value]) {
                          setModel(modelsByProvider[value][0].id);
                        }
                        // Reset API key status when provider changes
                        setHasStoredApiKey(false);
                      }}
                    >
                      <SelectTrigger id="provider">
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="openrouter">OpenRouter</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="model" className="text-sm font-medium">
                      Model
                    </label>
                    <Select value={model} onValueChange={setModel}>
                      <SelectTrigger id="model">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {modelsByProvider[provider]?.map((modelOption) => (
                          <SelectItem
                            key={modelOption.id}
                            value={modelOption.id}
                          >
                            {modelOption.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label
                        htmlFor="api-key"
                        className="text-sm font-medium flex items-center"
                      >
                        API Key
                        {isValidatingApiKey ? (
                          <Spinner size={16} className="ml-2" />
                        ) : hasStoredApiKey ? (
                          <ShieldCheck
                            size={16}
                            className="ml-2 text-green-500"
                          />
                        ) : (
                          <ShieldAlert
                            size={16}
                            className="ml-2 text-amber-500"
                          />
                        )}
                      </label>
                      {hasStoredApiKey && (
                        <span className="text-xs text-green-500 flex items-center">
                          <Shield size={12} className="mr-1" />
                          Securely Stored
                        </span>
                      )}
                    </div>
                    <Input
                      id="api-key"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={
                        hasStoredApiKey
                          ? `${provider === "openai" ? "OpenAI" : "OpenRouter"} API key is securely stored. Enter a new one to update.`
                          : `Enter your ${provider === "openai" ? "OpenAI" : "OpenRouter"} API key...`
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      {provider === "openai"
                        ? "Your OpenAI API key is used to access OpenAI models directly."
                        : "OpenRouter provides access to multiple LLM providers through a single API key."}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Your API key is stored securely and never exposed in the
                      browser.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label
                      htmlFor="temperature"
                      className="text-sm font-medium"
                    >
                      Temperature: {formatNumber(temperature, 1, 0.7)}
                    </label>
                    <div className="flex items-center gap-2">
                      <input
                        id="temperature"
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Controls randomness: Lower values are more deterministic,
                      higher values are more creative.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="max-tokens" className="text-sm font-medium">
                      Max Tokens: {maxTokens || 1024}
                    </label>
                    <Input
                      id="max-tokens"
                      type="number"
                      min="1"
                      max="8192"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                    />
                    <p className="text-xs text-muted-foreground">
                      Maximum number of tokens to generate in the response.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="top-p" className="text-sm font-medium">
                      Top P: {formatNumber(topP, 2, 0.9)}
                    </label>
                    <div className="flex items-center gap-2">
                      <input
                        id="top-p"
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={topP}
                        onChange={(e) => setTopP(Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Controls diversity via nucleus sampling.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label
                      htmlFor="frequency-penalty"
                      className="text-sm font-medium"
                    >
                      Frequency Penalty: {formatNumber(frequencyPenalty, 1, 0)}
                    </label>
                    <div className="flex items-center gap-2">
                      <input
                        id="frequency-penalty"
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={frequencyPenalty}
                        onChange={(e) =>
                          setFrequencyPenalty(Number(e.target.value))
                        }
                        className="w-full"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Reduces repetition of token sequences.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label
                      htmlFor="presence-penalty"
                      className="text-sm font-medium"
                    >
                      Presence Penalty: {formatNumber(presencePenalty, 1, 0)}
                    </label>
                    <div className="flex items-center gap-2">
                      <input
                        id="presence-penalty"
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={presencePenalty}
                        onChange={(e) =>
                          setPresencePenalty(Number(e.target.value))
                        }
                        className="w-full"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Encourages talking about new topics.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label
                      htmlFor="max-autonomous-messages"
                      className="text-sm font-medium"
                    >
                      Max Autonomous Messages: {maxAutonomousMessages}
                    </label>
                    <Input
                      id="max-autonomous-messages"
                      type="number"
                      min="1"
                      max="50"
                      value={maxAutonomousMessages}
                      onChange={(e) =>
                        setMaxAutonomousMessages(Number(e.target.value))
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Maximum number of messages in autonomous conversations
                      before they automatically end.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <label
                      htmlFor="conversation-cooldown"
                      className="text-sm font-medium"
                    >
                      Conversation Cooldown: {msToSeconds(conversationCooldown)}{" "}
                      seconds
                    </label>
                    <Input
                      id="conversation-cooldown"
                      type="number"
                      min="1"
                      max="300"
                      value={msToSeconds(conversationCooldown)}
                      onChange={(e) =>
                        setConversationCooldown(
                          secondsToMs(Number(e.target.value)),
                        )
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Time (in seconds) an agent must wait before starting a new
                      autonomous conversation.
                    </p>
                  </div>

                  {provider === "openai" && (
                    <div className="flex items-center space-x-2">
                      <input
                        id="system-fingerprint"
                        type="checkbox"
                        checked={systemFingerprint}
                        onChange={(e) => setSystemFingerprint(e.target.checked)}
                        className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                      />
                      <label
                        htmlFor="system-fingerprint"
                        className="text-sm font-medium"
                      >
                        Include system fingerprint
                      </label>
                    </div>
                  )}

                  <Button
                    type="submit"
                    className="w-full"
                    disabled={isProcessing}
                  >
                    {isProcessing ? (
                      <>
                        <Spinner size={16} className="mr-2" />
                        Saving...
                      </>
                    ) : (
                      "Save Settings"
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          <div>
            <LLMTest />
          </div>
        </div>
      </div>
    </div>
  );
}
