"use client";

import {
  extractBeliefs as serverExtractBeliefs,
  generateKnowledgeEntries as serverGenerateKnowledgeEntries,
  generateResponse,
  validateApiKey,
  saveLLMSettings,
} from "@/lib/llm-service";
import type { LLMSettings } from "@/lib/llm-settings";
import type { KnowledgeEntry } from "@/lib/types";
import { clientDefaultSettings } from "@/lib/llm-settings";
import { getApiKeyFromSession } from "@/lib/session-management";
import { isFeatureEnabled } from "@/lib/feature-flags";
import { createLogger } from "@/lib/debug-logger";
import { isBrowser } from "./browser-check";

const logger = createLogger("LLM-SECURE-CLIENT");

// Secure client-side wrapper for the LLM service
export class LLMSecureClient {
  private settings: LLMSettings;

  constructor(initialSettings: Partial<LLMSettings> = {}) {
    logger.log("LLMSecureClient constructor called with:", {
      initialSettingsType: typeof initialSettings,
      isObject: initialSettings && typeof initialSettings === "object",
      hasServerRef:
        initialSettings &&
        typeof initialSettings === "object" &&
        "__server_ref" in initialSettings,
      keys:
        initialSettings && typeof initialSettings === "object"
          ? Object.keys(initialSettings)
          : [],
    });

    // Create a fresh settings object with default values from client-side defaults
    this.settings = { ...clientDefaultSettings } as LLMSettings;

    // Try to load settings from localStorage first
    try {
      if (isBrowser) {
        const savedSettings = localStorage.getItem("llm-settings");
        if (savedSettings) {
          const parsedSettings = JSON.parse(savedSettings);
          logger.log("Loaded settings from localStorage:", {
            provider: parsedSettings.provider,
            model: parsedSettings.model,
            hasApiKey: !!parsedSettings.apiKey,
            apiKeyLength: parsedSettings.apiKey
              ? parsedSettings.apiKey.length
              : 0,
            hasApiKeySessionId: !!parsedSettings.apiKeySessionId,
          });

          // Apply saved settings
          if (parsedSettings.provider)
            this.settings.provider = parsedSettings.provider;
          if (parsedSettings.model) this.settings.model = parsedSettings.model;
          if (typeof parsedSettings.temperature === "number")
            this.settings.temperature = parsedSettings.temperature;
          if (typeof parsedSettings.maxTokens === "number")
            this.settings.maxTokens = parsedSettings.maxTokens;
          if (typeof parsedSettings.topP === "number")
            this.settings.topP = parsedSettings.topP;
          if (typeof parsedSettings.frequencyPenalty === "number")
            this.settings.frequencyPenalty = parsedSettings.frequencyPenalty;
          if (typeof parsedSettings.presencePenalty === "number")
            this.settings.presencePenalty = parsedSettings.presencePenalty;
          if (typeof parsedSettings.systemFingerprint === "boolean")
            this.settings.systemFingerprint = parsedSettings.systemFingerprint;

          // Handle API key or session ID
          if (parsedSettings.apiKeySessionId) {
            this.settings.apiKeySessionId = parsedSettings.apiKeySessionId;
          } else if (parsedSettings.apiKey) {
            // If we have an API key but no session ID, we'll need to migrate it
            // This will be handled in the migration utility
            this.settings.apiKey = parsedSettings.apiKey;
          }
        }
      }
    } catch (e) {
      logger.warn("Could not load settings from localStorage:", e);
    }

    // Only copy properties from initialSettings if it's a valid object without server refs
    // and if they weren't already loaded from localStorage
    if (
      initialSettings &&
      typeof initialSettings === "object" &&
      !("__server_ref" in initialSettings)
    ) {
      logger.log("Copying properties from initialSettings to this.settings");
      if (initialSettings.provider)
        this.settings.provider = initialSettings.provider;
      if (initialSettings.model) this.settings.model = initialSettings.model;
      if (typeof initialSettings.temperature === "number")
        this.settings.temperature = initialSettings.temperature;
      if (typeof initialSettings.maxTokens === "number")
        this.settings.maxTokens = initialSettings.maxTokens;
      if (typeof initialSettings.topP === "number")
        this.settings.topP = initialSettings.topP;
      if (typeof initialSettings.frequencyPenalty === "number")
        this.settings.frequencyPenalty = initialSettings.frequencyPenalty;
      if (typeof initialSettings.presencePenalty === "number")
        this.settings.presencePenalty = initialSettings.presencePenalty;
      if (typeof initialSettings.systemFingerprint === "boolean")
        this.settings.systemFingerprint = initialSettings.systemFingerprint;

      // Handle API key or session ID
      if (initialSettings.apiKeySessionId) {
        this.settings.apiKeySessionId = initialSettings.apiKeySessionId;
      } else if (initialSettings.apiKey) {
        this.settings.apiKey = initialSettings.apiKey;
      }
    } else {
      logger.log(
        "Not copying properties from initialSettings due to server ref or invalid object",
      );
    }

    logger.log("LLMSecureClient initialized with settings:", {
      ...this.settings,
      apiKey: this.settings.apiKey
        ? `[Length: ${this.settings.apiKey.length}]`
        : undefined,
      apiKeySessionId: this.settings.apiKeySessionId ? "[PRESENT]" : undefined,
      provider: this.settings.provider,
    });

    // Check if we need to migrate an API key to the secure storage
    this.migrateApiKeyIfNeeded();
  }

  // Migrate API key to secure storage if needed
  private async migrateApiKeyIfNeeded(): Promise<void> {
    // Only migrate if secure storage is enabled
    if (!isFeatureEnabled("useSecureApiStorage")) {
      return;
    }

    // Check if we have an API key but no session ID
    if (this.settings.apiKey && !this.settings.apiKeySessionId) {
      logger.log("Migrating API key to secure storage");
      try {
        // Store the API key securely
        const response = await fetch("/api/api-key/store", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            provider: this.settings.provider,
            apiKey: this.settings.apiKey,
          }),
        });

        const data = await response.json();
        if (data.success && data.sessionId) {
          logger.log("API key migrated successfully");
          // Store the session ID
          this.settings.apiKeySessionId = data.sessionId;
          // Remove the API key from settings
          delete this.settings.apiKey;
          // Save the updated settings
          this.saveSettingsToLocalStorage();
        } else {
          logger.error("Failed to migrate API key:", data.message);
        }
      } catch (error) {
        logger.error("Error migrating API key:", error);
      }
    }
  }

  // Save settings to localStorage
  private saveSettingsToLocalStorage(): void {
    try {
      localStorage.setItem("llm-settings", JSON.stringify(this.settings));
      logger.log("Settings saved to localStorage");
    } catch (e) {
      logger.warn("Could not save settings to localStorage:", e);
    }
  }

  // Update settings
  updateSettings(newSettings: Partial<LLMSettings>): void {
    logger.log("LLMSecureClient.updateSettings called with:", {
      newSettingsType: typeof newSettings,
      isObject: newSettings && typeof newSettings === "object",
      hasServerRef:
        newSettings &&
        typeof newSettings === "object" &&
        "__server_ref" in newSettings,
      keys:
        newSettings && typeof newSettings === "object"
          ? Object.keys(newSettings)
          : [],
      apiKeyPresent:
        newSettings && typeof newSettings === "object"
          ? "apiKey" in newSettings
          : false,
      apiKeyValue:
        newSettings &&
        typeof newSettings === "object" &&
        "apiKey" in newSettings
          ? typeof newSettings.apiKey === "string"
            ? `[Length: ${newSettings.apiKey.length}]`
            : String(newSettings.apiKey)
          : "undefined",
      apiKeySessionIdPresent:
        newSettings && typeof newSettings === "object"
          ? "apiKeySessionId" in newSettings
          : false,
    });

    // Handle server references or undefined values
    if (
      !newSettings ||
      typeof newSettings !== "object" ||
      "__server_ref" in newSettings
    ) {
      logger.warn(
        "Invalid settings update or server reference detected, ignoring",
      );
      return;
    }

    // CRITICAL FIX: Ensure provider is correctly updated
    // Log the provider change explicitly
    if (newSettings.provider) {
      logger.log(
        `Updating provider from ${this.settings.provider} to ${newSettings.provider}`,
      );
      this.settings.provider = newSettings.provider;
    }

    // Update only the properties that are provided
    logger.log("Updating settings properties");
    if (newSettings.model) this.settings.model = newSettings.model;
    if (typeof newSettings.temperature === "number")
      this.settings.temperature = newSettings.temperature;
    if (typeof newSettings.maxTokens === "number")
      this.settings.maxTokens = newSettings.maxTokens;
    if (typeof newSettings.topP === "number")
      this.settings.topP = newSettings.topP;
    if (typeof newSettings.frequencyPenalty === "number")
      this.settings.frequencyPenalty = newSettings.frequencyPenalty;
    if (typeof newSettings.presencePenalty === "number")
      this.settings.presencePenalty = newSettings.presencePenalty;
    if (typeof newSettings.systemFingerprint === "boolean")
      this.settings.systemFingerprint = newSettings.systemFingerprint;

    // Handle API key updates
    if (isFeatureEnabled("useSecureApiStorage")) {
      // Secure storage is enabled, handle API key securely
      if (
        "apiKey" in newSettings &&
        typeof newSettings.apiKey === "string" &&
        newSettings.apiKey.trim() !== ""
      ) {
        // Store the API key securely
        this.storeApiKeySecurely(newSettings.apiKey);
      } else if (newSettings.apiKeySessionId) {
        // Use the provided session ID
        this.settings.apiKeySessionId = newSettings.apiKeySessionId;
        // Remove any existing API key
        delete this.settings.apiKey;
      } else if (
        newSettings.apiKey === undefined ||
        newSettings.apiKey === null
      ) {
        // Clear both API key and session ID
        delete this.settings.apiKey;
        delete this.settings.apiKeySessionId;
      }
    } else {
      // Secure storage is disabled, handle API key directly
      if ("apiKey" in newSettings) {
        if (
          typeof newSettings.apiKey === "string" &&
          newSettings.apiKey.trim() !== ""
        ) {
          this.settings.apiKey = newSettings.apiKey;
        } else if (
          newSettings.apiKey === undefined ||
          newSettings.apiKey === null
        ) {
          delete this.settings.apiKey;
        }
      }
    }

    // Save settings to localStorage
    this.saveSettingsToLocalStorage();

    logger.log("Settings updated to:", {
      ...this.settings,
      apiKey: this.settings.apiKey
        ? `[Length: ${this.settings.apiKey.length}]`
        : undefined,
      apiKeySessionId: this.settings.apiKeySessionId ? "[PRESENT]" : undefined,
      provider: this.settings.provider,
    });
  }

  // Store API key securely
  private async storeApiKeySecurely(apiKey: string): Promise<void> {
    try {
      logger.log("Storing API key securely");
      const response = await fetch("/api/api-key/store", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          provider: this.settings.provider,
          apiKey: apiKey,
        }),
      });

      const data = await response.json();
      if (data.success && data.sessionId) {
        logger.log("API key stored securely");
        // Store the session ID
        this.settings.apiKeySessionId = data.sessionId;
        // Remove the API key from settings
        delete this.settings.apiKey;
      } else {
        logger.error("Failed to store API key securely:", data.message);
        // Fall back to storing the API key directly
        this.settings.apiKey = apiKey;
      }
    } catch (error) {
      logger.error("Error storing API key securely:", error);
      // Fall back to storing the API key directly
      this.settings.apiKey = apiKey;
    }
  }

  // Get current settings
  getSettings(): LLMSettings {
    logger.log("LLMSecureClient.getSettings called");
    try {
      // Return a copy to avoid reference issues
      const settingsCopy = { ...this.settings };
      logger.log("LLMSecureClient.getSettings returning:", {
        ...settingsCopy,
        apiKey: settingsCopy.apiKey
          ? `[Length: ${settingsCopy.apiKey.length}]`
          : undefined,
        apiKeySessionId: settingsCopy.apiKeySessionId ? "[PRESENT]" : undefined,
        provider: settingsCopy.provider,
      });
      return settingsCopy;
    } catch (error) {
      logger.error("Error in LLMSecureClient.getSettings:", error);
      // Return a safe default if there's an error
      return {
        provider: "openai",
        model: "gpt-4o",
        temperature: 0.7,
        maxTokens: 1024,
        topP: 0.9,
        frequencyPenalty: 0,
        presencePenalty: 0,
        systemFingerprint: false,
        maxAutonomousMessages: 4,
        conversationCooldown: 5000,
      };
    }
  }

  // Get API key (either from settings or from secure storage)
  private async getApiKey(): Promise<string | null> {
    if (
      isFeatureEnabled("useSecureApiStorage") &&
      this.settings.apiKeySessionId
    ) {
      // Get API key from secure storage
      return await getApiKeyFromSession(this.settings.provider);
    } else if (this.settings.apiKey) {
      // Get API key from settings
      return this.settings.apiKey;
    }
    return null;
  }

  // Generate a response using a system prompt
  async generateResponse(
    systemPrompt: string,
    userPrompt: string,
  ): Promise<string> {
    try {
      logger.log("[LLM SECURE CLIENT] generateResponse called with:", {
        systemPromptLength: systemPrompt?.length,
        userPromptLength: userPrompt?.length,
        provider: this.settings.provider,
        model: this.settings.model,
        apiKeyAvailable: !!(
          this.settings.apiKey || this.settings.apiKeySessionId
        ),
        apiKeyLength: this.settings.apiKey ? this.settings.apiKey.length : 0,
        hasApiKeySessionId: !!this.settings.apiKeySessionId,
      });

      // Create a copy of settings to ensure we're not passing a reference
      const settingsCopy = { ...this.settings };

      // Get the API key
      const apiKey = await this.getApiKey();
      if (!apiKey) {
        logger.warn("[LLM SECURE CLIENT] No API key available");
        return "Error: No API key available. Please set an API key in the settings.";
      }

      // Add the API key to the settings copy
      settingsCopy.apiKey = apiKey;

      // Call the server-side function with the copy
      const response = await generateResponse(
        systemPrompt,
        userPrompt,
        settingsCopy,
      );
      logger.log("[LLM SECURE CLIENT] Response received from server:", {
        responseLength: response?.length,
      });
      return response;
    } catch (error) {
      logger.error("[LLM SECURE CLIENT] Error in generateResponse:", error);
      return `Error: ${error instanceof Error ? error.message : "Unknown error"}`;
    }
  }

  // Extract beliefs from conversation
  async extractBeliefs(
    conversationText: string,
    agentName: string,
    extractionPriorities: string,
  ): Promise<string> {
    try {
      logger.log("LLMSecureClient.extractBeliefs called");

      // Create a copy of settings to ensure we're not passing a reference
      const settingsCopy = { ...this.settings };

      // Get the API key
      const apiKey = await this.getApiKey();
      if (!apiKey) {
        throw new Error(
          `API key is required for ${settingsCopy.provider} provider during belief extraction`,
        );
      }

      // Add the API key to the settings copy
      settingsCopy.apiKey = apiKey;

      return await serverExtractBeliefs(
        conversationText,
        agentName,
        extractionPriorities,
        settingsCopy,
      );
    } catch (error) {
      logger.error("Error in client extractBeliefs:", error);
      return `Error: ${error instanceof Error ? error.message : "Unknown error"}`;
    }
  }

  // Generate knowledge entries from beliefs
  async generateKnowledgeEntries(beliefs: string): Promise<KnowledgeEntry[]> {
    try {
      logger.log("LLMSecureClient.generateKnowledgeEntries called");

      // Create a copy of settings to ensure we're not passing a reference
      const settingsCopy = { ...this.settings };

      // Get the API key
      const apiKey = await this.getApiKey();
      if (!apiKey) {
        throw new Error(
          `API key is required for ${settingsCopy.provider} provider during knowledge generation`,
        );
      }

      // Add the API key to the settings copy
      settingsCopy.apiKey = apiKey;

      return await serverGenerateKnowledgeEntries(beliefs, settingsCopy);
    } catch (error) {
      logger.error("Error in client generateKnowledgeEntries:", error);
      return [
        {
          id: `error-${Date.now()}`,
          title: "Error",
          content: error instanceof Error ? error.message : "Unknown error",
          timestamp: new Date(),
          tags: ["error"],
        },
      ];
    }
  }

  // Stream response
  async streamResponse(
    systemPrompt: string,
    userPrompt: string,
    onChunk?: ((text: string, isComplete: boolean) => void) | null | undefined,
  ): Promise<string> {
    logger.log("[LLM SECURE CLIENT] streamResponse called with:", {
      systemPromptLength: systemPrompt?.length,
      userPromptLength: userPrompt?.length,
      hasOnChunkCallback: typeof onChunk === "function",
      onChunkType: typeof onChunk,
    });

    try {
      // Create a truly safe callback that won't throw if onChunk is not a function
      const safeCallback = (text: string, isComplete: boolean): void => {
        try {
          logger.log("[LLM SECURE CLIENT] safeCallback called with:", {
            textLength: text?.length,
            isComplete,
          });
          if (typeof onChunk === "function") {
            logger.log("[LLM SECURE CLIENT] Calling onChunk function");
            onChunk(text, isComplete);
          } else {
            logger.log(
              "[LLM SECURE CLIENT] Warning: onChunk is not a function",
              {
                onChunkType: typeof onChunk,
                text: text?.substring(0, 20) + "...",
                isComplete,
              },
            );
          }
        } catch (callbackError) {
          logger.error(
            "[LLM SECURE CLIENT] Error executing onChunk callback:",
            callbackError,
          );
        }
      };

      // Use non-streaming as fallback if streaming fails
      let fullResponse = "";
      let streamingFailed = false;

      try {
        // First attempt with streaming
        logger.log("[LLM SECURE CLIENT] Attempting to use streaming response");

        // Create a copy of settings to ensure we're not passing a reference
        const settingsCopy = { ...this.settings };

        // Get the API key
        const apiKey = await this.getApiKey();
        if (!apiKey) {
          logger.warn("[LLM SECURE CLIENT] No API key available");
          safeCallback(
            "Error: No API key available. Please set an API key in the settings.",
            false,
          );
          safeCallback("", true);
          return "Error: No API key available. Please set an API key in the settings.";
        }

        // Add the API key to the settings copy
        settingsCopy.apiKey = apiKey;

        // Call the server-side function
        logger.log("[LLM SECURE CLIENT] Calling generateResponse");
        const response = await generateResponse(
          systemPrompt,
          userPrompt,
          settingsCopy,
        );
        logger.log(
          "[LLM SECURE CLIENT] Response received from generateResponse:",
          {
            responseLength: response?.length,
          },
        );

        // Since we can't actually stream from the server to client with callbacks,
        // we'll simulate streaming by chunking the response
        const chunkSize = 10; // Characters per chunk
        for (let i = 0; i < response.length; i += chunkSize) {
          const chunk = response.substring(i, i + chunkSize);
          fullResponse += chunk;

          // Use the safe callback - NEVER directly call onChunk
          logger.log(
            `[LLM SECURE CLIENT] Processing chunk ${i / chunkSize + 1}/${Math.ceil(response.length / chunkSize)}`,
          );
          safeCallback(chunk, false);

          // Add a small delay to simulate streaming
          await new Promise((resolve) => setTimeout(resolve, 10));
        }

        // Signal completion
        logger.log("[LLM SECURE CLIENT] Signaling completion");
        safeCallback("", true);
      } catch (streamError) {
        logger.error(
          "[LLM SECURE CLIENT] Error in streaming response:",
          streamError,
        );
        streamingFailed = true;
      }

      // If streaming failed, fall back to non-streaming
      if (streamingFailed) {
        logger.log(
          "[LLM SECURE CLIENT] Streaming failed, falling back to non-streaming",
        );
        fullResponse = await this.generateResponse(systemPrompt, userPrompt);

        // Deliver the full response at once - using safe callback
        logger.log("[LLM SECURE CLIENT] Delivering full response at once");
        safeCallback(fullResponse, false);
        safeCallback("", true);
      }

      return fullResponse;
    } catch (error) {
      logger.error("[LLM SECURE CLIENT] Error in streamResponse:", error);

      // Try to notify through callback if possible - using safe callback
      const errorMessage = `Error: ${error instanceof Error ? error.message : String(error)}`;
      try {
        logger.log(
          "[LLM SECURE CLIENT] Attempting to notify error through callback",
        );
        if (typeof onChunk === "function") {
          logger.log("[LLM SECURE CLIENT] Calling onChunk with error message");
          onChunk(errorMessage, false);
          onChunk("", true);
        } else {
          logger.log(
            "[LLM SECURE CLIENT] Cannot notify error: onChunk is not a function",
          );
        }
      } catch (callbackError) {
        logger.error(
          "[LLM SECURE CLIENT] Error calling onChunk callback with error:",
          callbackError,
        );
      }

      // Return error message as string
      return errorMessage;
    }
  }

  // Validate API key
  async validateApiKey(
    provider: "openai" | "openrouter",
    apiKey: string,
  ): Promise<{ valid: boolean; message?: string }> {
    try {
      logger.log("LLMSecureClient.validateApiKey called");
      return await validateApiKey(provider, apiKey);
    } catch (error) {
      logger.error("Error in client validateApiKey:", error);
      return {
        valid: false,
        message:
          error instanceof Error ? error.message : "Error validating API key",
      };
    }
  }

  // Save settings
  async saveSettings(): Promise<boolean> {
    try {
      logger.log("LLMSecureClient.saveSettings called");

      // Create a copy of settings without the API key
      const settingsCopy = { ...this.settings };

      // If we're using secure storage, we don't need to send the API key
      if (isFeatureEnabled("useSecureApiStorage")) {
        delete settingsCopy.apiKey;
      }

      logger.log("Current settings to save:", {
        ...settingsCopy,
        apiKey: settingsCopy.apiKey
          ? `[Length: ${settingsCopy.apiKey.length}]`
          : undefined,
        apiKeySessionId: settingsCopy.apiKeySessionId ? "[PRESENT]" : undefined,
      });

      return await saveLLMSettings(settingsCopy);
    } catch (error) {
      logger.error("Error in client saveSettings:", error);
      return false;
    }
  }
}

// Create a singleton instance
logger.log("Creating llmSecureClient singleton instance");
export const llmSecureClient = new LLMSecureClient();
logger.log("llmSecureClient singleton instance created");
