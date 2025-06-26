"use client"

import {
  extractBeliefs as serverExtractBeliefs,
  generateKnowledgeEntries as serverGenerateKnowledgeEntries,
  generateResponse,
  validateApiKey,
  saveLLMSettings,
} from "@/lib/llm-service"
import type { LLMSettings } from "@/lib/llm-settings"
import type { KnowledgeEntry } from "@/lib/types"
import { clientDefaultSettings } from "@/lib/llm-settings"
import { storeSessionId, getSessionId, getApiKeyFromSession } from "@/lib/session-management"
import { createLogger } from "@/lib/debug-logger"
import { isBrowser } from "./browser-check"

const logger = createLogger("LLM-CLIENT")

// Client-side wrapper for the LLM service
export class LLMClient {
  private settings: LLMSettings
  private tempApiKey: string | null = null // Temporary storage for API key during operations
  private apiKeyRetrievalInProgress = false

  constructor(initialSettings: Partial<LLMSettings> = {}) {
    logger.info("LLMClient constructor called with:", {
      initialSettingsType: typeof initialSettings,
      isObject: initialSettings && typeof initialSettings === "object",
      hasServerRef: initialSettings && typeof initialSettings === "object" && "__server_ref" in initialSettings,
      keys: initialSettings && typeof initialSettings === "object" ? Object.keys(initialSettings) : [],
    })

    // Create a fresh settings object with default values from client-side defaults
    this.settings = { ...clientDefaultSettings } as LLMSettings

    // Try to load settings from localStorage first
    try {
      if (isBrowser) {
        const savedSettings = localStorage.getItem("llm-settings")
        if (savedSettings) {
          const parsedSettings = JSON.parse(savedSettings)
          logger.info("Loaded settings from localStorage:", {
            provider: parsedSettings.provider,
            model: parsedSettings.model,
            hasApiKeySessionId: !!parsedSettings.apiKeySessionId,
          })

          // Apply saved settings
          if (parsedSettings.provider) this.settings.provider = parsedSettings.provider
          if (parsedSettings.model) this.settings.model = parsedSettings.model
          if (typeof parsedSettings.temperature === "number") this.settings.temperature = parsedSettings.temperature
          if (typeof parsedSettings.maxTokens === "number") this.settings.maxTokens = parsedSettings.maxTokens
          if (typeof parsedSettings.topP === "number") this.settings.topP = parsedSettings.topP
          if (typeof parsedSettings.frequencyPenalty === "number")
            this.settings.frequencyPenalty = parsedSettings.frequencyPenalty
          if (typeof parsedSettings.presencePenalty === "number")
            this.settings.presencePenalty = parsedSettings.presencePenalty
          if (typeof parsedSettings.systemFingerprint === "boolean")
            this.settings.systemFingerprint = parsedSettings.systemFingerprint
          if (parsedSettings.apiKeySessionId) this.settings.apiKeySessionId = parsedSettings.apiKeySessionId
          if (typeof parsedSettings.maxAutonomousMessages === "number")
            this.settings.maxAutonomousMessages = parsedSettings.maxAutonomousMessages
          if (typeof parsedSettings.conversationCooldown === "number")
            this.settings.conversationCooldown = parsedSettings.conversationCooldown
        }
      }

      // Check if we have a session ID in localStorage but not in settings
      const sessionId = getSessionId(this.settings.provider)
      if (sessionId && !this.settings.apiKeySessionId) {
        logger.info(`Found session ID in localStorage for provider ${this.settings.provider}, adding to settings`)
        this.settings.apiKeySessionId = sessionId
      }
    } catch (e) {
      logger.warn("Could not load settings from localStorage:", e)
    }

    // Only copy properties from initialSettings if it's a valid object without server refs
    // and if they weren't already loaded from localStorage
    if (initialSettings && typeof initialSettings === "object" && !("__server_ref" in initialSettings)) {
      logger.info("Copying properties from initialSettings to this.settings")
      if (initialSettings.provider) this.settings.provider = initialSettings.provider
      if (initialSettings.model) this.settings.model = initialSettings.model
      if (typeof initialSettings.temperature === "number") this.settings.temperature = initialSettings.temperature
      if (typeof initialSettings.maxTokens === "number") this.settings.maxTokens = initialSettings.maxTokens
      if (typeof initialSettings.topP === "number") this.settings.topP = initialSettings.topP
      if (typeof initialSettings.frequencyPenalty === "number")
        this.settings.frequencyPenalty = initialSettings.frequencyPenalty
      if (typeof initialSettings.presencePenalty === "number")
        this.settings.presencePenalty = initialSettings.presencePenalty
      if (typeof initialSettings.systemFingerprint === "boolean")
        this.settings.systemFingerprint = initialSettings.systemFingerprint
      if (initialSettings.apiKeySessionId) this.settings.apiKeySessionId = initialSettings.apiKeySessionId
      if (typeof initialSettings.maxAutonomousMessages === "number")
        this.settings.maxAutonomousMessages = initialSettings.maxAutonomousMessages
      if (typeof initialSettings.conversationCooldown === "number")
        this.settings.conversationCooldown = initialSettings.conversationCooldown
    } else {
      logger.info("Not copying properties from initialSettings due to server ref or invalid object")
    }

    logger.info("LLMClient initialized with settings:", {
      ...this.settings,
      provider: this.settings.provider,
      hasApiKeySessionId: !!this.settings.apiKeySessionId,
    })
  }

  // Update settings
  updateSettings(newSettings: Partial<LLMSettings>): void {
    logger.info("LLMClient.updateSettings called with:", {
      newSettingsType: typeof newSettings,
      isObject: newSettings && typeof newSettings === "object",
      hasServerRef: newSettings && typeof newSettings === "object" && "__server_ref" in newSettings,
      keys: newSettings && typeof newSettings === "object" ? Object.keys(newSettings) : [],
      apiKeySessionIdPresent: newSettings && typeof newSettings === "object" ? "apiKeySessionId" in newSettings : false,
    })

    // Handle server references or undefined values
    if (!newSettings || typeof newSettings !== "object" || "__server_ref" in newSettings) {
      logger.warn("Invalid settings update or server reference detected, ignoring")
      return
    }

    // CRITICAL FIX: Ensure provider is correctly updated
    // Log the provider change explicitly
    if (newSettings.provider) {
      logger.info(`Updating provider from ${this.settings.provider} to ${newSettings.provider}`)
      this.settings.provider = newSettings.provider

      // Check if we need to update the apiKeySessionId when provider changes
      if (this.settings.provider !== newSettings.provider) {
        // Get session ID for the new provider
        const sessionId = getSessionId(newSettings.provider)
        if (sessionId) {
          logger.info(`Found session ID for new provider ${newSettings.provider}, updating apiKeySessionId`)
          this.settings.apiKeySessionId = sessionId
        } else {
          logger.info(`No session ID found for new provider ${newSettings.provider}, clearing apiKeySessionId`)
          delete this.settings.apiKeySessionId
        }
      }
    }

    // Update only the properties that are provided
    logger.info("Updating settings properties")
    if (newSettings.model) this.settings.model = newSettings.model
    if (typeof newSettings.temperature === "number") this.settings.temperature = newSettings.temperature
    if (typeof newSettings.maxTokens === "number") this.settings.maxTokens = newSettings.maxTokens
    if (typeof newSettings.topP === "number") this.settings.topP = newSettings.topP
    if (typeof newSettings.frequencyPenalty === "number") this.settings.frequencyPenalty = newSettings.frequencyPenalty
    if (typeof newSettings.presencePenalty === "number") this.settings.presencePenalty = newSettings.presencePenalty
    if (typeof newSettings.systemFingerprint === "boolean")
      this.settings.systemFingerprint = newSettings.systemFingerprint
    if (typeof newSettings.maxAutonomousMessages === "number")
      this.settings.maxAutonomousMessages = newSettings.maxAutonomousMessages
    if (typeof newSettings.conversationCooldown === "number")
      this.settings.conversationCooldown = newSettings.conversationCooldown

    // Handle API key session ID updates
    if ("apiKeySessionId" in newSettings) {
      logger.info("API key session ID found in settings update:", {
        apiKeySessionIdType: typeof newSettings.apiKeySessionId,
        apiKeySessionIdEmpty:
          typeof newSettings.apiKeySessionId === "string" ? newSettings.apiKeySessionId.trim() === "" : true,
      })

      // Only set if it's a non-empty string
      if (typeof newSettings.apiKeySessionId === "string" && newSettings.apiKeySessionId.trim() !== "") {
        logger.info(`Setting API key session ID from settings update`)
        this.settings.apiKeySessionId = newSettings.apiKeySessionId
      } else if (newSettings.apiKeySessionId === undefined || newSettings.apiKeySessionId === null) {
        logger.info("API key session ID is explicitly set to undefined/null, removing current API key session ID")
        delete this.settings.apiKeySessionId
      } else {
        logger.warn("API key session ID is present but not a valid string, ignoring:", {
          type: typeof newSettings.apiKeySessionId,
          value: String(newSettings.apiKeySessionId),
        })
      }
    }

    // Save settings to localStorage
    try {
      localStorage.setItem("llm-settings", JSON.stringify(this.settings))
      logger.info("Settings saved to localStorage")
    } catch (e) {
      logger.warn("Could not save settings to localStorage:", e)
    }

    logger.info("Settings updated to:", {
      ...this.settings,
      provider: this.settings.provider,
      hasApiKeySessionId: !!this.settings.apiKeySessionId,
    })
  }

  // Get current settings
  getSettings(): LLMSettings {
    logger.info("LLMClient.getSettings called")
    try {
      // Check if we have a session ID in localStorage but not in settings
      const sessionId = getSessionId(this.settings.provider)
      if (sessionId && !this.settings.apiKeySessionId) {
        logger.info(`Found session ID in localStorage for provider ${this.settings.provider}, adding to settings`)
        this.settings.apiKeySessionId = sessionId
      }

      // Return a copy to avoid reference issues
      const settingsCopy = { ...this.settings }
      logger.info("LLMClient.getSettings returning:", {
        ...settingsCopy,
        provider: settingsCopy.provider,
        hasApiKeySessionId: !!settingsCopy.apiKeySessionId,
      })
      return settingsCopy

    } catch (error) {
      logger.error("Error in LLMClient.getSettings:", error)
      // Return a safe default if there's an error
      return { ...clientDefaultSettings } as LLMSettings
    }
  }

  // Set API key securely
  async setApiKey(apiKey: string): Promise<boolean> {
    try {
      logger.info("LLMClient.setApiKey called")

      // Store the API key securely
      const response = await fetch("/api/api-key/store", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          provider: this.settings.provider,
          apiKey,
        }),
      })

      if (!response.ok) {
        logger.error(`Error storing API key: HTTP ${response.status}`)
        return false
      }

      const data = await response.json()

      if (!data.success) {
        logger.error("Failed to store API key:", data.message)
        return false
      }

      // Store the session ID
      const { sessionId } = data
      storeSessionId(this.settings.provider, sessionId)

      // Update settings with the session ID
      this.settings.apiKeySessionId = sessionId

      // Save settings to localStorage
      try {
        localStorage.setItem("llm-settings", JSON.stringify(this.settings))
        logger.info("Settings with API key session ID saved to localStorage:", {
          provider: this.settings.provider,
          hasApiKeySessionId: !!this.settings.apiKeySessionId,
          apiKeySessionId: this.settings.apiKeySessionId,
        })
      } catch (e) {
        logger.warn("Could not save settings to localStorage:", e)
      }

      return true
    } catch (error) {
      logger.error("Error setting API key:", error)
      return false
    }
  }

  // Get API key securely for operations
  private async getApiKey(): Promise<string | null> {
    try {
      // If we already have a temporary API key, use it
      if (this.tempApiKey) {
        return this.tempApiKey
      }

      // Prevent multiple simultaneous retrievals
      if (this.apiKeyRetrievalInProgress) {
        logger.info("API key retrieval already in progress, waiting...")
        // Wait for the current retrieval to complete
        await new Promise((resolve) => setTimeout(resolve, 100))
        return this.getApiKey()
      }

      this.apiKeyRetrievalInProgress = true

      try {
        // Get the session ID from settings or localStorage
        const sessionId = this.settings.apiKeySessionId || getSessionId(this.settings.provider)

        if (!sessionId) {
          logger.warn("No API key session ID available")
          return null
        }

        // Retrieve the API key
        this.tempApiKey = await getApiKeyFromSession(this.settings.provider)

        if (!this.tempApiKey) {
          logger.warn("Failed to retrieve API key from session")
        }

        return this.tempApiKey
      } finally {
        this.apiKeyRetrievalInProgress = false
      }
    } catch (error) {
      logger.error("Error getting API key:", error)
      this.apiKeyRetrievalInProgress = false
      return null
    }
  }

  // Clear temporary API key after operations
  private clearTempApiKey(): void {
    this.tempApiKey = null
  }

  // Generate a response using a system prompt
  async generateResponse(systemPrompt: string, userPrompt: string): Promise<string> {
    try {
      logger.info("[LLM CLIENT] generateResponse called with:", {
        systemPromptLength: systemPrompt?.length,
        userPromptLength: userPrompt?.length,
        provider: this.settings.provider,
        model: this.settings.model,
      })

      // Get the API key
      const apiKey = await this.getApiKey()

      // Clear the temporary API key after use
      this.clearTempApiKey()

      // Ensure API key is available
      if (!apiKey) {
        logger.warn("[LLM CLIENT] No API key available")
        return "Error: No API key available. Please set an API key in the settings."
      }

      // Create a copy of settings to ensure we're not passing a reference
      const settingsCopy = { ...this.settings }

      // Add the API key to the settings copy for the server call
      settingsCopy.apiKey = apiKey

      // Call the server-side function with the copy
      const response = await generateResponse(systemPrompt, userPrompt, settingsCopy)
      logger.info("[LLM CLIENT] Response received from server:", { responseLength: response?.length })
      return response
    } catch (error) {
      logger.error("[LLM CLIENT] Error in generateResponse:", error)
      return `Error: ${error instanceof Error ? error.message : "Unknown error"}`
    }
  }

  // Extract beliefs from conversation
  async extractBeliefs(conversationText: string, agentName: string, extractionPriorities: string): Promise<string> {
    try {
      logger.info("LLMClient.extractBeliefs called")

      // Get the API key
      const apiKey = await this.getApiKey()

      // Clear the temporary API key after use
      this.clearTempApiKey()

      // Ensure API key is available
      if (!apiKey) {
        logger.warn("No API key available")
        return "Error: No API key available. Please set an API key in the settings."
      }

      // Create a copy of settings to ensure we're not passing a reference
      const settingsCopy = { ...this.settings }

      // Add the API key to the settings copy for the server call
      settingsCopy.apiKey = apiKey

      // Log the settings to verify API key is present
      logger.info("Belief extraction settings:", {
        provider: settingsCopy.provider,
        model: settingsCopy.model,
        hasApiKey: !!settingsCopy.apiKey,
      })

      return await serverExtractBeliefs(conversationText, agentName, extractionPriorities, settingsCopy)
    } catch (error) {
      logger.error("Error in client extractBeliefs:", error)
      return `Error: ${error instanceof Error ? error.message : "Unknown error"}`
    }
  }

  // Generate knowledge entries from beliefs
  async generateKnowledgeEntries(beliefs: string): Promise<KnowledgeEntry[]> {
    try {
      logger.info("LLMClient.generateKnowledgeEntries called")

      // Get the API key
      const apiKey = await this.getApiKey()

      // Clear the temporary API key after use
      this.clearTempApiKey()

      // Ensure API key is available
      if (!apiKey) {
        logger.warn("No API key available")
        return [
          {
            id: `error-${Date.now()}`,
            title: "Error",
            content: "No API key available. Please set an API key in the settings.",
            timestamp: new Date(),
            tags: ["error"],
          },
        ]
      }

      // Create a copy of settings to ensure we're not passing a reference
      const settingsCopy = { ...this.settings }

      // Add the API key to the settings copy for the server call
      settingsCopy.apiKey = apiKey

      return await serverGenerateKnowledgeEntries(beliefs, settingsCopy)
    } catch (error) {
      logger.error("Error in client generateKnowledgeEntries:", error)
      return [
        {
          id: `error-${Date.now()}`,
          title: "Error",
          content: error instanceof Error ? error.message : "Unknown error",
          timestamp: new Date(),
          tags: ["error"],
        },
      ]
    }
  }

  // Stream response
  async streamResponse(
    systemPrompt: string,
    userPrompt: string,
    onChunk?: ((text: string, isComplete: boolean) => void) | null | undefined,
  ): Promise<string> {
    logger.info("[LLM CLIENT] streamResponse called with:", {
      systemPromptLength: systemPrompt?.length,
      userPromptLength: userPrompt?.length,
      hasOnChunkCallback: typeof onChunk === "function",
      onChunkType: typeof onChunk,
    })

    try {
      // CRITICAL FIX: More robust callback handling
      // Create a truly safe callback that won't throw if onChunk is not a function
      const safeCallback = (text: string, isComplete: boolean): void => {
        try {
          logger.info("[LLM CLIENT] safeCallback called with:", { textLength: text?.length, isComplete })
          if (typeof onChunk === "function") {
            logger.info("[LLM CLIENT] Calling onChunk function")
            onChunk(text, isComplete)
          } else {
            logger.info("[LLM CLIENT] Warning: onChunk is not a function", {
              onChunkType: typeof onChunk,
              text: text?.substring(0, 20) + "...",
              isComplete,
            })
          }
        } catch (callbackError) {
          logger.error("[LLM CLIENT] Error executing onChunk callback:", callbackError)
        }
      }

      // Get the API key
      const apiKey = await this.getApiKey()

      // Clear the temporary API key after use
      this.clearTempApiKey()

      // Ensure API key is available
      if (!apiKey) {
        logger.warn("[LLM CLIENT] No API key available")
        safeCallback("Error: No API key available. Please set an API key in the settings.", false)
        safeCallback("", true)
        return "Error: No API key available. Please set an API key in the settings."
      }

      // Use true streaming instead of simulated streaming
      let fullResponse = ""
      let streamingFailed = false

      try {
        logger.info("[LLM CLIENT] Attempting to use true streaming response")

        // Create a copy of settings to ensure we're not passing a reference
        const settingsCopy = { ...this.settings }

        // Add the API key to the settings copy for the server call
        settingsCopy.apiKey = apiKey

        // Call the new streaming API endpoint
        logger.info("[LLM CLIENT] Calling streaming API endpoint")

        const response = await fetch("/api/llm/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            systemPrompt,
            userPrompt,
            settings: settingsCopy,
          }),
        })

        if (!response.ok) {
          throw new Error(`Stream API error: ${response.status} ${response.statusText}`)
        }

        if (!response.body) {
          throw new Error("Response body is null")
        }

        // Read the streaming response
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            const chunk = decoder.decode(value, { stream: true })
            buffer += chunk

            // Process complete lines from the buffer
            let lineEnd = buffer.indexOf("\n")
            while (lineEnd !== -1) {
              const line = buffer.substring(0, lineEnd).trim()
              buffer = buffer.substring(lineEnd + 1)

              if (line) {
                try {
                  const chunkData = JSON.parse(line)
                  if (chunkData.text) {
                    fullResponse += chunkData.text
                    safeCallback(chunkData.text, false)
                  }

                  // Check if this is the completion signal
                  if (chunkData.isComplete) {
                    logger.info("[LLM CLIENT] Stream completed")
                    safeCallback("", true)
                    return fullResponse
                  }
                } catch (parseError) {
                  logger.error("[LLM CLIENT] Error parsing chunk:", parseError, "Line:", line)
                }
              }

              lineEnd = buffer.indexOf("\n")
            }
          }
        } finally {
          reader.releaseLock()
        }

        // Signal completion if we reach here
        logger.info("[LLM CLIENT] Stream ended without completion signal")
        safeCallback("", true)
      } catch (streamError) {
        logger.error("[LLM CLIENT] Error in true streaming response:", streamError)
        streamingFailed = true
      }

      // If streaming failed, fall back to non-streaming
      if (streamingFailed) {
        logger.info("[LLM CLIENT] True streaming failed, falling back to non-streaming")
        fullResponse = await this.generateResponse(systemPrompt, userPrompt)

        // Deliver the full response at once - using safe callback
        logger.info("[LLM CLIENT] Delivering full response at once")
        safeCallback(fullResponse, false)
        safeCallback("", true)
      }

      return fullResponse
    } catch (error) {
      logger.error("[LLM CLIENT] Error in streamResponse:", error)

      // Try to notify through callback if possible - using safe callback
      const errorMessage = `Error: ${error instanceof Error ? error.message : String(error)}`
      try {
        logger.info("[LLM CLIENT] Attempting to notify error through callback")
        if (typeof onChunk === "function") {
          logger.info("[LLM CLIENT] Calling onChunk with error message")
          onChunk(errorMessage, false)
          onChunk("", true)
        } else {
          logger.info("[LLM CLIENT] Cannot notify error: onChunk is not a function")
        }
      } catch (callbackError) {
        logger.error("[LLM CLIENT] Error calling onChunk callback with error:", callbackError)
      }

      // Return error message as string
      return errorMessage
    }
  }

  // Validate API key
  async validateApiKey(
    provider: "openai" | "openrouter",
    apiKey: string,
  ): Promise<{ valid: boolean; message?: string }> {
    try {
      logger.info("LLMClient.validateApiKey called")
      return await validateApiKey(provider, apiKey)
    } catch (error) {
      logger.error("Error in client validateApiKey:", error)
      return { valid: false, message: error instanceof Error ? error.message : "Error validating API key" }
    }
  }

  // Save settings
  async saveSettings(): Promise<boolean> {
    try {
      logger.info("LLMClient.saveSettings called")
      logger.info("Current settings to save:", {
        ...this.settings,
        hasApiKeySessionId: !!this.settings.apiKeySessionId,
      })
      return await saveLLMSettings(this.settings)
    } catch (error) {
      logger.error("Error in client saveSettings:", error)
      return false
    }
  }
}

// Create a singleton instance
logger.info("Creating llmClient singleton instance")
export const llmClient = new LLMClient()
logger.info("llmClient singleton instance created")
