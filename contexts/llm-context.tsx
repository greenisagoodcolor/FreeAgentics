"use client"

import React, { createContext, useContext, useEffect, useState, useCallback } from "react"
import type { LLMSettings } from "@/lib/llm-settings"
import { llmClient } from "@/lib/llm-client"
import { llmSecureClient } from "@/lib/llm-secure-client"
import { isFeatureEnabled } from "@/lib/feature-flags"
import { createLogger } from "@/lib/debug-logger"

const logger = createLogger("LLM-CONTEXT")

// Create the context with default values
export interface LLMContextType {
  // Client management
  client: typeof llmClient | typeof llmSecureClient | null
  clientType: "LLMClient" | "LLMSecureClient" | null

  // Settings management
  settings: LLMSettings | null
  updateSettings: (newSettings: Partial<LLMSettings>) => void
  saveSettings: () => Promise<boolean>

  // Status
  isProcessing: boolean
  setIsProcessing: (processing: boolean) => void
}

// Create context with default values
const LLMContext = createContext<LLMContextType>({
  client: isFeatureEnabled("useSecureApiStorage") ? llmSecureClient : llmClient,
  settings: null,
  updateSettings: () => {},
  saveSettings: async () => false,
  isProcessing: false,
  setIsProcessing: () => {},
  clientType: isFeatureEnabled("useSecureApiStorage") ? "LLMSecureClient" : "LLMClient",
})

// Provider component
export function LLMProvider({ children }: { children: React.ReactNode }) {
  logger.info("LLMProvider rendering")
  const [settings, setSettings] = useState<LLMSettings | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  // Determine which client to use based on feature flag
  const useSecureStorage = isFeatureEnabled("useSecureApiStorage")
  const activeClient = useSecureStorage ? llmSecureClient : llmClient

  logger.info("LLMProvider using client:", {
    useSecureStorage,
    clientType: useSecureStorage ? "LLMSecureClient" : "LLMClient",
  })

  // Load initial settings from the client
  useEffect(() => {
    logger.info("Loading initial settings from client")
    
    if (!activeClient) {
      logger.info("No active client available, using defaults")
      setSettings(null)
      return
    }

    try {
      const clientSettings = activeClient.getSettings()
      logger.info("Retrieved settings from client:", {
        provider: clientSettings?.provider,
        model: clientSettings?.model,
        hasApiKey: !!clientSettings?.apiKey,
        hasApiKeySessionId: !!clientSettings?.apiKeySessionId,
        hasServerRef: clientSettings && typeof clientSettings === "object" && "__server_ref" in clientSettings,
      })

      // Create clean merged settings with robust null/undefined handling
      const mergedSettings: LLMSettings = {
        provider: clientSettings?.provider || "openai",
        model: clientSettings?.model || "gpt-4o",
        temperature: typeof clientSettings?.temperature === "number" ? clientSettings.temperature : 0.7,
        maxTokens: typeof clientSettings?.maxTokens === "number" ? clientSettings.maxTokens : 1024,
        topP: typeof clientSettings?.topP === "number" ? clientSettings.topP : 0.9,
        frequencyPenalty: typeof clientSettings?.frequencyPenalty === "number" ? clientSettings.frequencyPenalty : 0,
        presencePenalty: typeof clientSettings?.presencePenalty === "number" ? clientSettings.presencePenalty : 0,
        systemFingerprint: typeof clientSettings?.systemFingerprint === "boolean" ? clientSettings.systemFingerprint : false,
        // Handle optional properties - only include if they have valid values
        ...(clientSettings?.apiKey && { apiKey: clientSettings.apiKey }),
        ...(clientSettings?.apiKeySessionId && { apiKeySessionId: clientSettings.apiKeySessionId }),
        ...(typeof clientSettings?.maxAutonomousMessages === "number" && { maxAutonomousMessages: clientSettings.maxAutonomousMessages }),
        ...(typeof clientSettings?.conversationCooldown === "number" && { conversationCooldown: clientSettings.conversationCooldown }),
      }

      setSettings(mergedSettings)

      logger.info("LLM context initialized with settings:", {
        provider: mergedSettings.provider,
        model: mergedSettings.model,
        hasApiKey: !!mergedSettings.apiKey,
        hasApiKeySessionId: !!mergedSettings.apiKeySessionId,
      })
    } catch (error) {
      logger.error("Error loading initial settings:", error)
      setSettings(null)
    }
  }, [activeClient])

  // Update settings in the client
  const updateSettings = (newSettings: Partial<LLMSettings>) => {
    logger.info("updateSettings called with:", {
      provider: newSettings.provider,
      model: newSettings.model,
      hasApiKey: !!newSettings.apiKey,
      hasApiKeySessionId: !!newSettings.apiKeySessionId,
    })

    if (!newSettings || typeof newSettings !== "object") {
      logger.error("Invalid settings update")
      return
    }

    try {
      // Merge new settings with current settings, ensuring required fields are present
      const updatedSettings: LLMSettings = {
        // Provide defaults for required fields
        provider: newSettings.provider || settings?.provider || "openai",
        model: newSettings.model || settings?.model || "gpt-4o",
        temperature: newSettings.temperature ?? settings?.temperature ?? 0.7,
        maxTokens: newSettings.maxTokens ?? settings?.maxTokens ?? 1024,
        topP: newSettings.topP ?? settings?.topP ?? 0.9,
        frequencyPenalty: newSettings.frequencyPenalty ?? settings?.frequencyPenalty ?? 0,
        presencePenalty: newSettings.presencePenalty ?? settings?.presencePenalty ?? 0,
        systemFingerprint: newSettings.systemFingerprint ?? settings?.systemFingerprint ?? false,
        // Handle optional properties
        ...(newSettings.apiKey !== undefined && { apiKey: newSettings.apiKey }),
        ...(newSettings.apiKeySessionId !== undefined && { apiKeySessionId: newSettings.apiKeySessionId }),
        ...(newSettings.maxAutonomousMessages !== undefined && { maxAutonomousMessages: newSettings.maxAutonomousMessages }),
        ...(newSettings.conversationCooldown !== undefined && { conversationCooldown: newSettings.conversationCooldown }),
      }

      // Update local state
      setSettings(updatedSettings)

      // Update client settings
      if (activeClient && typeof activeClient.updateSettings === "function") {
        activeClient.updateSettings(updatedSettings)
      } else {
        logger.error("activeClient.updateSettings is not available")
      }
    } catch (error) {
      logger.error("Error updating settings:", error)
    }
  }

  // Save settings to the server
  const saveSettings = async (): Promise<boolean> => {
    logger.info("saveSettings called")
    try {
      if (activeClient && typeof activeClient.saveSettings === "function") {
        return await activeClient.saveSettings()
      } else {
        logger.error("activeClient.saveSettings is not available")
        return false
      }
    } catch (error) {
      logger.error("Error saving settings:", error)
      return false
    }
  }

  // Context value
  const value = {
    client: activeClient,
    settings,
    updateSettings,
    saveSettings,
    isProcessing,
    setIsProcessing,
    clientType: (useSecureStorage ? "LLMSecureClient" : "LLMClient") as "LLMSecureClient" | "LLMClient",
  }

  logger.info("LLMProvider rendering with context value:", {
    clientAvailable: !!value.client,
    clientType: useSecureStorage ? "LLMSecureClient" : "LLMClient",
    settingsProvider: value.settings?.provider,
    settingsModel: value.settings?.model,
    isProcessing: value.isProcessing,
  })

  return <LLMContext.Provider value={value}>{children}</LLMContext.Provider>
}

// Hook to use the LLM context
export function useLLM() {
  logger.info("useLLM hook called")
  const context = useContext(LLMContext)
  logger.info("useLLM returning context with:", {
    clientAvailable: !!context.client,
    clientType: context.clientType,
    settingsProvider: context.settings?.provider,
    settingsModel: context.settings?.model,
    isProcessing: context.isProcessing,
  })
  return context
}
