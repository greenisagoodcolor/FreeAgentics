"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import { type LLMClient, llmClient } from "@/lib/llm-client"
import { type LLMSecureClient, llmSecureClient } from "@/lib/llm-secure-client"
import type { LLMSettings } from "@/lib/llm-settings"
import { clientDefaultSettings } from "@/lib/llm-settings"
import { isFeatureEnabled } from "@/lib/feature-flags"

// Create the context with default values
interface LLMContextType {
  client: LLMClient | LLMSecureClient
  settings: LLMSettings
  updateSettings: (newSettings: Partial<LLMSettings>) => void
  saveSettings: () => Promise<boolean>
  isProcessing: boolean
  setIsProcessing: (isProcessing: boolean) => void
}

// Create context with default values
const LLMContext = createContext<LLMContextType>({
  client: isFeatureEnabled("useSecureApiStorage") ? llmSecureClient : llmClient,
  settings: clientDefaultSettings,
  updateSettings: () => {},
  saveSettings: async () => false,
  isProcessing: false,
  setIsProcessing: () => {},
})

// Provider component
export function LLMProvider({ children }: { children: React.ReactNode }) {
  console.log("LLMProvider rendering")
  const [settings, setSettings] = useState<LLMSettings>(clientDefaultSettings)
  const [isProcessing, setIsProcessing] = useState(false)

  // Determine which client to use based on feature flag
  const useSecureStorage = isFeatureEnabled("useSecureApiStorage")
  const activeClient = useSecureStorage ? llmSecureClient : llmClient

  console.log("LLMProvider using client:", {
    useSecureStorage,
    clientType: useSecureStorage ? "LLMSecureClient" : "LLMClient",
  })

  // Load initial settings from client
  useEffect(() => {
    console.log("LLMProvider useEffect running")
    try {
      console.log("LLM context useEffect - before getting client settings")
      console.log("activeClient available:", !!activeClient)
      console.log("activeClient.getSettings is function:", typeof activeClient.getSettings === "function")

      // Get settings from client
      console.log("About to call activeClient.getSettings()")
      let clientSettings
      try {
        clientSettings = activeClient.getSettings()
        console.log("activeClient.getSettings() returned successfully")

        // Log API key status
        console.log("API key status:", {
          hasApiKey: !!clientSettings.apiKey,
          apiKeyLength: clientSettings.apiKey ? clientSettings.apiKey.length : 0,
          hasApiKeySessionId: !!clientSettings.apiKeySessionId,
          provider: clientSettings.provider,
        })
      } catch (getSettingsError) {
        console.error("Error calling activeClient.getSettings():", getSettingsError)
        throw getSettingsError
      }

      console.log("LLM context useEffect - client settings retrieved:", {
        provider: clientSettings.provider,
        model: clientSettings.model,
        hasApiKey: !!clientSettings.apiKey,
        apiKeyLength: clientSettings.apiKey ? clientSettings.apiKey.length : 0,
        hasApiKeySessionId: !!clientSettings.apiKeySessionId,
        temperature: clientSettings.temperature,
        maxTokens: clientSettings.maxTokens,
        hasServerRef: "__server_ref" in clientSettings,
        keys: Object.keys(clientSettings),
      })

      // Check if we received a server reference
      if (clientSettings && typeof clientSettings === "object" && "__server_ref" in clientSettings) {
        console.log("Detected server reference in settings, creating clean object")
        // Create a clean settings object without server references
        const cleanSettings = {
          ...clientDefaultSettings,
          provider: clientSettings.provider || clientDefaultSettings.provider,
          model: clientSettings.model || clientDefaultSettings.model,
          temperature:
            typeof clientSettings.temperature === "number"
              ? clientSettings.temperature
              : clientDefaultSettings.temperature,
          maxTokens:
            typeof clientSettings.maxTokens === "number" ? clientSettings.maxTokens : clientDefaultSettings.maxTokens,
          topP: typeof clientSettings.topP === "number" ? clientSettings.topP : clientDefaultSettings.topP,
          frequencyPenalty:
            typeof clientSettings.frequencyPenalty === "number"
              ? clientSettings.frequencyPenalty
              : clientDefaultSettings.frequencyPenalty,
          presencePenalty:
            typeof clientSettings.presencePenalty === "number"
              ? clientSettings.presencePenalty
              : clientDefaultSettings.presencePenalty,
          systemFingerprint:
            typeof clientSettings.systemFingerprint === "boolean"
              ? clientSettings.systemFingerprint
              : clientDefaultSettings.systemFingerprint,
          apiKey: clientSettings.apiKey || undefined,
          apiKeySessionId: clientSettings.apiKeySessionId || undefined,
        }
        console.log("About to call setSettings with clean settings")
        setSettings(cleanSettings)

        console.log("Created clean settings object:", {
          ...cleanSettings,
          apiKey: cleanSettings.apiKey ? `[Length: ${cleanSettings.apiKey.length}]` : undefined,
          apiKeySessionId: cleanSettings.apiKeySessionId ? "[PRESENT]" : undefined,
          hasServerRef: "__server_ref" in cleanSettings,
        })
        return
      }

      // Ensure all required properties exist by merging with defaults
      console.log("Creating complete settings by merging with defaults")
      console.log("clientDefaultSettings before merge:", {
        ...clientDefaultSettings,
        hasServerRef: "__server_ref" in clientDefaultSettings,
      })

      // Create a clean copy of clientDefaultSettings first
      const cleanDefaults = { ...clientDefaultSettings }
      console.log("cleanDefaults after copy:", {
        ...cleanDefaults,
        hasServerRef: "__server_ref" in cleanDefaults,
      })

      // Now merge with client settings
      const completeSettings = {
        ...cleanDefaults,
        ...clientSettings,
        // Ensure numeric values are properly initialized
        temperature: clientSettings.temperature ?? cleanDefaults.temperature,
        maxTokens: clientSettings.maxTokens ?? cleanDefaults.maxTokens,
        topP: clientSettings.topP ?? cleanDefaults.topP,
        frequencyPenalty: clientSettings.frequencyPenalty ?? cleanDefaults.frequencyPenalty,
        presencePenalty: clientSettings.presencePenalty ?? cleanDefaults.presencePenalty,
      }

      console.log("completeSettings after merge:", {
        ...completeSettings,
        hasServerRef: "__server_ref" in completeSettings,
        keys: Object.keys(completeSettings),
      })

      console.log("About to call setSettings with complete settings")
      setSettings(completeSettings)

      console.log("LLM context initialized with settings:", {
        ...completeSettings,
        apiKey: completeSettings.apiKey ? `[Length: ${completeSettings.apiKey.length}]` : undefined,
        apiKeySessionId: completeSettings.apiKeySessionId ? "[PRESENT]" : undefined,
        provider: completeSettings.provider,
        hasServerRef: "__server_ref" in completeSettings,
      })
    } catch (error) {
      console.error("Error loading initial settings:", error)
      console.error("Error stack:", error instanceof Error ? error.stack : "No stack available")
      // Fall back to defaults
      console.log("Falling back to default settings")
      setSettings(clientDefaultSettings)
    }
  }, [activeClient])

  // Update settings in the client
  const updateSettings = (newSettings: Partial<LLMSettings>) => {
    console.log("updateSettings called with:", {
      ...newSettings,
      apiKey: newSettings.apiKey ? `[Length: ${newSettings.apiKey.length}]` : undefined,
      apiKeySessionId: newSettings.apiKeySessionId ? "[PRESENT]" : undefined,
      hasServerRef: "__server_ref" in newSettings,
    })
    try {
      if (!newSettings || typeof newSettings !== "object") {
        console.error("Invalid settings update")
        return
      }

      // Create a complete settings object with defaults
      const updatedSettings = { ...settings, ...newSettings }
      console.log("updatedSettings after merge:", {
        ...updatedSettings,
        apiKey: updatedSettings.apiKey ? `[Length: ${updatedSettings.apiKey.length}]` : undefined,
        apiKeySessionId: updatedSettings.apiKeySessionId ? "[PRESENT]" : undefined,
        hasServerRef: "__server_ref" in updatedSettings,
      })

      console.log("Updating settings in context:", {
        ...updatedSettings,
        apiKey: updatedSettings.apiKey ? `[Length: ${updatedSettings.apiKey.length}]` : undefined,
        apiKeySessionId: updatedSettings.apiKeySessionId ? "[PRESENT]" : undefined,
        provider: updatedSettings.provider,
      })

      // Update local state
      console.log("About to call setSettings")
      setSettings(updatedSettings)

      // Update client settings
      if (activeClient && typeof activeClient.updateSettings === "function") {
        console.log("About to call activeClient.updateSettings")
        activeClient.updateSettings(updatedSettings)
      } else {
        console.error("activeClient.updateSettings is not available")
      }
    } catch (error) {
      console.error("Error updating settings:", error)
    }
  }

  // Save settings to the server
  const saveSettings = async (): Promise<boolean> => {
    console.log("saveSettings called")
    try {
      console.log("Saving settings from context:", {
        ...settings,
        apiKey: settings.apiKey ? `[Length: ${settings.apiKey.length}]` : undefined,
        apiKeySessionId: settings.apiKeySessionId ? "[PRESENT]" : undefined,
        provider: settings.provider,
        hasServerRef: "__server_ref" in settings,
      })

      if (activeClient && typeof activeClient.saveSettings === "function") {
        console.log("About to call activeClient.saveSettings")
        return await activeClient.saveSettings()
      } else {
        console.error("activeClient.saveSettings is not available")
        return false
      }
    } catch (error) {
      console.error("Error saving settings:", error)
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
  }

  console.log("LLMProvider rendering with context value:", {
    clientAvailable: !!value.client,
    clientType: useSecureStorage ? "LLMSecureClient" : "LLMClient",
    settingsProvider: value.settings.provider,
    settingsModel: value.settings.model,
    isProcessing: value.isProcessing,
    settingsHasServerRef: "__server_ref" in value.settings,
  })

  return <LLMContext.Provider value={value}>{children}</LLMContext.Provider>
}

// Hook to use the LLM context
export function useLLM() {
  console.log("useLLM hook called")
  const context = useContext(LLMContext)
  console.log("useLLM returning context with:", {
    clientAvailable: !!context.client,
    clientType: isFeatureEnabled("useSecureApiStorage") ? "LLMSecureClient" : "LLMClient",
    settingsProvider: context.settings.provider,
    settingsModel: context.settings.model,
    isProcessing: context.isProcessing,
    settingsHasServerRef: "__server_ref" in context.settings,
  })
  return context
}
