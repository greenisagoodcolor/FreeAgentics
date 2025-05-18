/**
 * API Key Migration Utility
 *
 * This module provides functions to help migrate API keys from
 * localStorage to the secure server-side storage.
 */

import { storeSessionId } from "./session-management"
import { isFeatureEnabled } from "./feature-flags"

/**
 * Migrates an API key from localStorage to secure storage
 * @param provider The API provider
 * @param apiKey The API key to migrate
 * @returns Promise resolving to true if migration was successful
 */
export async function migrateApiKey(provider: string, apiKey: string): Promise<boolean> {
  try {
    console.log(`Migrating API key for ${provider} to secure storage`)

    // Only proceed if secure storage is enabled
    if (!isFeatureEnabled("useSecureApiStorage")) {
      console.log("Secure API storage is not enabled, skipping migration")
      return false
    }

    // Store the API key securely
    const response = await fetch("/api/api-key/store", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        provider,
        apiKey,
      }),
    })

    const data = await response.json()
    if (data.success && data.sessionId) {
      console.log(`API key for ${provider} migrated successfully`)
      // Store the session ID in localStorage
      storeSessionId(provider, data.sessionId)
      return true
    } else {
      console.error(`Failed to migrate API key for ${provider}:`, data.message)
      return false
    }
  } catch (error) {
    console.error(`Error migrating API key for ${provider}:`, error)
    return false
  }
}

/**
 * Checks if there are API keys in localStorage that need to be migrated
 * @returns Array of providers that have API keys in localStorage
 */
export function checkForApiKeysToMigrate(): string[] {
  try {
    const providersToMigrate: string[] = []

    // Check for llm-settings in localStorage
    const savedSettings = localStorage.getItem("llm-settings")
    if (savedSettings) {
      const parsedSettings = JSON.parse(savedSettings)
      if (parsedSettings.apiKey && parsedSettings.provider) {
        providersToMigrate.push(parsedSettings.provider)
      }
    }

    return providersToMigrate
  } catch (error) {
    console.error("Error checking for API keys to migrate:", error)
    return []
  }
}

/**
 * Migrates all API keys found in localStorage to secure storage
 * @returns Promise resolving to an array of providers that were migrated
 */
export async function migrateAllApiKeys(): Promise<string[]> {
  try {
    const migratedProviders: string[] = []

    // Check for llm-settings in localStorage
    const savedSettings = localStorage.getItem("llm-settings")
    if (savedSettings) {
      const parsedSettings = JSON.parse(savedSettings)
      if (parsedSettings.apiKey && parsedSettings.provider) {
        const success = await migrateApiKey(parsedSettings.provider, parsedSettings.apiKey)
        if (success) {
          migratedProviders.push(parsedSettings.provider)

          // Remove the API key from localStorage settings
          parsedSettings.apiKey = undefined
          localStorage.setItem("llm-settings", JSON.stringify(parsedSettings))
        }
      }
    }

    return migratedProviders
  } catch (error) {
    console.error("Error migrating all API keys:", error)
    return []
  }
}
