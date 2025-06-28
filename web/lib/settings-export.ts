import type { LLMSettings } from "./llm-settings";
import { createLogger } from "./debug-logger";
import { getApiKey } from "./api-key-storage";

// Create a module-specific logger
const logger = createLogger("settings-export");

/**
 * Prepare settings for export
 * @param settings LLM settings to export
 * @param includeApiKeys Whether to include API keys in the export
 * @returns Settings object ready for export
 */
export async function prepareSettingsForExport(
  settings: LLMSettings,
  includeApiKeys = false,
): Promise<LLMSettings> {
  // Create a copy of the settings
  const exportSettings = { ...settings };

  // Handle API keys for export
  if (includeApiKeys && settings.apiKeySessionId) {
    try {
      // Retrieve the actual API key for export
      const apiKey = await getApiKey(
        settings.provider,
        settings.apiKeySessionId,
      );
      if (apiKey) {
        // Add the API key to the export settings
        exportSettings.apiKey = apiKey;
      } else {
        logger.warn("Could not retrieve API key for export");
      }
    } catch (error) {
      logger.error("Error retrieving API key for export", error);
    }
  }

  // Always remove the session ID from exports as it's only valid for the current browser
  delete exportSettings.apiKeySessionId;

  logger.debug("Prepared settings for export", {
    provider: exportSettings.provider,
    model: exportSettings.model,
    includesApiKey: includeApiKeys && !!exportSettings.apiKey,
  });

  return exportSettings;
}

/**
 * Parse settings from JSON
 * @param json JSON string containing settings
 * @returns Parsed settings object
 */
export function parseSettingsFromJSON(json: string): LLMSettings | undefined {
  try {
    const parsed = JSON.parse(json);

    // Basic validation to ensure it's a settings object
    if (!parsed || typeof parsed !== "object") {
      logger.warn("Invalid settings JSON: not an object");
      return undefined;
    }

    if (!parsed.provider || !parsed.model) {
      logger.warn("Invalid settings JSON: missing required fields");
      return undefined;
    }

    // Create a clean settings object with required fields
    const settings: LLMSettings = {
      provider: parsed.provider,
      model: parsed.model,
      temperature: parsed.temperature ?? 0.7,
      maxTokens: parsed.maxTokens ?? 1024,
      topP: parsed.topP ?? 0.9,
      frequencyPenalty: parsed.frequencyPenalty ?? 0,
      presencePenalty: parsed.presencePenalty ?? 0,
      systemFingerprint: parsed.systemFingerprint ?? false,
      maxAutonomousMessages: parsed.maxAutonomousMessages ?? 4,
      conversationCooldown: parsed.conversationCooldown ?? 5000,
    };

    // Handle API key if present
    if (
      "apiKey" in parsed &&
      typeof parsed.apiKey === "string" &&
      parsed.apiKey.trim() !== ""
    ) {
      logger.debug(
        `Valid API key found in settings JSON (length: ${parsed.apiKey.length})`,
      );
      // Note: We don't set apiKeySessionId here - that will be handled by the LLMClient
      // when the settings are applied and the API key is stored securely
      settings.apiKey = parsed.apiKey;
    } else {
      logger.debug("No valid API key found in settings JSON");
    }

    return settings;
  } catch (error) {
    logger.error("Error parsing settings JSON", error);
    return undefined;
  }
}
