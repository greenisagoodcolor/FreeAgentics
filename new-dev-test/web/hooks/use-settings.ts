import { useState, useCallback, useEffect } from "react";

export type LLMProvider = "openai" | "anthropic" | "ollama";
export type LLMModel = string;

export interface Settings {
  llmProvider: LLMProvider;
  llmModel: LLMModel;
  gnnEnabled: boolean;
  debugLogs: boolean;
  autoSuggest: boolean;
}

export interface SettingsState {
  settings: Settings;
  updateSettings: (updates: Partial<Settings>) => void;
  resetSettings: () => void;
}

const SETTINGS_KEY = "freeagentics_settings";

const DEFAULT_SETTINGS: Settings = {
  llmProvider: "openai",
  llmModel: "gpt-4",
  gnnEnabled: true,
  debugLogs: false,
  autoSuggest: true,
};

const LLM_MODELS: Record<LLMProvider, string[]> = {
  openai: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
  anthropic: ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
  ollama: ["llama2", "mistral", "codellama"],
};

export function useSettings(): SettingsState {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);

  // Load settings from localStorage on mount
  useEffect(() => {
    try {
      const storedSettings = localStorage.getItem(SETTINGS_KEY);
      if (storedSettings) {
        const parsed = JSON.parse(storedSettings);
        setSettings({ ...DEFAULT_SETTINGS, ...parsed });
      }
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  }, []);

  const updateSettings = useCallback((updates: Partial<Settings>) => {
    setSettings((current) => {
      const newSettings = { ...current, ...updates };

      // Validate model for provider
      if (updates.llmProvider && !updates.llmModel) {
        // If provider changed but model wasn't specified, set default model for new provider
        const availableModels = LLM_MODELS[updates.llmProvider];
        if (availableModels && !availableModels.includes(current.llmModel)) {
          newSettings.llmModel = availableModels[0];
        }
      }

      // Persist to localStorage
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(newSettings));
      } catch (error) {
        console.error("Failed to save settings:", error);
      }

      return newSettings;
    });
  }, []);

  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(DEFAULT_SETTINGS));
    } catch (error) {
      console.error("Failed to reset settings:", error);
    }
  }, []);

  return {
    settings,
    updateSettings,
    resetSettings,
  };
}

// Export constants for use in components
export { LLM_MODELS, DEFAULT_SETTINGS };
