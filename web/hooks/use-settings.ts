import { useState, useCallback, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api-client";

export type LLMProvider = "openai" | "anthropic" | "ollama";
export type LLMModel = string;

export interface Settings {
  llmProvider: LLMProvider;
  llmModel: LLMModel;
  openaiApiKey: string;
  anthropicApiKey: string;
  gnnEnabled: boolean;
  debugLogs: boolean;
  autoSuggest: boolean;
}

export interface SettingsState {
  settings: Settings;
  updateSettings: (updates: Partial<Settings>) => void;
  resetSettings: () => void;
  isSaving: boolean;
  saveError: string | null;
  isLoading: boolean;
}

const SETTINGS_KEY = "freeagentics_settings";

const DEFAULT_SETTINGS: Settings = {
  llmProvider: "openai",
  llmModel: "gpt-4",
  openaiApiKey: "",
  anthropicApiKey: "",
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
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Load settings from backend on mount, fallback to localStorage
  useEffect(() => {
    const loadSettings = async () => {
      try {
        // Try to load from backend first
        const response = await apiClient.getSettings();
        if (response.success && response.data) {
          const backendSettings = {
            llmProvider: response.data.llmProvider as Settings["llmProvider"],
            llmModel: response.data.llmModel,
            openaiApiKey: response.data.openaiApiKey || "",
            anthropicApiKey: response.data.anthropicApiKey || "",
            gnnEnabled: response.data.gnnEnabled,
            debugLogs: response.data.debugLogs,
            autoSuggest: response.data.autoSuggest,
          };
          setSettings(backendSettings);
          // Sync to localStorage
          localStorage.setItem(SETTINGS_KEY, JSON.stringify(backendSettings));
        } else {
          // Fallback to localStorage
          const storedSettings = localStorage.getItem(SETTINGS_KEY);
          if (storedSettings) {
            const parsed = JSON.parse(storedSettings);
            setSettings({ ...DEFAULT_SETTINGS, ...parsed });
          }
        }
      } catch (error) {
        console.error("Failed to load settings from backend:", error);
        // Fallback to localStorage
        try {
          const storedSettings = localStorage.getItem(SETTINGS_KEY);
          if (storedSettings) {
            const parsed = JSON.parse(storedSettings);
            setSettings({ ...DEFAULT_SETTINGS, ...parsed });
          }
        } catch (localError) {
          console.error("Failed to load settings from localStorage:", localError);
        }
      } finally {
        setIsLoading(false);
      }
    };

    loadSettings();
  }, []);

  const updateSettings = useCallback((updates: Partial<Settings>) => {
    setSaveError(null);

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

      // Persist to localStorage immediately for optimistic update
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(newSettings));
      } catch (error) {
        console.error("Failed to save settings to localStorage:", error);
      }

      // Clear existing timeout
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }

      // Debounce backend save by 500ms
      saveTimeoutRef.current = setTimeout(async () => {
        setIsSaving(true);
        try {
          // Map frontend key names to backend format
          const backendUpdates: Record<string, any> = {};
          if ("llmProvider" in updates) backendUpdates.llm_provider = updates.llmProvider;
          if ("llmModel" in updates) backendUpdates.llm_model = updates.llmModel;
          if ("openaiApiKey" in updates) backendUpdates.openai_api_key = updates.openaiApiKey;
          if ("anthropicApiKey" in updates)
            backendUpdates.anthropic_api_key = updates.anthropicApiKey;
          if ("gnnEnabled" in updates) backendUpdates.gnn_enabled = updates.gnnEnabled;
          if ("debugLogs" in updates) backendUpdates.debug_logs = updates.debugLogs;
          if ("autoSuggest" in updates) backendUpdates.auto_suggest = updates.autoSuggest;

          const response = await apiClient.updateSettings(backendUpdates);
          if (!response.success) {
            setSaveError(response.error || "Failed to save settings");
            console.error("Failed to save settings to backend:", response.error);
          }
        } catch (error) {
          setSaveError("Failed to save settings. Please try again.");
          console.error("Error saving settings to backend:", error);
        } finally {
          setIsSaving(false);
        }
      }, 500);

      return newSettings;
    });
  }, []);

  const resetSettings = useCallback(async () => {
    setSaveError(null);
    setSettings(DEFAULT_SETTINGS);

    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(DEFAULT_SETTINGS));
    } catch (error) {
      console.error("Failed to reset settings in localStorage:", error);
    }

    // Also clear API keys on backend
    setIsSaving(true);
    try {
      await apiClient.clearApiKeys();
      // Update backend with default settings
      await apiClient.updateAllSettings({
        llmProvider: DEFAULT_SETTINGS.llmProvider,
        llmModel: DEFAULT_SETTINGS.llmModel,
        openaiApiKey: "",
        anthropicApiKey: "",
        gnnEnabled: DEFAULT_SETTINGS.gnnEnabled,
        debugLogs: DEFAULT_SETTINGS.debugLogs,
        autoSuggest: DEFAULT_SETTINGS.autoSuggest,
      });
    } catch (error) {
      setSaveError("Failed to reset settings on server");
      console.error("Error resetting settings on backend:", error);
    } finally {
      setIsSaving(false);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, []);

  return {
    settings,
    updateSettings,
    resetSettings,
    isSaving,
    saveError,
    isLoading,
  };
}

// Export constants for use in components
export { LLM_MODELS, DEFAULT_SETTINGS };
