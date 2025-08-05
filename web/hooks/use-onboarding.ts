import { useState, useEffect } from "react";
import { useSettings } from "./use-settings";

export interface OnboardingState {
  needsOnboarding: boolean;
  isOnboardingComplete: boolean;
  markOnboardingComplete: () => void;
  resetOnboarding: () => void;
  hasValidApiKey: boolean;
}

const ONBOARDING_KEY = "freeagentics_onboarding_complete";

/**
 * Hook to manage first-run onboarding experience
 * Shows API key setup if user has no valid API keys configured
 */
export function useOnboarding(): OnboardingState {
  const { settings, isLoading: settingsLoading } = useSettings();
  const [isOnboardingComplete, setIsOnboardingComplete] = useState<boolean>(false);

  // Check if user has any valid API key configured
  const hasValidApiKey = !!(
    settings.openaiApiKey?.trim() || 
    settings.anthropicApiKey?.trim()
  );

  // Load onboarding completion status from localStorage
  useEffect(() => {
    try {
      const completed = localStorage.getItem(ONBOARDING_KEY);
      setIsOnboardingComplete(completed === "true");
    } catch (error) {
      console.warn("Failed to load onboarding status:", error);
      setIsOnboardingComplete(false);
    }
  }, []);

  // Determine if user needs onboarding
  const needsOnboarding = !settingsLoading && !hasValidApiKey && !isOnboardingComplete;

  const markOnboardingComplete = () => {
    try {
      localStorage.setItem(ONBOARDING_KEY, "true");
      setIsOnboardingComplete(true);
    } catch (error) {
      console.error("Failed to save onboarding completion:", error);
    }
  };

  const resetOnboarding = () => {
    try {
      localStorage.removeItem(ONBOARDING_KEY);
      setIsOnboardingComplete(false);
    } catch (error) {
      console.error("Failed to reset onboarding:", error);
    }
  };

  return {
    needsOnboarding,
    isOnboardingComplete,
    markOnboardingComplete,
    resetOnboarding,
    hasValidApiKey,
  };
}