/**
 * Feature Flags
 *
 * This module provides a simple feature flag system to control
 * the rollout of new features and changes.
 */

// Define the available feature flags
export interface FeatureFlags {
  useSecureApiStorage: boolean;
  // Add more feature flags as needed
}

// Default values for feature flags
const defaultFlags: FeatureFlags = {
  useSecureApiStorage: false, // Start with the old implementation
};

// Get the current feature flags
export function getFeatureFlags(): FeatureFlags {
  // In a real implementation, this might fetch from a server or localStorage
  return { ...defaultFlags };
}

// Check if a specific feature is enabled
export function isFeatureEnabled(feature: keyof FeatureFlags): boolean {
  return getFeatureFlags()[feature];
}

// Enable a specific feature
export function enableFeature(feature: keyof FeatureFlags): void {
  // In a real implementation, this would update the stored flags
  console.log(`Feature ${feature} would be enabled`);
}

// Disable a specific feature
export function disableFeature(feature: keyof FeatureFlags): void {
  // In a real implementation, this would update the stored flags
  console.log(`Feature ${feature} would be disabled`);
}
