// Define the LLM settings interface
export interface LLMSettings {
  provider: "openai" | "openrouter"
  model: string
  temperature: number
  maxTokens: number
  topP: number
  frequencyPenalty: number
  presencePenalty: number
  systemFingerprint: boolean
  apiKey?: string // Only used temporarily during API calls, never stored
  apiKeySessionId?: string // Used to retrieve the API key securely
  maxAutonomousMessages?: number
  conversationCooldown?: number
}

// Default settings for the client
export const clientDefaultSettings: LLMSettings = {
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
}
