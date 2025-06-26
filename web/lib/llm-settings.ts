// LLM Settings types and defaults
// Separated from llm-service.ts due to "use server" restrictions

export interface LLMSettings {
  provider: "openai" | "anthropic" | "openrouter"
  model: string
  temperature: number
  maxTokens: number
  topP: number
  frequencyPenalty: number
  presencePenalty: number
  systemFingerprint?: string | boolean
  apiKey?: string
  apiKeySessionId?: string
  hasServerRef?: boolean
  maxAutonomousMessages?: number
  conversationCooldown?: number
}

export const defaultSettings: LLMSettings = {
  provider: "openai",
  model: "gpt-4o",
  temperature: 0.7,
  maxTokens: 1024,
  topP: 0.9,
  frequencyPenalty: 0,
  presencePenalty: 0,
  systemFingerprint: false,
  hasServerRef: false,
  maxAutonomousMessages: 4,
  conversationCooldown: 5000,
}

export const clientDefaultSettings: Partial<LLMSettings> = {
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

// Provider configurations
export const providerModels = {
  openai: [
    { id: "gpt-4o", name: "GPT-4o" },
    { id: "gpt-4o-mini", name: "GPT-4o Mini" },
    { id: "gpt-4-turbo", name: "GPT-4 Turbo" },
    { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo" },
  ],
  anthropic: [
    { id: "claude-3-5-sonnet-20241022", name: "Claude 3.5 Sonnet" },
    { id: "claude-3-opus-20240229", name: "Claude 3 Opus" },
    { id: "claude-3-haiku-20240307", name: "Claude 3 Haiku" },
  ],
  openrouter: [
    { id: "anthropic/claude-3-5-sonnet", name: "Claude 3.5 Sonnet (OpenRouter)" },
    { id: "openai/gpt-4o", name: "GPT-4o (OpenRouter)" },
  ],
}
