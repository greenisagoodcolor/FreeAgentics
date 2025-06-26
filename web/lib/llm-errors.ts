// Custom error types for better error handling in LLM services
export class LLMError extends Error {
  constructor(
    message: string,
    public readonly type: "api_key_missing" | "api_error" | "timeout" | "network_error" | "unknown",
    public readonly provider?: string,
    public readonly statusCode?: number,
  ) {
    super(message)
    this.name = "LLMError"
  }
}

export class ApiKeyError extends LLMError {
  constructor(provider: string) {
    super(`API key is required for ${provider} provider`, "api_key_missing", provider)
    this.name = "ApiKeyError"
  }
}

export class TimeoutError extends LLMError {
  constructor(provider: string, timeoutMs: number) {
    super(`${provider} API request timed out after ${timeoutMs / 1000} seconds`, "timeout", provider)
    this.name = "TimeoutError"
  }
}

export class NetworkError extends LLMError {
  constructor(provider: string, statusCode: number, message: string) {
    super(`${provider} API error: ${statusCode} - ${message}`, "network_error", provider, statusCode)
    this.name = "NetworkError"
  }
}

// Add this utility function for handling timeouts in API calls
export async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, provider: string): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new TimeoutError(provider, timeoutMs)), timeoutMs)
  })

  return Promise.race([promise, timeoutPromise])
}
