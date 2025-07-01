export class LLMError extends Error {
  public code?: string;
  public provider?: string;
  public type?: string;
  public statusCode?: number;

  constructor(message: string, code?: string) {
    super(message);
    this.name = "LLMError";
    this.code = code;
  }
}

export class RateLimitError extends LLMError {
  constructor(message: string) {
    super(message, "RATE_LIMIT");
    this.name = "RateLimitError";
  }
}

export class AuthenticationError extends LLMError {
  constructor(message: string) {
    super(message, "AUTH_ERROR");
    this.name = "AuthenticationError";
  }
}

export class ApiKeyError extends LLMError {
  constructor(message: string) {
    super(message, "API_KEY_ERROR");
    this.name = "ApiKeyError";
  }
}

export class TimeoutError extends LLMError {
  constructor(message: string) {
    super(message, "TIMEOUT_ERROR");
    this.name = "TimeoutError";
  }
}

export class NetworkError extends LLMError {
  constructor(message: string) {
    super(message, "NETWORK_ERROR");
    this.name = "NetworkError";
  }
}

export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string = "Operation timed out",
): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new TimeoutError(timeoutMessage)), timeoutMs),
  );

  return Promise.race([promise, timeoutPromise]);
}
