/**
 * Comprehensive error handling utilities for FreeAgentics frontend
 */

export interface AppError {
  id: string;
  type: "api" | "network" | "validation" | "auth" | "system" | "unknown";
  message: string;
  userMessage: string;
  status?: number;
  timestamp: number;
  context?: Record<string, unknown>;
  retryable: boolean;
  originalError?: Error;
}

export interface ErrorReportingConfig {
  enableConsoleLogging: boolean;
  enableRemoteReporting: boolean;
  enableUserFeedback: boolean;
  maxRetries: number;
  retryDelay: number;
}

interface ApiErrorInput {
  status?: number;
  detail?: string;
  message?: string;
}

class ErrorHandler {
  private config: ErrorReportingConfig;
  private errorQueue: AppError[] = [];

  constructor(config: Partial<ErrorReportingConfig> = {}) {
    this.config = {
      enableConsoleLogging: true,
      enableRemoteReporting: process.env.NODE_ENV === "production",
      enableUserFeedback: true,
      maxRetries: 3,
      retryDelay: 1000,
      ...config,
    };
  }

  /**
   * Create a standardized AppError from various error types
   */
  createError(
    error: Error | string,
    type: AppError["type"] = "unknown",
    context?: Record<string, unknown>,
  ): AppError {
    const id = this.generateErrorId();
    const timestamp = Date.now();

    let message: string;
    let originalError: Error | undefined;

    if (typeof error === "string") {
      message = error;
    } else {
      message = error.message;
      originalError = error;
    }

    const userMessage = this.generateUserFriendlyMessage(type, message);
    const retryable = this.isRetryable(type, message);

    return {
      id,
      type,
      message,
      userMessage,
      timestamp,
      context,
      retryable,
      originalError,
    };
  }

  /**
   * Handle API errors with enhanced error information
   */
  handleApiError(error: ApiErrorInput, endpoint?: string, method?: string): AppError {
    let type: AppError["type"] = "api";
    let status: number | undefined;
    let message: string;

    if (error.status) {
      status = error.status;

      // Categorize by HTTP status
      if (status && status >= 400 && status < 500) {
        type = status === 401 || status === 403 ? "auth" : "validation";
      } else if (status && status >= 500) {
        type = "system";
      }
    }

    if (error.detail) {
      message = error.detail;
    } else if (error.message) {
      message = error.message;
    } else {
      message = "Unknown API error occurred";
    }

    const context = {
      endpoint,
      method,
      status,
      timestamp: new Date().toISOString(),
    };

    const appError = this.createError(message, type, context);
    appError.status = status;

    this.reportError(appError);
    return appError;
  }

  /**
   * Handle network errors
   */
  handleNetworkError(error: Error, context?: Record<string, unknown>): AppError {
    const appError = this.createError(error, "network", {
      ...context,
      offline: !navigator.onLine,
      userAgent: navigator.userAgent,
    });

    this.reportError(appError);
    return appError;
  }

  /**
   * Report error through various channels
   */
  reportError(error: AppError): void {
    // Add to error queue
    this.errorQueue.push(error);

    // Keep only last 100 errors
    if (this.errorQueue.length > 100) {
      this.errorQueue = this.errorQueue.slice(-100);
    }

    // Console logging
    if (this.config.enableConsoleLogging) {
      console.error(`[${error.type.toUpperCase()}] ${error.message}`, {
        id: error.id,
        context: error.context,
        originalError: error.originalError,
      });
    }

    // Remote error reporting
    if (this.config.enableRemoteReporting) {
      this.sendToRemoteService(error);
    }
  }

  /**
   * Get user-friendly error message
   */
  private generateUserFriendlyMessage(type: AppError["type"], message: string): string {
    const lowercaseMessage = message.toLowerCase();

    switch (type) {
      case "network":
        if (!navigator.onLine) {
          return "You appear to be offline. Please check your internet connection.";
        }
        return "Network error occurred. Please check your connection and try again.";

      case "auth":
        if (lowercaseMessage.includes("unauthorized") || lowercaseMessage.includes("401")) {
          return "Your session has expired. Please log in again.";
        }
        if (lowercaseMessage.includes("forbidden") || lowercaseMessage.includes("403")) {
          return "You don't have permission to perform this action.";
        }
        return "Authentication error. Please try logging in again.";

      case "validation":
        if (lowercaseMessage.includes("not found") || lowercaseMessage.includes("404")) {
          return "The requested resource was not found.";
        }
        if (lowercaseMessage.includes("bad request") || lowercaseMessage.includes("400")) {
          return "Invalid request. Please check your input and try again.";
        }
        return "Invalid data provided. Please check your input.";

      case "system":
        return "A server error occurred. Our team has been notified. Please try again later.";

      case "api":
        return "An error occurred while communicating with the server. Please try again.";

      default:
        return "An unexpected error occurred. Please try again or contact support if the problem persists.";
    }
  }

  /**
   * Determine if an error is retryable
   */
  private isRetryable(type: AppError["type"], message: string): boolean {
    switch (type) {
      case "network":
        return true;
      case "system":
        return true;
      case "api":
        // Retry on 5xx errors but not 4xx
        return (
          !message.includes("400") &&
          !message.includes("401") &&
          !message.includes("403") &&
          !message.includes("404")
        );
      case "auth":
        return false; // Don't auto-retry auth errors
      case "validation":
        return false; // Don't retry validation errors
      default:
        return false;
    }
  }

  /**
   * Generate unique error ID
   */
  private generateErrorId(): string {
    return `err_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Send error to remote error tracking service
   */
  private async sendToRemoteService(error: AppError): Promise<void> {
    try {
      // In production, this would send to services like Sentry, LogRocket, etc.
      // For now, we'll just log that we would send it
      if (process.env.NODE_ENV === "development") {
        console.info("Would send to remote error service:", {
          id: error.id,
          type: error.type,
          message: error.message,
          context: error.context,
        });
      }

      // Example implementation:
      // await fetch('/api/errors', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(error),
      // });
    } catch (err) {
      console.error("Failed to send error to remote service:", err);
    }
  }

  /**
   * Get recent errors for debugging
   */
  getRecentErrors(limit: number = 10): AppError[] {
    return this.errorQueue.slice(-limit);
  }

  /**
   * Clear error queue
   */
  clearErrors(): void {
    this.errorQueue = [];
  }

  /**
   * Retry function with exponential backoff
   */
  async retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number = this.config.maxRetries,
    baseDelay: number = this.config.retryDelay,
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;

        if (attempt === maxRetries) {
          throw lastError;
        }

        // Exponential backoff with jitter
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    throw lastError!;
  }
}

// Singleton instance
export const errorHandler = new ErrorHandler();

/**
 * React hook for error handling
 */
export function useErrorHandler() {
  const handleError = (
    error: Error | string,
    type?: AppError["type"],
    context?: Record<string, unknown>,
  ): AppError => {
    return errorHandler.createError(error, type, context);
  };

  const handleApiError = (error: ApiErrorInput, endpoint?: string, method?: string): AppError => {
    return errorHandler.handleApiError(error, endpoint, method);
  };

  const handleNetworkError = (error: Error, context?: Record<string, unknown>): AppError => {
    return errorHandler.handleNetworkError(error, context);
  };

  const retryOperation = async <T>(
    fn: () => Promise<T>,
    maxRetries?: number,
    baseDelay?: number,
  ): Promise<T> => {
    return errorHandler.retryWithBackoff(fn, maxRetries, baseDelay);
  };

  return {
    handleError,
    handleApiError,
    handleNetworkError,
    retryOperation,
    getRecentErrors: () => errorHandler.getRecentErrors(),
    clearErrors: () => errorHandler.clearErrors(),
  };
}
