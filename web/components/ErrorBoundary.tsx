"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { errorHandler, AppError } from "@/lib/error-handling";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { AlertTriangle, RefreshCw, Home, ChevronDown, ChevronUp } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  appError: AppError | null;
  showDetails: boolean;
  retryCount: number;
}

/**
 * Production-ready error boundary with user-friendly error display
 * Implements error recovery, reporting, and graceful degradation
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      appError: null,
      showDetails: false,
      retryCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    const appError = errorHandler.createError(error, "system", {
      component: "ErrorBoundary",
      timestamp: new Date().toISOString(),
    });

    return {
      hasError: true,
      error,
      errorInfo: null,
      appError,
      showDetails: false,
      retryCount: 0,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to error reporting service
    const appError = errorHandler.createError(error, "system", {
      component: "ErrorBoundary",
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
    });

    errorHandler.reportError(appError);

    this.setState({
      errorInfo,
      appError,
    });
  }

  handleReset = () => {
    const { onReset } = this.props;

    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      appError: null,
      showDetails: false,
      retryCount: this.state.retryCount + 1,
    });

    if (onReset) {
      onReset();
    }
  };

  handleGoHome = () => {
    window.location.href = "/";
  };

  toggleDetails = () => {
    this.setState((prevState) => ({
      showDetails: !prevState.showDetails,
    }));
  };

  render() {
    const { hasError, error, errorInfo, appError, showDetails, retryCount } = this.state;
    const { children, fallback } = this.props;

    if (hasError && error) {
      // Custom fallback provided
      if (fallback) {
        return <>{fallback}</>;
      }

      // Default error UI
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <Card className="max-w-2xl w-full shadow-lg">
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                <AlertTriangle className="w-8 h-8 text-red-600" />
              </div>
              <CardTitle className="text-2xl font-bold text-gray-900">
                Oops! Something went wrong
              </CardTitle>
              <CardDescription className="text-lg text-gray-600 mt-2">
                {appError?.userMessage ||
                  "An unexpected error occurred. Our team has been notified."}
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
              {/* Action buttons */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button
                  onClick={this.handleReset}
                  variant="default"
                  className="flex items-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                  {retryCount > 0 && (
                    <span className="text-xs opacity-70">
                      ({retryCount} {retryCount === 1 ? "retry" : "retries"})
                    </span>
                  )}
                </Button>
                <Button
                  onClick={this.handleGoHome}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <Home className="w-4 h-4" />
                  Go to Home
                </Button>
              </div>

              {/* Error ID for support */}
              {appError?.id && (
                <div className="text-center text-sm text-gray-500">
                  Error ID:{" "}
                  <code className="font-mono bg-gray-100 px-2 py-1 rounded">{appError.id}</code>
                </div>
              )}

              {/* Expandable error details for development */}
              {process.env.NODE_ENV === "development" && (
                <div className="mt-6 border-t pt-4">
                  <button
                    onClick={this.toggleDetails}
                    className="w-full flex items-center justify-between text-sm text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <span className="font-medium">Technical Details</span>
                    {showDetails ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>

                  {showDetails && (
                    <div className="mt-4 space-y-4">
                      {/* Error message */}
                      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <h4 className="text-sm font-semibold text-red-900 mb-1">Error Message</h4>
                        <p className="text-sm text-red-700 font-mono">{error.message}</p>
                      </div>

                      {/* Stack trace */}
                      {error.stack && (
                        <div className="bg-gray-100 rounded-lg p-4 overflow-x-auto">
                          <h4 className="text-sm font-semibold text-gray-900 mb-2">Stack Trace</h4>
                          <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                            {error.stack}
                          </pre>
                        </div>
                      )}

                      {/* Component stack */}
                      {errorInfo?.componentStack && (
                        <div className="bg-gray-100 rounded-lg p-4 overflow-x-auto">
                          <h4 className="text-sm font-semibold text-gray-900 mb-2">
                            Component Stack
                          </h4>
                          <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                            {errorInfo.componentStack}
                          </pre>
                        </div>
                      )}

                      {/* Additional context */}
                      {appError?.context && (
                        <div className="bg-gray-100 rounded-lg p-4">
                          <h4 className="text-sm font-semibold text-gray-900 mb-2">Context</h4>
                          <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                            {JSON.stringify(appError.context, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      );
    }

    return children;
  }
}

/**
 * Hook to reset error boundaries programmatically
 */
export function useErrorBoundary() {
  const [resetKey, setResetKey] = React.useState(0);

  const resetErrorBoundary = React.useCallback(() => {
    setResetKey((prev) => prev + 1);
  }, []);

  return { resetKey, resetErrorBoundary };
}
