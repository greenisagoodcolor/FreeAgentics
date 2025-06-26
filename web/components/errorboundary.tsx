"use client";

import React, { Component, ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle, RefreshCw } from "lucide-react";
import { LLMError } from "@/lib/llm-errors";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface IState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, IState> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): IState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const error = this.state.error;
      const isLLMError = error instanceof LLMError;

      return (
        <Alert variant="destructive" className="m-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>
            {isLLMError
              ? `${error.provider?.toUpperCase()} Error`
              : "Application Error"}
          </AlertTitle>
          <AlertDescription className="mt-2">
            <p>{error?.message || "An unexpected error occurred"}</p>
            {isLLMError && (
              <p className="text-sm text-muted-foreground mt-1">
                Error type: {error.type}
                {error.statusCode && ` (Status: ${error.statusCode})`}
              </p>
            )}
            <Button
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() => this.setState({ hasError: false, error: null })}
            >
              <RefreshCw className="h-3 w-3 mr-1" />
              Try Again
            </Button>
          </AlertDescription>
        </Alert>
      );
    }

    return this.props.children;
  }
}

// Hook-based error boundary for functional components
export function useErrorHandler() {
  return (error: Error) => {
    console.error("Error handled:", error);
    throw error; // This will be caught by the ErrorBoundary
  };
}
