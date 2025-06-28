"use client";

import React from "react";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";
import { motion } from "framer-motion";

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
  errorId?: string;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; retry: () => void }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  level?: "page" | "component" | "widget";
}

export class DashboardErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  private retryTimeoutId: number | null = null;

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
      errorId: `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const { onError } = this.props;
    
    this.setState({ errorInfo });
    
    // Log error details
    console.error("Dashboard Error Boundary caught an error:", {
      error,
      errorInfo,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });

    // Call custom error handler
    onError?.(error, errorInfo);

    // Report to analytics if available
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'exception', {
        description: error.message,
        fatal: false,
        custom_map: {
          component_stack: errorInfo.componentStack
        }
      });
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  handleAutoRetry = () => {
    // Auto-retry after 5 seconds for component-level errors
    if (this.props.level === "component" || this.props.level === "widget") {
      this.retryTimeoutId = window.setTimeout(() => {
        this.handleRetry();
      }, 5000);
    }
  };

  render() {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback: CustomFallback, level = "component" } = this.props;

    if (hasError && error) {
      // Use custom fallback if provided
      if (CustomFallback) {
        return <CustomFallback error={error} retry={this.handleRetry} />;
      }

      // Default error UI based on level
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.2 }}
          className={`error-boundary-container ${level}`}
        >
          {level === "page" ? (
            <PageErrorFallback error={error} onRetry={this.handleRetry} />
          ) : level === "component" ? (
            <ComponentErrorFallback 
              error={error} 
              onRetry={this.handleRetry}
              onAutoRetry={this.handleAutoRetry}
            />
          ) : (
            <WidgetErrorFallback error={error} onRetry={this.handleRetry} />
          )}
        </motion.div>
      );
    }

    return children;
  }
}

// Page-level error fallback
const PageErrorFallback: React.FC<{
  error: Error;
  onRetry: () => void;
}> = ({ error, onRetry }) => (
  <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center p-4">
    <div className="max-w-md w-full text-center space-y-6">
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", duration: 0.5 }}
        className="w-16 h-16 mx-auto text-red-400"
      >
        <AlertTriangle className="w-full h-full" />
      </motion.div>
      
      <div className="space-y-2">
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">
          Something went wrong
        </h1>
        <p className="text-[var(--text-secondary)]">
          The dashboard encountered an unexpected error. Please try refreshing the page.
        </p>
      </div>

      <div className="space-y-3">
        <button
          onClick={onRetry}
          className="button button-primary w-full"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Try Again
        </button>
        
        <button
          onClick={() => window.location.href = "/"}
          className="button button-secondary w-full"
        >
          <Home className="w-4 h-4 mr-2" />
          Go Home
        </button>
      </div>

      <details className="text-left text-xs text-[var(--text-tertiary)]">
        <summary className="cursor-pointer hover:text-[var(--text-secondary)]">
          Error Details
        </summary>
        <pre className="mt-2 p-3 bg-[var(--bg-secondary)] rounded border overflow-auto">
          {error.message}
        </pre>
      </details>
    </div>
  </div>
);

// Component-level error fallback
const ComponentErrorFallback: React.FC<{
  error: Error;
  onRetry: () => void;
  onAutoRetry: () => void;
}> = ({ error, onRetry, onAutoRetry }) => {
  React.useEffect(() => {
    onAutoRetry();
  }, [onAutoRetry]);

  return (
    <div className="bg-red-900/10 border border-red-500/20 rounded-lg p-6 text-center space-y-4">
      <div className="w-8 h-8 mx-auto text-red-400">
        <AlertTriangle className="w-full h-full" />
      </div>
      
      <div className="space-y-1">
        <h3 className="font-semibold text-red-400">Component Error</h3>
        <p className="text-sm text-[var(--text-secondary)]">
          This component failed to load properly.
        </p>
      </div>

      <div className="space-y-2">
        <button
          onClick={onRetry}
          className="button button-sm button-secondary"
        >
          <RefreshCw className="w-3 h-3 mr-1" />
          Retry
        </button>
        <div className="text-xs text-[var(--text-tertiary)]">
          Auto-retry in 5 seconds...
        </div>
      </div>
    </div>
  );
};

// Widget-level error fallback
const WidgetErrorFallback: React.FC<{
  error: Error;
  onRetry: () => void;
}> = ({ error, onRetry }) => (
  <div className="bg-red-900/5 border border-red-500/10 rounded p-4 text-center space-y-2">
    <div className="w-5 h-5 mx-auto text-red-400">
      <AlertTriangle className="w-full h-full" />
    </div>
    <div className="text-xs text-red-400">Widget Error</div>
    <button
      onClick={onRetry}
      className="text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] underline"
    >
      Retry
    </button>
  </div>
);

// Higher-order component for easy wrapping
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Partial<ErrorBoundaryProps>
) => {
  const WrappedComponent = (props: P) => (
    <DashboardErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </DashboardErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
};

// Utility for manual error reporting
export const reportError = (error: Error, context?: string) => {
  console.error(`Manual error report${context ? ` (${context})` : ''}:`, error);
  
  if (typeof window !== 'undefined' && (window as any).gtag) {
    (window as any).gtag('event', 'exception', {
      description: error.message,
      fatal: false,
      custom_map: { context }
    });
  }
};

export default DashboardErrorBoundary;
