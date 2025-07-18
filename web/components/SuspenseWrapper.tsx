"use client";

import React, { Suspense, ComponentType, ReactNode } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { LoadingState } from "./LoadingState";
import { Skeleton, SkeletonContainer } from "./Skeleton";

interface SuspenseWrapperProps {
  children: ReactNode;
  fallback?: ReactNode;
  errorFallback?: ReactNode;
  loadingMessage?: string;
  showErrorDetails?: boolean;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

/**
 * Production-ready wrapper that combines Suspense and Error Boundary
 * Provides consistent loading and error states across the application
 */
export function SuspenseWrapper({
  children,
  fallback,
  errorFallback,
  loadingMessage = "Loading...",
  showErrorDetails: _showErrorDetails = false,
  onError: _onError,
}: SuspenseWrapperProps) {
  const defaultFallback = (
    <div className="flex items-center justify-center min-h-[200px]">
      <LoadingState message={loadingMessage} size="medium" />
    </div>
  );

  return (
    <ErrorBoundary fallback={errorFallback} onReset={() => window.location.reload()}>
      <Suspense fallback={fallback || defaultFallback}>{children}</Suspense>
    </ErrorBoundary>
  );
}

/**
 * Higher-order component for adding suspense and error boundaries
 */
export function withSuspense<P extends object>(
  Component: ComponentType<P>,
  options?: {
    fallback?: ReactNode;
    errorFallback?: ReactNode;
    loadingMessage?: string;
  },
) {
  return function WithSuspenseComponent(props: P) {
    return (
      <SuspenseWrapper {...options}>
        <Component {...props} />
      </SuspenseWrapper>
    );
  };
}

/**
 * Lazy load component with built-in loading and error states
 */
export function lazyWithPreload<T extends ComponentType<Record<string, unknown>>>(
  factory: () => Promise<{ default: T }>,
  options?: {
    fallback?: ReactNode;
    errorFallback?: ReactNode;
    preloadDelay?: number;
  },
) {
  let loadPromise: Promise<{ default: T }> | null = null;

  const load = () => {
    if (!loadPromise) {
      loadPromise = factory();
    }
    return loadPromise;
  };

  const LazyComponent = React.lazy(load);

  const WrappedComponent = (props: React.ComponentProps<T>) => (
    <SuspenseWrapper fallback={options?.fallback} errorFallback={options?.errorFallback}>
      <LazyComponent {...(props as any)} />
    </SuspenseWrapper>
  );

  // Add preload method
  (WrappedComponent as typeof WrappedComponent & { preload: () => void }).preload = () => {
    if (options?.preloadDelay) {
      setTimeout(load, options.preloadDelay);
    } else {
      load();
    }
  };

  return WrappedComponent;
}

/**
 * Common loading states for different scenarios
 */
export const LoadingStates = {
  Page: () => (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <LoadingState size="large" message="Loading page..." />
    </div>
  ),

  Section: ({ message = "Loading section..." }: { message?: string }) => (
    <div className="py-8">
      <LoadingState size="medium" message={message} />
    </div>
  ),

  Inline: () => <LoadingState size="small" />,

  List: ({ count = 3 }: { count?: number }) => (
    <SkeletonContainer>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="bg-white rounded-lg shadow p-4">
          <Skeleton variant="text" width="60%" className="mb-2" />
          <Skeleton variant="text" width="40%" />
        </div>
      ))}
    </SkeletonContainer>
  ),

  Table: ({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) => (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <table className="min-w-full">
        <thead className="bg-gray-50 border-b">
          <tr>
            {Array.from({ length: cols }).map((_, i) => (
              <th key={i} className="p-4">
                <Skeleton variant="text" width="80%" />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i} className="border-b">
              {Array.from({ length: cols }).map((_, j) => (
                <td key={j} className="p-4">
                  <Skeleton variant="text" width={`${60 + Math.random() * 40}%`} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ),

  Card: () => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <Skeleton variant="rectangular" height={200} className="mb-4" />
      <Skeleton variant="text" width="60%" className="mb-2" />
      <Skeleton variant="text" width="80%" className="mb-2" />
      <Skeleton variant="text" width="40%" />
    </div>
  ),
};

/**
 * Hook for managing loading states
 */
export function useLoadingState(initialState = false) {
  const [isLoading, setIsLoading] = React.useState(initialState);
  const [error, setError] = React.useState<Error | null>(null);

  const execute = React.useCallback(
    async <T,>(asyncFunction: () => Promise<T>): Promise<T | undefined> => {
      try {
        setIsLoading(true);
        setError(null);
        const result = await asyncFunction();
        return result;
      } catch (err) {
        setError(err as Error);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  const reset = React.useCallback(() => {
    setIsLoading(false);
    setError(null);
  }, []);

  return {
    isLoading,
    error,
    execute,
    reset,
    setIsLoading,
    setError,
  };
}
