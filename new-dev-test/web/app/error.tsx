"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, RefreshCw, Home, FileText } from "lucide-react";
import { errorHandler } from "@/lib/error-handling";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to error reporting service
    const appError = errorHandler.createError(error, "system", {
      page: "app/error",
      digest: error.digest,
      timestamp: new Date().toISOString(),
    });
    errorHandler.reportError(appError);
  }, [error]);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <Card className="max-w-2xl w-full shadow-lg">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
          <CardTitle className="text-2xl font-bold text-gray-900">Something went wrong!</CardTitle>
          <CardDescription className="text-lg text-gray-600 mt-2">
            An error occurred while processing your request. Don&apos;t worry, we&apos;re on it!
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Error details in production */}
          {error.digest && (
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <p className="text-sm text-gray-600 mb-1">Error Reference</p>
              <code className="font-mono text-xs bg-white px-3 py-1 rounded border border-gray-200">
                {error.digest}
              </code>
            </div>
          )}

          {/* Development error details */}
          {process.env.NODE_ENV === "development" && (
            <details className="bg-red-50 border border-red-200 rounded-lg p-4">
              <summary className="cursor-pointer font-medium text-red-900 mb-2">
                Error Details (Development Only)
              </summary>
              <div className="mt-2 space-y-2">
                <div>
                  <p className="text-sm font-semibold text-red-800">Message:</p>
                  <p className="text-sm text-red-700 font-mono">{error.message}</p>
                </div>
                {error.stack && (
                  <div>
                    <p className="text-sm font-semibold text-red-800">Stack:</p>
                    <pre className="text-xs text-red-700 font-mono overflow-x-auto whitespace-pre-wrap">
                      {error.stack}
                    </pre>
                  </div>
                )}
              </div>
            </details>
          )}

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button onClick={reset} variant="default" className="flex items-center gap-2">
              <RefreshCw className="w-4 h-4" />
              Try Again
            </Button>
            <Button
              onClick={() => (window.location.href = "/")}
              variant="outline"
              className="flex items-center gap-2"
            >
              <Home className="w-4 h-4" />
              Go to Home
            </Button>
          </div>

          {/* Support information */}
          <div className="border-t pt-4 text-center text-sm text-gray-500">
            <p>If this problem persists, please contact our support team.</p>
            <div className="mt-2 flex items-center justify-center gap-4">
              <a
                href="/support"
                className="text-blue-600 hover:text-blue-800 flex items-center gap-1"
              >
                <FileText className="w-3 h-3" />
                Support Center
              </a>
              <span className="text-gray-300">|</span>
              <a
                href="mailto:support@freeagentics.com"
                className="text-blue-600 hover:text-blue-800"
              >
                support@freeagentics.com
              </a>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
