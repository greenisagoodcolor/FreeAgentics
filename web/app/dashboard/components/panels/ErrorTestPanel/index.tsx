"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Bug, Zap } from "lucide-react";

// Component designed to test error boundaries
export default function ErrorTestPanel() {
  const [shouldError, setShouldError] = useState(false);
  const [errorType, setErrorType] = useState<"render" | "async" | "redux">(
    "render",
  );

  if (shouldError && errorType === "render") {
    // This will be caught by ErrorBoundary
    throw new Error("Deliberate render error for testing error boundary");
  }

  const triggerAsyncError = async () => {
    // This won't be caught by ErrorBoundary (needs different handling)
    throw new Error("Deliberate async error for testing");
  };

  const triggerReduxError = () => {
    // This could corrupt Redux state
    try {
      // Deliberately try to dispatch invalid action
      const invalidAction = {
        type: "INVALID_ACTION",
        payload: { malformed: true },
      };
      // This would need to be imported from store to actually test
      console.error("Would dispatch invalid action:", invalidAction);
    } catch (error) {
      console.error("Redux error:", error);
    }
  };

  return (
    <div className="p-4 space-y-4 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-primary)]">
      <div className="flex items-center gap-2">
        <Bug className="w-5 h-5 text-yellow-500" />
        <h3 className="font-semibold text-[var(--text-primary)]">
          Error Boundary Testing
        </h3>
      </div>

      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <label className="text-sm text-[var(--text-secondary)]">
            Error Type:
          </label>
          <select
            value={errorType}
            onChange={(e) => setErrorType(e.target.value as any)}
            className="text-sm px-2 py-1 rounded border bg-[var(--bg-primary)] text-[var(--text-primary)] border-[var(--border-primary)]"
          >
            <option value="render">Render Error</option>
            <option value="async">Async Error</option>
            <option value="redux">Redux Error</option>
          </select>
        </div>

        <div className="flex gap-2">
          <Button
            size="sm"
            variant="destructive"
            onClick={() => {
              if (errorType === "render") {
                setShouldError(true);
              } else if (errorType === "async") {
                triggerAsyncError().catch(console.error);
              } else {
                triggerReduxError();
              }
            }}
            className="flex items-center gap-2"
          >
            <AlertTriangle className="w-4 h-4" />
            Trigger {errorType} Error
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={() => setShouldError(false)}
          >
            Reset
          </Button>
        </div>
      </div>

      <div className="text-xs text-[var(--text-tertiary)] bg-[var(--bg-tertiary)] p-2 rounded">
        <p>
          <strong>Testing:</strong>
        </p>
        <p>• Render errors should be caught by ErrorBoundary</p>
        <p>• Async errors need separate error handling</p>
        <p>• Redux errors should be validated at action level</p>
      </div>
    </div>
  );
}
