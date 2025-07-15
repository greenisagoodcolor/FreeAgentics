"use client";

import { useEffect } from "react";
import { errorHandler } from "@/lib/error-handling";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to error reporting service
    const appError = errorHandler.createError(error, "system", {
      page: "global-error",
      digest: error.digest,
      timestamp: new Date().toISOString(),
      critical: true,
    });
    errorHandler.reportError(appError);
  }, [error]);

  // Minimal HTML/CSS since this runs when the entire app crashes
  return (
    <html lang="en">
      <head>
        <title>Critical Error - FreeAgentics</title>
      </head>
      <body style={{ margin: 0, padding: 0, fontFamily: "system-ui, sans-serif" }}>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#f9fafb",
            padding: "1rem",
          }}
        >
          <div
            style={{
              maxWidth: "600px",
              width: "100%",
              backgroundColor: "white",
              borderRadius: "0.5rem",
              boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
              padding: "2rem",
              textAlign: "center",
            }}
          >
            <div
              style={{
                width: "64px",
                height: "64px",
                margin: "0 auto 1rem",
                backgroundColor: "#fee2e2",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#dc2626"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </div>

            <h1
              style={{
                fontSize: "1.5rem",
                fontWeight: "bold",
                color: "#111827",
                marginBottom: "0.5rem",
              }}
            >
              Critical Application Error
            </h1>

            <p
              style={{
                fontSize: "1rem",
                color: "#6b7280",
                marginBottom: "1.5rem",
                lineHeight: "1.5",
              }}
            >
              We&apos;re experiencing a critical error. Our team has been automatically notified and is
              working to resolve this issue.
            </p>

            {error.digest && (
              <div
                style={{
                  backgroundColor: "#f3f4f6",
                  padding: "0.75rem",
                  borderRadius: "0.375rem",
                  marginBottom: "1.5rem",
                }}
              >
                <p style={{ fontSize: "0.875rem", color: "#6b7280", margin: 0 }}>
                  Error Reference: <code style={{ fontFamily: "monospace" }}>{error.digest}</code>
                </p>
              </div>
            )}

            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "center" }}>
              <button
                onClick={reset}
                style={{
                  padding: "0.5rem 1rem",
                  backgroundColor: "#3b82f6",
                  color: "white",
                  border: "none",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem",
                  fontWeight: "500",
                  cursor: "pointer",
                }}
                onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#2563eb")}
                onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#3b82f6")}
              >
                Try Again
              </button>
              <button
                onClick={() => (window.location.href = "/")}
                style={{
                  padding: "0.5rem 1rem",
                  backgroundColor: "white",
                  color: "#374151",
                  border: "1px solid #d1d5db",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem",
                  fontWeight: "500",
                  cursor: "pointer",
                }}
                onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#f9fafb")}
                onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "white")}
              >
                Go to Home
              </button>
            </div>

            <div
              style={{
                marginTop: "2rem",
                paddingTop: "1rem",
                borderTop: "1px solid #e5e7eb",
                fontSize: "0.875rem",
                color: "#6b7280",
              }}
            >
              <p style={{ margin: "0 0 0.5rem 0" }}>
                If this problem continues, please contact support:
              </p>
              <a
                href="mailto:support@freeagentics.com"
                style={{ color: "#3b82f6", textDecoration: "none" }}
              >
                support@freeagentics.com
              </a>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
