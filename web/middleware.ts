import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Middleware for security headers and request processing
 * Runs on every request before it reaches the route handlers
 */
export function middleware(request: NextRequest) {
  // Clone the request headers
  const requestHeaders = new Headers(request.headers);

  // Add request ID for tracing
  const requestId = crypto.randomUUID();
  requestHeaders.set("x-request-id", requestId);

  // Create response with cloned headers
  const response = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });

  // Security headers that aren't handled by next.config.js
  const securityHeaders = {
    "X-Request-ID": requestId,
    "X-Powered-By": "FreeAgentics",
    "X-Download-Options": "noopen",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Origin-Agent-Cluster": "?1",
  };

  // Apply security headers
  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  // Log request for monitoring (in production, send to logging service)
  if (process.env.NODE_ENV === "production") {
    console.log(
      JSON.stringify({
        timestamp: new Date().toISOString(),
        requestId,
        method: request.method,
        url: request.url,
        userAgent: request.headers.get("user-agent"),
        referer: request.headers.get("referer"),
        ip: request.headers.get("x-forwarded-for") || request.headers.get("x-real-ip"),
      }),
    );
  }

  return response;
}

// Configure which paths the middleware runs on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    "/((?!_next/static|_next/image|favicon.ico|.*\\..*|_next).*)",
  ],
};
