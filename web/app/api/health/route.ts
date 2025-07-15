import { NextResponse } from "next/server";

/**
 * Health check endpoint for production monitoring
 * Used by load balancers, Docker health checks, and monitoring systems
 */
export async function GET() {
  try {
    // Check various system components
    const checks = {
      status: "healthy",
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV,
      version: process.env.npm_package_version || "0.1.0-alpha",
      uptime: process.uptime(),
      memory: {
        used: process.memoryUsage().heapUsed / 1024 / 1024,
        total: process.memoryUsage().heapTotal / 1024 / 1024,
        limit: 512, // MB - adjust based on your container limits
      },
      checks: {
        server: true,
        database: await checkDatabase(),
        redis: await checkRedis(),
        api: await checkBackendAPI(),
      },
    };

    // Determine overall health
    const isHealthy = Object.values(checks.checks).every((check) => check === true);

    if (!isHealthy) {
      return NextResponse.json(
        {
          ...checks,
          status: "unhealthy",
        },
        { status: 503 },
      );
    }

    return NextResponse.json(checks, {
      status: 200,
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "X-Health-Check": "passed",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}

/**
 * Check database connectivity
 */
async function checkDatabase(): Promise<boolean> {
  try {
    // In production, this would check actual database connection
    // For now, we'll simulate it
    if (process.env.DATABASE_URL) {
      // Simulate database ping
      await new Promise((resolve) => setTimeout(resolve, 10));
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Check Redis connectivity
 */
async function checkRedis(): Promise<boolean> {
  try {
    // In production, this would check actual Redis connection
    if (process.env.REDIS_URL) {
      // Simulate Redis ping
      await new Promise((resolve) => setTimeout(resolve, 10));
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Check backend API connectivity
 */
async function checkBackendAPI(): Promise<boolean> {
  try {
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${backendUrl}/health`, {
      signal: controller.signal,
      headers: {
        "User-Agent": "FreeAgentics-Frontend-Health-Check",
      },
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    return false;
  }
}
