import { type NextRequest, NextResponse } from "next/server"
import { getApiKey } from "@/lib/api-key-storage"

/**
 * POST /api/llm/health-check
 * Performs health checks on LLM providers
 *
 * Request body:
 * {
 *   providerId?: string,     // Check specific provider
 *   providerType?: string,   // Provider type for targeted check
 *   providerIds?: string[],  // Check multiple providers
 *   includeDetails?: boolean // Include detailed response metrics
 * }
 *
 * Response:
 * {
 *   success: boolean,
 *   results: Record<string, HealthCheckResult>
 * }
 */
export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const body = await request.json()
    const { 
      providerId, 
      providerType, 
      providerIds, 
      includeDetails = false 
    } = body

    let targetProviders: string[] = []

    if (providerId) {
      targetProviders = [providerId]
    } else if (providerIds && Array.isArray(providerIds)) {
      targetProviders = providerIds
    } else {
      // Get all configured providers if none specified
      targetProviders = await getAllConfiguredProviders()
    }

    if (targetProviders.length === 0) {
      return NextResponse.json({
        success: false,
        error: "No providers specified for health check"
      }, { status: 400 })
    }

    // Perform health checks for all target providers
    const healthCheckPromises = targetProviders.map(id => 
      performProviderHealthCheck(id, includeDetails)
    )

    const healthCheckResults = await Promise.allSettled(healthCheckPromises)

    // Process results
    const results: Record<string, any> = {}
    
    healthCheckResults.forEach((result, index) => {
      const providerId = targetProviders[index]
      
      if (result.status === "fulfilled") {
        results[providerId] = result.value
      } else {
        results[providerId] = {
          isHealthy: false,
          status: "error",
          error: result.reason?.message || "Health check failed",
          responseTimeMs: 0,
          timestamp: new Date().toISOString()
        }
      }
    })

    const totalDuration = Date.now() - startTime

    return NextResponse.json({
      success: true,
      results,
      summary: {
        totalProviders: targetProviders.length,
        healthyProviders: Object.values(results).filter((r: any) => r.isHealthy).length,
        totalDurationMs: totalDuration
      }
    })

  } catch (error) {
    console.error("[HEALTH-CHECK] Error performing health checks:", error)
    return NextResponse.json({
      success: false,
      error: "Health check failed",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

/**
 * GET /api/llm/health-check
 * Get current health status of all providers
 */
export async function GET(request: NextRequest) {
  try {
    // In a real implementation, this would fetch from database/cache
    const healthStatus = {
      "openai-primary": {
        isHealthy: true,
        status: "healthy",
        responseTimeMs: 245,
        uptime: 99.8,
        lastCheck: new Date().toISOString(),
        consecutiveFailures: 0,
        errorCount: 2
      },
      "anthropic-secondary": {
        isHealthy: true,
        status: "healthy", 
        responseTimeMs: 180,
        uptime: 99.5,
        lastCheck: new Date().toISOString(),
        consecutiveFailures: 0,
        errorCount: 1
      }
    }

    return NextResponse.json({
      success: true,
      providers: healthStatus,
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error("[HEALTH-CHECK] Error fetching health status:", error)
    return NextResponse.json({
      success: false,
      error: "Failed to fetch health status"
    }, { status: 500 })
  }
}

/**
 * Perform health check for a specific provider
 */
async function performProviderHealthCheck(
  providerId: string, 
  includeDetails: boolean = false
): Promise<any> {
  const startTime = Date.now()

  try {
    // Get provider configuration (in real implementation, from database)
    const providerConfig = await getProviderConfig(providerId)
    
    if (!providerConfig) {
      throw new Error(`Provider ${providerId} not found`)
    }

    // Get stored API credentials
    const apiKey = await getApiKey(providerConfig.type, providerConfig.sessionId)
    
    if (!apiKey) {
      throw new Error(`No API key found for provider ${providerId}`)
    }

    // Perform provider-specific health check
    const healthResult = await checkProviderHealth(
      providerConfig.type,
      apiKey,
      providerConfig,
      includeDetails
    )

    const responseTime = Date.now() - startTime

    return {
      isHealthy: healthResult.isHealthy,
      status: healthResult.status,
      responseTimeMs: responseTime,
      timestamp: new Date().toISOString(),
      ...(includeDetails && {
        details: healthResult.details,
        rateLimits: healthResult.rateLimits,
        availableModels: healthResult.availableModels
      })
    }

  } catch (error) {
    const responseTime = Date.now() - startTime
    
    return {
      isHealthy: false,
      status: "error",
      error: error instanceof Error ? error.message : "Unknown error",
      responseTimeMs: responseTime,
      timestamp: new Date().toISOString()
    }
  }
}

/**
 * Check health for specific provider type
 */
async function checkProviderHealth(
  providerType: string,
  apiKey: string,
  config: any,
  includeDetails: boolean
): Promise<any> {
  switch (providerType) {
    case "openai":
      return await checkOpenAIHealth(apiKey, config, includeDetails)
    case "anthropic":
      return await checkAnthropicHealth(apiKey, config, includeDetails)
    case "openrouter":
      return await checkOpenRouterHealth(apiKey, config, includeDetails)
    case "azure_openai":
      return await checkAzureOpenAIHealth(apiKey, config, includeDetails)
    default:
      throw new Error(`Health check not implemented for provider type: ${providerType}`)
  }
}

/**
 * OpenAI health check
 */
async function checkOpenAIHealth(apiKey: string, config: any, includeDetails: boolean) {
  try {
    // Simple models endpoint check
    const response = await fetch("https://api.openai.com/v1/models", {
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        ...(config.organizationId && { "OpenAI-Organization": config.organizationId })
      },
      signal: AbortSignal.timeout(10000) // 10 second timeout
    })

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    const rateLimits = extractRateLimits(response.headers)

    return {
      isHealthy: true,
      status: "healthy",
      details: includeDetails ? {
        modelsCount: data.data?.length || 0,
        apiVersion: response.headers.get("openai-version"),
        organization: response.headers.get("openai-organization")
      } : undefined,
      rateLimits: includeDetails ? rateLimits : undefined,
      availableModels: includeDetails ? data.data?.slice(0, 5).map((m: any) => m.id) : undefined
    }

  } catch (error) {
    return {
      isHealthy: false,
      status: "unhealthy",
      error: error instanceof Error ? error.message : "Unknown error"
    }
  }
}

/**
 * Anthropic health check
 */
async function checkAnthropicHealth(apiKey: string, config: any, includeDetails: boolean) {
  try {
    // Simple message API check with minimal tokens
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: "claude-3-haiku-20240307",
        max_tokens: 1,
        messages: [{ role: "user", content: "Hi" }]
      }),
      signal: AbortSignal.timeout(10000)
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Anthropic API error: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    const rateLimits = extractAnthropicRateLimits(response.headers)

    return {
      isHealthy: true,
      status: "healthy",
      details: includeDetails ? {
        model: data.model,
        usage: data.usage
      } : undefined,
      rateLimits: includeDetails ? rateLimits : undefined,
      availableModels: includeDetails ? [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307"
      ] : undefined
    }

  } catch (error) {
    return {
      isHealthy: false,
      status: "unhealthy",
      error: error instanceof Error ? error.message : "Unknown error"
    }
  }
}

/**
 * OpenRouter health check
 */
async function checkOpenRouterHealth(apiKey: string, config: any, includeDetails: boolean) {
  try {
    // Check models endpoint
    const response = await fetch("https://openrouter.ai/api/v1/models", {
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "HTTP-Referer": "https://freeagentics.vercel.app",
        "X-Title": "FreeAgentics"
      },
      signal: AbortSignal.timeout(10000)
    })

    if (!response.ok) {
      throw new Error(`OpenRouter API error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()

    return {
      isHealthy: true,
      status: "healthy",
      details: includeDetails ? {
        modelsCount: data.data?.length || 0
      } : undefined,
      availableModels: includeDetails ? data.data?.slice(0, 5).map((m: any) => m.id) : undefined
    }

  } catch (error) {
    return {
      isHealthy: false,
      status: "unhealthy",
      error: error instanceof Error ? error.message : "Unknown error"
    }
  }
}

/**
 * Azure OpenAI health check
 */
async function checkAzureOpenAIHealth(apiKey: string, config: any, includeDetails: boolean) {
  if (!config.endpointUrl) {
    return {
      isHealthy: false,
      status: "configuration_error",
      error: "Azure endpoint URL not configured"
    }
  }

  try {
    // Check deployments endpoint
    const response = await fetch(`${config.endpointUrl}/openai/deployments?api-version=2023-12-01-preview`, {
      headers: {
        "api-key": apiKey
      },
      signal: AbortSignal.timeout(10000)
    })

    if (!response.ok) {
      throw new Error(`Azure OpenAI API error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()

    return {
      isHealthy: true,
      status: "healthy",
      details: includeDetails ? {
        deploymentsCount: data.data?.length || 0,
        endpoint: config.endpointUrl
      } : undefined,
      availableModels: includeDetails ? data.data?.map((d: any) => d.model) : undefined
    }

  } catch (error) {
    return {
      isHealthy: false,
      status: "unhealthy",
      error: error instanceof Error ? error.message : "Unknown error"
    }
  }
}

/**
 * Extract rate limit information from OpenAI headers
 */
function extractRateLimits(headers: Headers): any {
  return {
    requestsLimit: parseInt(headers.get("x-ratelimit-limit-requests") || "0"),
    requestsRemaining: parseInt(headers.get("x-ratelimit-remaining-requests") || "0"),
    tokensLimit: parseInt(headers.get("x-ratelimit-limit-tokens") || "0"),
    tokensRemaining: parseInt(headers.get("x-ratelimit-remaining-tokens") || "0"),
    resetRequests: headers.get("x-ratelimit-reset-requests"),
    resetTokens: headers.get("x-ratelimit-reset-tokens")
  }
}

/**
 * Extract rate limit information from Anthropic headers
 */
function extractAnthropicRateLimits(headers: Headers): any {
  return {
    requestsLimit: parseInt(headers.get("anthropic-ratelimit-requests-limit") || "0"),
    requestsRemaining: parseInt(headers.get("anthropic-ratelimit-requests-remaining") || "0"),
    tokensLimit: parseInt(headers.get("anthropic-ratelimit-tokens-limit") || "0"),
    tokensRemaining: parseInt(headers.get("anthropic-ratelimit-tokens-remaining") || "0"),
    resetTime: headers.get("anthropic-ratelimit-requests-reset")
  }
}

/**
 * Get all configured provider IDs
 */
async function getAllConfiguredProviders(): Promise<string[]> {
  // In a real implementation, this would query the database
  return ["openai-primary", "anthropic-secondary"]
}

/**
 * Get provider configuration by ID
 */
async function getProviderConfig(providerId: string): Promise<any> {
  // Mock configuration - in real implementation, fetch from database
  const configs: Record<string, any> = {
    "openai-primary": {
      type: "openai",
      sessionId: "session-openai-123",
      organizationId: null
    },
    "anthropic-secondary": {
      type: "anthropic", 
      sessionId: "session-anthropic-456"
    }
  }

  return configs[providerId] || null
} 