import { type NextRequest, NextResponse } from "next/server"
import { getApiKey, storeApiKey, deleteApiKey } from "@/lib/api-key-storage"
import { encrypt, decrypt } from "@/lib/crypto-utils"
import { validateJWT, requireRole } from "@/lib/auth-middleware"
import { auditLog } from "@/lib/audit-logger"

/**
 * LLM Provider Management API
 * 
 * Implements ADR-008 (API Interface Layer Architecture) and 
 * ADR-011 (Security and Authentication Architecture) compliance.
 * 
 * Provides secure CRUD operations for LLM provider configurations
 * with encrypted credential handling and comprehensive audit logging.
 */

interface ProviderConfig {
  id: string
  name: string
  type: 'openai' | 'anthropic' | 'openrouter' | 'azure_openai' | 'vertex_ai' | 'cohere' | 'ollama'
  enabled: boolean
  priority: number
  credentials: {
    hasApiKey: boolean
    sessionId?: string
    organizationId?: string
    endpoint?: string
  }
  configuration: {
    maxRequestsPerMinute: number
    maxConcurrentRequests: number
    timeoutMs: number
    retryAttempts: number
    retryDelayMs: number
    fallbackEnabled: boolean
    healthCheckInterval: number
  }
  status?: {
    isHealthy: boolean
    lastHealthCheck: string
    errorCount: number
    responseTimeMs: number
  }
  usage?: {
    totalRequests: number
    successfulRequests: number
    totalCost: number
    lastUsed: string
  }
  createdAt: string
  updatedAt: string
}

/**
 * GET /api/llm/providers
 * Retrieve all configured LLM providers
 */
export async function GET(request: NextRequest) {
  try {
    // Validate authentication
    const authResult = await validateJWT(request)
    if (!authResult.success) {
      return NextResponse.json({
        success: false,
        error: "Authentication required"
      }, { status: 401 })
    }

    // Audit log access
    await auditLog("provider_list_access", {
      userId: authResult.user.id,
      timestamp: new Date().toISOString(),
      ip: request.ip || request.headers.get('x-forwarded-for') || 'unknown'
    })

    // Get all providers from database/storage
    const providers = await getAllProviders()

    return NextResponse.json({
      success: true,
      providers,
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error("[PROVIDERS] Error retrieving providers:", error)
    return NextResponse.json({
      success: false,
      error: "Failed to retrieve providers"
    }, { status: 500 })
  }
}

/**
 * POST /api/llm/providers
 * Create a new LLM provider configuration
 */
export async function POST(request: NextRequest) {
  try {
    // Validate authentication and authorization
    const authResult = await validateJWT(request)
    if (!authResult.success) {
      return NextResponse.json({
        success: false,
        error: "Authentication required"
      }, { status: 401 })
    }

    // Require admin or developer role
    if (!requireRole(authResult.user, ['admin', 'developer'])) {
      return NextResponse.json({
        success: false,
        error: "Insufficient permissions"
      }, { status: 403 })
    }

    const body = await request.json()
    const { name, type, credentials, configuration = {}, enabled = false } = body

    // Validate required fields
    if (!name || !type || !credentials) {
      return NextResponse.json({
        success: false,
        error: "Missing required fields: name, type, credentials"
      }, { status: 400 })
    }

    // Generate provider ID
    const providerId = `${type}-${Date.now()}`

    // Handle credential encryption
    let sessionId: string | undefined
    let encryptedCredentials: any = {}

    if (credentials.apiKey) {
      // Encrypt API key using Web Crypto API equivalent
      const encryptionResult = await encrypt(credentials.apiKey)
      sessionId = encryptionResult.sessionId
      
      // Store encrypted credentials
      await storeApiKey(type, sessionId, credentials.apiKey, {
        organizationId: credentials.organizationId,
        endpoint: credentials.endpoint
      })

      encryptedCredentials = {
        hasApiKey: true,
        sessionId,
        organizationId: credentials.organizationId,
        endpoint: credentials.endpoint
      }
    }

    // Create provider configuration
    const providerConfig: ProviderConfig = {
      id: providerId,
      name,
      type,
      enabled,
      priority: await getNextPriority(),
      credentials: encryptedCredentials,
      configuration: {
        maxRequestsPerMinute: configuration.maxRequestsPerMinute || 60,
        maxConcurrentRequests: configuration.maxConcurrentRequests || 5,
        timeoutMs: configuration.timeoutMs || 30000,
        retryAttempts: configuration.retryAttempts || 3,
        retryDelayMs: configuration.retryDelayMs || 1000,
        fallbackEnabled: configuration.fallbackEnabled ?? true,
        healthCheckInterval: configuration.healthCheckInterval || 300
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }

    // Save provider configuration
    await saveProviderConfig(providerConfig)

    // Audit log creation
    await auditLog("provider_created", {
      userId: authResult.user.id,
      providerId,
      providerType: type,
      providerName: name,
      timestamp: new Date().toISOString(),
      ip: request.ip || request.headers.get('x-forwarded-for') || 'unknown'
    })

    return NextResponse.json({
      success: true,
      provider: {
        ...providerConfig,
        credentials: {
          hasApiKey: encryptedCredentials.hasApiKey,
          sessionId: encryptedCredentials.sessionId
        }
      }
    }, { status: 201 })

  } catch (error) {
    console.error("[PROVIDERS] Error creating provider:", error)
    return NextResponse.json({
      success: false,
      error: "Failed to create provider"
    }, { status: 500 })
  }
}

/**
 * PUT /api/llm/providers
 * Update provider configurations or reorder priorities
 */
export async function PUT(request: NextRequest) {
  try {
    // Validate authentication and authorization
    const authResult = await validateJWT(request)
    if (!authResult.success) {
      return NextResponse.json({
        success: false,
        error: "Authentication required"
      }, { status: 401 })
    }

    if (!requireRole(authResult.user, ['admin', 'developer'])) {
      return NextResponse.json({
        success: false,
        error: "Insufficient permissions"
      }, { status: 403 })
    }

    const body = await request.json()
    const { action, providerId, providers, updates } = body

    if (action === "reorder") {
      // Handle priority reordering
      if (!providers || !Array.isArray(providers)) {
        return NextResponse.json({
          success: false,
          error: "Invalid providers array for reordering"
        }, { status: 400 })
      }

      await reorderProviders(providers)

      // Audit log reordering
      await auditLog("providers_reordered", {
        userId: authResult.user.id,
        providerOrder: providers.map(p => ({ id: p.id, priority: p.priority })),
        timestamp: new Date().toISOString()
      })

      return NextResponse.json({
        success: true,
        message: "Provider priorities updated"
      })

    } else if (action === "update") {
      // Handle individual provider update
      if (!providerId || !updates) {
        return NextResponse.json({
          success: false,
          error: "Missing providerId or updates for provider update"
        }, { status: 400 })
      }

      const updatedProvider = await updateProviderConfig(providerId, updates)

      // Audit log update
      await auditLog("provider_updated", {
        userId: authResult.user.id,
        providerId,
        updates,
        timestamp: new Date().toISOString()
      })

      return NextResponse.json({
        success: true,
        provider: updatedProvider
      })

    } else {
      return NextResponse.json({
        success: false,
        error: "Invalid action. Use 'reorder' or 'update'"
      }, { status: 400 })
    }

  } catch (error) {
    console.error("[PROVIDERS] Error updating providers:", error)
    return NextResponse.json({
      success: false,
      error: "Failed to update providers"
    }, { status: 500 })
  }
}

/**
 * DELETE /api/llm/providers
 * Delete a provider configuration
 */
export async function DELETE(request: NextRequest) {
  try {
    // Validate authentication and authorization
    const authResult = await validateJWT(request)
    if (!authResult.success) {
      return NextResponse.json({
        success: false,
        error: "Authentication required"
      }, { status: 401 })
    }

    // Require admin role for deletion
    if (!requireRole(authResult.user, ['admin'])) {
      return NextResponse.json({
        success: false,
        error: "Admin role required for provider deletion"
      }, { status: 403 })
    }

    const { searchParams } = new URL(request.url)
    const providerId = searchParams.get('id')

    if (!providerId) {
      return NextResponse.json({
        success: false,
        error: "Provider ID required"
      }, { status: 400 })
    }

    // Get provider config before deletion for audit
    const providerConfig = await getProviderConfig(providerId)
    if (!providerConfig) {
      return NextResponse.json({
        success: false,
        error: "Provider not found"
      }, { status: 404 })
    }

    // Delete encrypted credentials
    if (providerConfig.credentials.sessionId) {
      await deleteApiKey(providerConfig.type, providerConfig.credentials.sessionId)
    }

    // Delete provider configuration
    await deleteProviderConfig(providerId)

    // Audit log deletion
    await auditLog("provider_deleted", {
      userId: authResult.user.id,
      providerId,
      providerType: providerConfig.type,
      providerName: providerConfig.name,
      timestamp: new Date().toISOString()
    })

    return NextResponse.json({
      success: true,
      message: "Provider deleted successfully"
    })

  } catch (error) {
    console.error("[PROVIDERS] Error deleting provider:", error)
    return NextResponse.json({
      success: false,
      error: "Failed to delete provider"
    }, { status: 500 })
  }
}

// Helper functions (would be implemented in separate modules)

async function getAllProviders(): Promise<ProviderConfig[]> {
  // Mock implementation - in production, fetch from database
  return [
    {
      id: "openai-primary",
      name: "OpenAI Primary",
      type: "openai",
      enabled: true,
      priority: 1,
      credentials: {
        hasApiKey: true,
        sessionId: "session-abc123",
        organizationId: "org-xyz"
      },
      configuration: {
        maxRequestsPerMinute: 100,
        maxConcurrentRequests: 10,
        timeoutMs: 30000,
        retryAttempts: 3,
        retryDelayMs: 1000,
        fallbackEnabled: true,
        healthCheckInterval: 300
      },
      status: {
        isHealthy: true,
        lastHealthCheck: new Date().toISOString(),
        errorCount: 0,
        responseTimeMs: 250
      },
      usage: {
        totalRequests: 1547,
        successfulRequests: 1523,
        totalCost: 12.45,
        lastUsed: new Date().toISOString()
      },
      createdAt: "2025-01-15T10:00:00Z",
      updatedAt: new Date().toISOString()
    }
  ]
}

async function getNextPriority(): Promise<number> {
  const providers = await getAllProviders()
  return Math.max(...providers.map(p => p.priority), 0) + 1
}

async function saveProviderConfig(config: ProviderConfig): Promise<void> {
  // Mock implementation - in production, save to database
  console.log("Saving provider config:", config.id)
}

async function getProviderConfig(providerId: string): Promise<ProviderConfig | null> {
  // Mock implementation - in production, fetch from database
  const providers = await getAllProviders()
  return providers.find(p => p.id === providerId) || null
}

async function updateProviderConfig(
  providerId: string, 
  updates: Partial<ProviderConfig>
): Promise<ProviderConfig> {
  // Mock implementation - in production, update in database
  const provider = await getProviderConfig(providerId)
  if (!provider) {
    throw new Error("Provider not found")
  }
  
  const updatedProvider = {
    ...provider,
    ...updates,
    updatedAt: new Date().toISOString()
  }
  
  await saveProviderConfig(updatedProvider)
  return updatedProvider
}

async function deleteProviderConfig(providerId: string): Promise<void> {
  // Mock implementation - in production, delete from database
  console.log("Deleting provider config:", providerId)
}

async function reorderProviders(
  providers: { id: string; priority: number }[]
): Promise<void> {
  // Mock implementation - in production, update priorities in database
  console.log("Reordering providers:", providers)
} 