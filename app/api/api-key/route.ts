import { type NextRequest, NextResponse } from "next/server"
import { storeApiKey, deleteApiKey, validateSession } from "@/lib/api-key-service"

/**
 * POST /api/api-key
 * Stores an API key securely
 *
 * Request body:
 * {
 *   provider: string, // "openai" or "openrouter"
 *   apiKey: string    // The API key to store
 * }
 *
 * Response:
 * {
 *   success: boolean,
 *   sessionId?: string,
 *   error?: string
 * }
 */
export async function POST(request: NextRequest) {
  try {
    // Parse request body
    const body = await request.json()
    const { provider, apiKey } = body

    // Validate inputs
    if (!provider || !apiKey) {
      return NextResponse.json({ success: false, error: "Missing provider or API key" }, { status: 400 })
    }

    if (typeof provider !== "string" || typeof apiKey !== "string") {
      return NextResponse.json({ success: false, error: "Invalid provider or API key format" }, { status: 400 })
    }

    // Only allow supported providers
    if (provider !== "openai" && provider !== "openrouter") {
      return NextResponse.json({ success: false, error: "Unsupported provider" }, { status: 400 })
    }

    // Store the API key
    const sessionId = await storeApiKey(provider, apiKey)

    return NextResponse.json({ success: true, sessionId })
  } catch (error) {
    console.error("Error storing API key:", error)
    return NextResponse.json({ success: false, error: "Server error" }, { status: 500 })
  }
}

/**
 * DELETE /api/api-key
 * Deletes a stored API key
 *
 * Request body:
 * {
 *   provider: string,  // "openai" or "openrouter"
 *   sessionId: string  // The session ID for the stored key
 * }
 *
 * Response:
 * {
 *   success: boolean,
 *   error?: string
 * }
 */
export async function DELETE(request: NextRequest) {
  try {
    // Parse request body
    const body = await request.json()
    const { provider, sessionId } = body

    // Validate inputs
    if (!provider || !sessionId) {
      return NextResponse.json({ success: false, error: "Missing provider or session ID" }, { status: 400 })
    }

    // Delete the API key
    await deleteApiKey(provider, sessionId)

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error deleting API key:", error)
    return NextResponse.json({ success: false, error: "Server error" }, { status: 500 })
  }
}

/**
 * GET /api/api-key/validate
 * Validates if a session has a valid API key
 *
 * Query parameters:
 * - provider: string  // "openai" or "openrouter"
 * - sessionId: string // The session ID to validate
 *
 * Response:
 * {
 *   success: boolean,
 *   valid: boolean,
 *   error?: string
 * }
 */
export async function GET(request: NextRequest) {
  try {
    // Get query parameters
    const { searchParams } = new URL(request.url)
    const provider = searchParams.get("provider")
    const sessionId = searchParams.get("sessionId")

    // Validate inputs
    if (!provider || !sessionId) {
      return NextResponse.json({ success: false, error: "Missing provider or session ID" }, { status: 400 })
    }

    // Validate the session
    const valid = await validateSession(provider, sessionId)

    return NextResponse.json({ success: true, valid })
  } catch (error) {
    console.error("Error validating session:", error)
    return NextResponse.json({ success: false, error: "Server error" }, { status: 500 })
  }
}
