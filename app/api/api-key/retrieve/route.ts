import { type NextRequest, NextResponse } from "next/server"
import { getApiKey } from "@/lib/api-key-storage"

export async function GET(request: NextRequest) {
  try {
    console.log("[API] GET /api/api-key/retrieve - Request received")
    
    // Get provider and sessionId from query parameters
    const provider = request.nextUrl.searchParams.get("provider")
    const sessionId = request.nextUrl.searchParams.get("sessionId")

    console.log(`[API] Provider: ${provider}, Session ID: ${sessionId}`)

    // Validate input
    if (!provider) {
      console.error("[API] Missing provider")
      return NextResponse.json(
        {
          success: false,
          message: "Provider is required",
        },
        { status: 400 },
      )
    }

    // Session ID is optional - the storage system will try to get it from cookies
    // Retrieve the API key (will check cookies if sessionId not provided)
    const apiKey = await getApiKey(provider, sessionId || undefined)

    // If no API key was found, return an error
    if (!apiKey) {
      console.warn(`[API] API key not found for provider: ${provider}`)
      return NextResponse.json(
        {
          success: false,
          message: "API key not found or session invalid",
        },
        { status: 404 },
      )
    }

    console.log(`[API] API key retrieved successfully for provider: ${provider}`)
    // Return the API key
    return NextResponse.json({ success: true, apiKey })
  } catch (error) {
    console.error("[API] Error retrieving API key:", error)
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Unknown error retrieving API key",
      },
      { status: 500 },
    )
  }
}
