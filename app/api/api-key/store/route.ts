import { type NextRequest, NextResponse } from "next/server"
import { storeApiKey } from "@/lib/api-key-service"

export async function POST(request: NextRequest) {
  try {
    console.log("[API] POST /api/api-key/store - Request received")

    // Parse the request body
    let body
    try {
      body = await request.json()
    } catch (error) {
      console.error("[API] Error parsing request body:", error)
      return NextResponse.json(
        {
          success: false,
          message: "Invalid JSON in request body",
        },
        { status: 400 },
      )
    }

    const { provider, apiKey } = body
    console.log(`[API] Provider: ${provider}, API key length: ${apiKey ? apiKey.length : 0}`)

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

    if (!apiKey || typeof apiKey !== "string" || apiKey.trim() === "") {
      console.error("[API] Invalid API key")
      return NextResponse.json(
        {
          success: false,
          message: "Valid API key is required",
        },
        { status: 400 },
      )
    }

    // Store the API key and get a session ID
    const sessionId = await storeApiKey(provider, apiKey)
    console.log(`[API] API key stored successfully with session ID: ${sessionId}`)

    // Return the session ID
    return NextResponse.json({ success: true, sessionId })
  } catch (error) {
    console.error("[API] Error storing API key:", error)
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Unknown error storing API key",
      },
      { status: 500 },
    )
  }
}
