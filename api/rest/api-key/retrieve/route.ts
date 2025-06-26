import { type NextRequest, NextResponse } from "next/server"
import { retrieveApiKey } from "@/lib/api-key-service-server"

export async function POST(request: NextRequest) {
  try {
    const { provider, sessionId } = await request.json()

    // Validate input
    if (!provider || !sessionId) {
      return NextResponse.json(
        { error: "Provider and sessionId are required" },
        { status: 400 }
      )
    }

    // Retrieve the API key
    const apiKey = await retrieveApiKey(provider, sessionId)

    if (!apiKey) {
      console.warn(`[API] API key not found for provider: ${provider}`)
      return NextResponse.json(
        { error: "API key not found" },
        { status: 404 }
      )
    }

    return NextResponse.json({ apiKey })
  } catch (error) {
    console.error("[API] Error retrieving API key:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
