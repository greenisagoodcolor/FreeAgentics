import { type NextRequest, NextResponse } from "next/server"
import { getApiKey } from "@/lib/api-key-service"

export async function GET(request: NextRequest) {
  try {
    // Get provider and sessionId from query parameters
    const provider = request.nextUrl.searchParams.get("provider")
    const sessionId = request.nextUrl.searchParams.get("sessionId")

    // Validate input
    if (!provider) {
      return NextResponse.json(
        {
          success: false,
          message: "Provider is required",
        },
        { status: 400 },
      )
    }

    if (!sessionId) {
      return NextResponse.json(
        {
          success: false,
          message: "Session ID is required",
        },
        { status: 400 },
      )
    }

    // Retrieve the API key
    const apiKey = await getApiKey(provider, sessionId)

    // If no API key was found, return an error
    if (!apiKey) {
      return NextResponse.json(
        {
          success: false,
          message: "API key not found or session invalid",
        },
        { status: 404 },
      )
    }

    // Return the API key
    return NextResponse.json({ success: true, apiKey })
  } catch (error) {
    console.error("Error retrieving API key:", error)
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Unknown error retrieving API key",
      },
      { status: 500 },
    )
  }
}
