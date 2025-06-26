import { type NextRequest, NextResponse } from "next/server"
import { deleteApiKey } from "@/lib/api-key-storage"

export async function DELETE(request: NextRequest) {
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

    // Delete the API key
    const success = deleteApiKey(provider, sessionId)

    // Return the result
    return NextResponse.json({ success })
  } catch (error) {
    console.error("Error deleting API key:", error)
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Unknown error deleting API key",
      },
      { status: 500 },
    )
  }
}
