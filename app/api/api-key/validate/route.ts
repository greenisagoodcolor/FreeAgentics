import { type NextRequest, NextResponse } from "next/server"
import { validateSession } from "@/lib/api-key-service"

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

    // Validate the session
    const valid = await validateSession(provider, sessionId)

    // Return the validation result
    return NextResponse.json({ success: true, valid })
  } catch (error) {
    console.error("Error validating session:", error)
    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : "Unknown error validating session",
      },
      { status: 500 },
    )
  }
}
