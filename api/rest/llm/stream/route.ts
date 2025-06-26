import { NextRequest, NextResponse } from "next/server"

// Import just the function, not the service module with "use server"
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { systemPrompt, userPrompt, settings } = body

    if (!systemPrompt || !userPrompt || !settings) {
      return NextResponse.json(
        { error: "Missing required fields: systemPrompt, userPrompt, settings" },
        { status: 400 }
      )
    }

    // Dynamically import the streaming function to avoid "use server" issues
    const { streamGenerateResponse } = await import("@/lib/llm-service")

    // Create a ReadableStream that will send chunks of the response
    const encoder = new TextEncoder()

    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Use the server-side streaming function
          for await (const chunk of streamGenerateResponse(systemPrompt, userPrompt, settings)) {
            // Send each chunk as a JSON object with text and completion status
            const chunkData = JSON.stringify(chunk) + "\n"
            controller.enqueue(encoder.encode(chunkData))

            // If this is the completion signal, close the stream
            if (chunk.isComplete) {
              controller.close()
              return
            }
          }
        } catch (error) {
          console.error("Error in streaming:", error)
          const errorChunk = JSON.stringify({
            text: `Error: ${error instanceof Error ? error.message : String(error)}`,
            isComplete: true
          }) + "\n"
          controller.enqueue(encoder.encode(errorChunk))
          controller.close()
        }
      }
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    })
  } catch (error) {
    console.error("Error in stream API:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
