import { type NextRequest, NextResponse } from "next/server"
import { getApiKey } from "@/lib/api-key-storage"
import { generateOpenAIResponse, generateOpenRouterResponse, type LLMRequestOptions } from "@/lib/llm-providers"

/**
 * POST /api/llm/generate
 * Generates a response from an LLM provider using a securely stored API key
 *
 * Request body:
 * {
 *   provider: "openai" | "openrouter",
 *   sessionId: string,
 *   model: string,
 *   systemPrompt: string,
 *   userPrompt: string,
 *   temperature: number,
 *   maxTokens: number,
 *   topP: number,
 *   frequencyPenalty: number,
 *   presencePenalty: number,
 *   systemFingerprint?: boolean
 * }
 *
 * Response:
 * {
 *   success: boolean,
 *   response?: string,
 *   error?: string
 * }
 */
export async function POST(request: NextRequest) {
  try {
    // Parse request body
    const body = await request.json()
    const {
      provider,
      sessionId,
      model,
      systemPrompt,
      userPrompt,
      temperature = 0.7,
      maxTokens = 1024,
      topP = 0.9,
      frequencyPenalty = 0,
      presencePenalty = 0,
      systemFingerprint = false,
    } = body

    // Validate required inputs
    if (!provider || !sessionId || !model || !systemPrompt || !userPrompt) {
      return NextResponse.json({ success: false, error: "Missing required parameters" }, { status: 400 })
    }

    // Validate provider
    if (provider !== "openai" && provider !== "openrouter") {
      return NextResponse.json({ success: false, error: "Unsupported provider" }, { status: 400 })
    }

    // Get API key from secure storage
    const apiKey = await getApiKey(provider, sessionId)

    if (!apiKey) {
      return NextResponse.json(
        { success: false, error: "API key not found. Please re-enter your API key in settings." },
        { status: 401 },
      )
    }

    // Prepare options
    const options: LLMRequestOptions = {
      temperature,
      maxTokens,
      topP,
      frequencyPenalty,
      presencePenalty,
      systemFingerprint,
    }

    // Generate response based on provider
    let response: string
    if (provider === "openai") {
      response = await generateOpenAIResponse(apiKey, model, systemPrompt, userPrompt, options)
    } else {
      response = await generateOpenRouterResponse(apiKey, model, systemPrompt, userPrompt, options)
    }

    return NextResponse.json({ success: true, response })
  } catch (error) {
    console.error("Error generating LLM response:", error)
    const errorMessage = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ success: false, error: `Error generating response: ${errorMessage}` }, { status: 500 })
  }
}
