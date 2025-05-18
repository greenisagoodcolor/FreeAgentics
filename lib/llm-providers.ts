/**
 * Provider-specific implementations for LLM API calls
 * This file contains the code for making API calls to different LLM providers
 */

// Common options interface for all providers
export interface LLMRequestOptions {
  temperature: number
  maxTokens: number
  topP: number
  frequencyPenalty: number
  presencePenalty: number
  systemFingerprint?: boolean
}

/**
 * Generates a response from OpenAI
 */
export async function generateOpenAIResponse(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  options: LLMRequestOptions,
): Promise<string> {
  const url = "https://api.openai.com/v1/chat/completions"

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      top_p: options.topP,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
      ...(options.systemFingerprint ? { system_fingerprint: true } : {}),
    }),
  })

  if (!response.ok) {
    const errorData = await response.text()
    throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorData}`)
  }

  const data = await response.json()
  return data.choices[0].message.content
}

/**
 * Generates a response from OpenRouter
 */
export async function generateOpenRouterResponse(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  options: LLMRequestOptions,
): Promise<string> {
  const url = "https://openrouter.ai/api/v1/chat/completions"

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      "HTTP-Referer": "https://cogniticnet.vercel.app",
      "X-Title": "CogniticNet",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      top_p: options.topP,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
    }),
  })

  if (!response.ok) {
    const errorData = await response.text()
    throw new Error(`OpenRouter API error: ${response.status} ${response.statusText} - ${errorData}`)
  }

  const data = await response.json()
  return data.choices[0].message.content
}

/**
 * Streams a response from OpenAI
 * This is a placeholder for future implementation
 */
export async function streamOpenAIResponse(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  options: LLMRequestOptions,
): Promise<ReadableStream> {
  // This would be implemented for streaming responses
  throw new Error("Streaming not yet implemented")
}

/**
 * Streams a response from OpenRouter
 * This is a placeholder for future implementation
 */
export async function streamOpenRouterResponse(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  options: LLMRequestOptions,
): Promise<ReadableStream> {
  // This would be implemented for streaming responses
  throw new Error("Streaming not yet implemented")
}
