/**
 * Stores an API key session ID in localStorage
 * @param provider The API provider (e.g., "openai", "openrouter")
 * @param sessionId The session ID to store
 */
export function storeSessionId(provider: string, sessionId: string): void {
  try {
    console.log(`Storing session ID for provider ${provider}:`, sessionId)
    localStorage.setItem(`api_session_${provider}`, sessionId)
  } catch (error) {
    console.error("Error storing session ID:", error)
  }
}

/**
 * Retrieves an API key session ID from localStorage
 * @param provider The API provider
 * @returns The session ID, or null if not found
 */
export function getSessionId(provider: string): string | null {
  try {
    const sessionId = localStorage.getItem(`api_session_${provider}`)
    console.log(`Retrieved session ID for provider ${provider}:`, sessionId)
    return sessionId
  } catch (error) {
    console.error("Error getting session ID:", error)
    return null
  }
}

/**
 * Validates if a stored session ID is valid
 * @param provider The API provider
 * @returns Promise resolving to true if valid, false otherwise
 */
export async function validateStoredSession(provider: string): Promise<boolean> {
  try {
    const sessionId = getSessionId(provider)
    if (!sessionId) return false

    try {
      const response = await fetch(
        `/api/api-key/validate?provider=${encodeURIComponent(provider)}&sessionId=${encodeURIComponent(sessionId)}`,
      )

      if (!response.ok) {
        console.error(`Error validating session: HTTP ${response.status}`)
        return false
      }

      const data = await response.json()
      return data.success && data.valid
    } catch (error) {
      console.error("Error validating session:", error)
      return false
    }
  } catch (error) {
    console.error("Error validating session:", error)
    return false
  }
}

/**
 * Retrieves an API key using the stored session ID
 * @param provider The API provider
 * @returns Promise resolving to the API key, or null if not found
 */
export async function getApiKeyFromSession(provider: string): Promise<string | null> {
  const sessionId = getSessionId(provider)
  if (!sessionId) return null

  try {
    const response = await fetch(
      `/api/api-key/retrieve?provider=${encodeURIComponent(provider)}&sessionId=${encodeURIComponent(sessionId)}`,
    )

    if (!response.ok) {
      console.error(`Error retrieving API key: HTTP ${response.status}`)
      return null
    }

    const data = await response.json()

    if (!data.success) {
      console.error("Failed to retrieve API key:", data.message)
      return null
    }

    return data.apiKey
  } catch (error) {
    console.error("Error retrieving API key:", error)
    return null
  }
}

/**
 * Clears a stored session ID
 * @param provider The API provider
 */
export function clearSessionId(provider: string): void {
  localStorage.removeItem(`api_session_${provider}`)
}
