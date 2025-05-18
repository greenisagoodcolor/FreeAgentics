import { encrypt, decrypt } from "@/lib/encryption"

// Store an API key securely and return a session ID
export async function storeApiKey(provider: string, apiKey: string): Promise<string> {
  try {
    console.log(`[API-KEY-SERVICE] Storing API key for provider: ${provider}`)

    // Generate a unique session ID
    const sessionId = generateSessionId()

    // Encrypt the API key
    const encryptedApiKey = await encrypt(apiKey)

    // Store the encrypted API key in the session storage
    sessionStorage.setItem(`api_key_${provider}_${sessionId}`, encryptedApiKey)

    console.log(`[API-KEY-SERVICE] API key stored with session ID: ${sessionId}`)
    return sessionId
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error storing API key:", error)
    throw new Error("Failed to store API key securely")
  }
}

// Retrieve an API key using a session ID
export async function getApiKey(provider: string, sessionId: string): Promise<string | null> {
  try {
    console.log(`[API-KEY-SERVICE] Retrieving API key for provider: ${provider}, session ID: ${sessionId}`)

    // Get the encrypted API key from session storage
    const encryptedApiKey = sessionStorage.getItem(`api_key_${provider}_${sessionId}`)

    if (!encryptedApiKey) {
      console.warn(`[API-KEY-SERVICE] No API key found for provider: ${provider}, session ID: ${sessionId}`)
      return null
    }

    // Decrypt the API key
    const apiKey = await decrypt(encryptedApiKey)

    console.log(`[API-KEY-SERVICE] API key retrieved successfully`)
    return apiKey
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error retrieving API key:", error)
    return null
  }
}

// Validate if a session ID is valid
export async function validateSession(provider: string, sessionId: string): Promise<boolean> {
  try {
    console.log(`[API-KEY-SERVICE] Validating session for provider: ${provider}, session ID: ${sessionId}`)

    // Check if the API key exists in session storage
    const encryptedApiKey = sessionStorage.getItem(`api_key_${provider}_${sessionId}`)

    const isValid = !!encryptedApiKey
    console.log(`[API-KEY-SERVICE] Session validation result: ${isValid}`)

    return isValid
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error validating session:", error)
    return false
  }
}

// Delete an API key
export async function deleteApiKey(provider: string, sessionId: string): Promise<boolean> {
  try {
    console.log(`[API-KEY-SERVICE] Deleting API key for provider: ${provider}, session ID: ${sessionId}`)

    // Remove the API key from session storage
    sessionStorage.removeItem(`api_key_${provider}_${sessionId}`)

    console.log(`[API-KEY-SERVICE] API key deleted successfully`)
    return true
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error deleting API key:", error)
    return false
  }
}

// Generate a random session ID
function generateSessionId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
}
