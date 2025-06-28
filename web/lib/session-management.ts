import { isBrowser } from "./browser-check";

export function getSessionId(provider: string): string | null {
  if (!isBrowser) {
    console.log("getSessionId called on server, returning null");
    return null;
  }

  try {
    const sessionId = localStorage.getItem(`api_session_${provider}`);
    console.log(`Retrieved session ID for provider ${provider}:`, sessionId);
    return sessionId;
  } catch (error) {
    console.error("Error getting session ID:", error);
    return null;
  }
}

export function setSessionId(provider: string, sessionId: string): void {
  if (!isBrowser) {
    console.log("setSessionId called on server, skipping");
    return;
  }

  try {
    localStorage.setItem(`api_session_${provider}`, sessionId);
    console.log(`Stored session ID for provider ${provider}:`, sessionId);
  } catch (error) {
    console.error("Error setting session ID:", error);
  }
}

export function clearSessionId(provider: string): void {
  if (!isBrowser) {
    console.log("clearSessionId called on server, skipping");
    return;
  }

  try {
    localStorage.removeItem(`api_session_${provider}`);
    console.log(`Cleared session ID for provider ${provider}`);
  } catch (error) {
    console.error("Error clearing session ID:", error);
  }
}

export function storeSessionId(provider: string, sessionId: string): void {
  if (!isBrowser) {
    console.log("storeSessionId called on server, skipping");
    return;
  }

  try {
    localStorage.setItem(`api_session_${provider}`, sessionId);
    console.log(`Stored session ID for provider ${provider}:`, sessionId);
  } catch (error) {
    console.error("Error storing session ID:", error);
  }
}

export async function getApiKeyFromSession(
  provider: string,
): Promise<string | null> {
  try {
    const sessionId = getSessionId(provider);
    if (!sessionId) return null;

    const response = await fetch("/api/api-key/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, sessionId }),
    });

    if (!response.ok) return null;
    const data = await response.json();
    return data.apiKey;
  } catch (error) {
    console.error("Error retrieving API key:", error);
    return null;
  }
}

export function validateStoredSession(
  provider: string,
  sessionId: string,
): boolean {
  const storedSessionId = getSessionId(provider);
  return storedSessionId === sessionId;
}
