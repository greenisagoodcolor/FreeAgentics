"use server";

import { encrypt, decrypt } from "@/lib/encryption";
import { cookies } from "next/headers";

// In-memory storage for demo purposes
// In production, use a proper database like Redis, PostgreSQL, etc.
const apiKeyStorage = new Map<
  string,
  { encryptedApiKey: string; provider: string; createdAt: Date }
>();

// Session cleanup interval (in milliseconds)
const SESSION_LIFETIME = 24 * 60 * 60 * 1000; // 24 hours
const CLEANUP_INTERVAL = 60 * 60 * 1000; // 1 hour

// Cleanup expired sessions periodically
let cleanupTimer: NodeJS.Timeout | null = null;

function startCleanupTimer() {
  if (cleanupTimer) return;

  cleanupTimer = setInterval(() => {
    const now = new Date();
    apiKeyStorage.forEach((session, sessionId) => {
      if (now.getTime() - session.createdAt.getTime() > SESSION_LIFETIME) {
        apiKeyStorage.delete(sessionId);
        console.log(
          `[API-KEY-STORAGE] Cleaned up expired session: ${sessionId}`,
        );
      }
    });
  }, CLEANUP_INTERVAL);
}

/**
 * Store an API key securely on the server and return a session ID
 * Also sets an HTTP-only cookie for the session
 */
export async function storeApiKey(
  provider: string,
  apiKey: string,
): Promise<string> {
  try {
    console.log(`[API-KEY-STORAGE] Storing API key for provider: ${provider}`);

    // Generate a cryptographically secure session ID
    const sessionId = generateSecureSessionId();

    // Encrypt the API key using environment-based encryption
    const encryptedApiKey = await encrypt(apiKey);

    // Store in server-side memory (in production, use a database)
    apiKeyStorage.set(sessionId, {
      encryptedApiKey,
      provider,
      createdAt: new Date(),
    });

    // Set HTTP-only cookie for session management
    const cookieStore = await cookies();
    cookieStore.set(`api_session_${provider}`, sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      maxAge: SESSION_LIFETIME / 1000, // Convert to seconds
      path: "/",
    });

    // Start cleanup timer if not already running
    startCleanupTimer();

    console.log(
      `[API-KEY-STORAGE] API key stored with session ID: ${sessionId}`,
    );
    return sessionId;
  } catch (error) {
    console.error("[API-KEY-STORAGE] Error storing API key:", error);
    throw new Error("Failed to store API key securely");
  }
}

/**
 * Retrieve an API key using a session ID
 * Validates the session and returns the decrypted API key
 */
export async function getApiKey(
  provider: string,
  sessionId?: string,
): Promise<string | null> {
  try {
    // If no sessionId provided, try to get it from cookies
    if (!sessionId) {
      const cookieStore = await cookies();
      sessionId = cookieStore.get(`api_session_${provider}`)?.value;
    }

    if (!sessionId) {
      console.warn(
        `[API-KEY-STORAGE] No session ID found for provider: ${provider}`,
      );
      return null;
    }

    console.log(
      `[API-KEY-STORAGE] Retrieving API key for provider: ${provider}, session ID: ${sessionId}`,
    );

    // Get the stored session data
    const sessionData = apiKeyStorage.get(sessionId);

    if (!sessionData) {
      console.warn(
        `[API-KEY-STORAGE] No session data found for session ID: ${sessionId}`,
      );
      return null;
    }

    // Verify the provider matches
    if (sessionData.provider !== provider) {
      console.warn(
        `[API-KEY-STORAGE] Provider mismatch for session ID: ${sessionId}`,
      );
      return null;
    }

    // Check if session has expired
    const now = new Date();
    if (now.getTime() - sessionData.createdAt.getTime() > SESSION_LIFETIME) {
      console.warn(
        `[API-KEY-STORAGE] Session expired for session ID: ${sessionId}`,
      );
      apiKeyStorage.delete(sessionId);
      return null;
    }

    // Decrypt the API key
    const apiKey = await decrypt(sessionData.encryptedApiKey);

    console.log(`[API-KEY-STORAGE] API key retrieved successfully`);
    return apiKey;
  } catch (error) {
    console.error("[API-KEY-STORAGE] Error retrieving API key:", error);
    return null;
  }
}

/**
 * Validate if a session ID is valid and not expired
 */
export async function validateSession(
  provider: string,
  sessionId?: string,
): Promise<boolean> {
  try {
    // If no sessionId provided, try to get it from cookies
    if (!sessionId) {
      const cookieStore = await cookies();
      sessionId = cookieStore.get(`api_session_${provider}`)?.value;
    }

    if (!sessionId) {
      return false;
    }

    console.log(
      `[API-KEY-STORAGE] Validating session for provider: ${provider}, session ID: ${sessionId}`,
    );

    // Get the stored session data
    const sessionData = apiKeyStorage.get(sessionId);

    if (!sessionData || sessionData.provider !== provider) {
      return false;
    }

    // Check if session has expired
    const now = new Date();
    const isExpired =
      now.getTime() - sessionData.createdAt.getTime() > SESSION_LIFETIME;

    if (isExpired) {
      apiKeyStorage.delete(sessionId);
      return false;
    }

    console.log(`[API-KEY-STORAGE] Session validation result: true`);
    return true;
  } catch (error) {
    console.error("[API-KEY-STORAGE] Error validating session:", error);
    return false;
  }
}

/**
 * Delete an API key and clear the session
 */
export async function deleteApiKey(
  provider: string,
  sessionId?: string,
): Promise<boolean> {
  try {
    // If no sessionId provided, try to get it from cookies
    if (!sessionId) {
      const cookieStore = await cookies();
      sessionId = cookieStore.get(`api_session_${provider}`)?.value;
    }

    if (!sessionId) {
      return false;
    }

    console.log(
      `[API-KEY-STORAGE] Deleting API key for provider: ${provider}, session ID: ${sessionId}`,
    );

    // Remove from storage
    const deleted = apiKeyStorage.delete(sessionId);

    // Clear the cookie
    const cookieStore = await cookies();
    cookieStore.delete(`api_session_${provider}`);

    console.log(`[API-KEY-STORAGE] API key deleted successfully: ${deleted}`);
    return deleted;
  } catch (error) {
    console.error("[API-KEY-STORAGE] Error deleting API key:", error);
    return false;
  }
}

/**
 * Generate a cryptographically secure session ID
 */
function generateSecureSessionId(): string {
  // Use crypto.randomBytes for secure random generation
  const crypto = require("node:crypto");
  return crypto.randomBytes(32).toString("hex");
}

/**
 * Clear all expired sessions (for maintenance)
 */
export async function clearExpiredSessions(): Promise<number> {
  let cleared = 0;
  const now = new Date();

  apiKeyStorage.forEach((session, sessionId) => {
    if (now.getTime() - session.createdAt.getTime() > SESSION_LIFETIME) {
      apiKeyStorage.delete(sessionId);
      cleared++;
    }
  });

  console.log(`[API-KEY-STORAGE] Cleared ${cleared} expired sessions`);
  return cleared;
}
