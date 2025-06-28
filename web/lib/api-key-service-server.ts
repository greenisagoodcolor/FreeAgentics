// lib/api-key-service-server.ts
// Server-side API key storage using cookies instead of sessionStorage

import { cookies } from "next/headers";
import crypto from "crypto";

// Ensure ENCRYPTION_KEY is available
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY;
if (!ENCRYPTION_KEY) {
  throw new Error("ENCRYPTION_KEY environment variable is not set");
}

function encrypt(text: string): string {
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(
    "aes-256-cbc",
    Buffer.from(ENCRYPTION_KEY, "hex"),
    iv,
  );
  let encrypted = cipher.update(text);
  encrypted = Buffer.concat([encrypted, cipher.final()]);
  return iv.toString("hex") + ":" + encrypted.toString("hex");
}

function decrypt(text: string): string {
  const parts = text.split(":");
  const iv = Buffer.from(parts.shift()!, "hex");
  const encryptedText = Buffer.from(parts.join(":"), "hex");
  const decipher = crypto.createDecipheriv(
    "aes-256-cbc",
    Buffer.from(ENCRYPTION_KEY, "hex"),
    iv,
  );
  let decrypted = decipher.update(encryptedText);
  decrypted = Buffer.concat([decrypted, decipher.final()]);
  return decrypted.toString();
}

export async function storeApiKey(
  provider: string,
  apiKey: string,
): Promise<string> {
  try {
    const sessionId = crypto.randomUUID();
    const encryptedApiKey = encrypt(apiKey);

    const cookieStore = await cookies();
    cookieStore.set(`api_key_${provider}_${sessionId}`, encryptedApiKey, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      maxAge: 60 * 60 * 24, // 24 hours
    });

    console.log(
      `[API-KEY-SERVICE] API key stored with session ID: ${sessionId}`,
    );
    return sessionId;
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error storing API key:", error);
    throw new Error("Failed to store API key securely");
  }
}

export async function retrieveApiKey(
  provider: string,
  sessionId: string,
): Promise<string | null> {
  try {
    const cookieStore = await cookies();
    const encryptedApiKey = cookieStore.get(
      `api_key_${provider}_${sessionId}`,
    )?.value;

    if (!encryptedApiKey) {
      console.log(
        `[API-KEY-SERVICE] No API key found for provider: ${provider}, session: ${sessionId}`,
      );
      return null;
    }

    const decryptedKey = decrypt(encryptedApiKey);
    console.log(
      `[API-KEY-SERVICE] Retrieved API key for provider: ${provider}`,
    );
    return decryptedKey;
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error retrieving API key:", error);
    return null;
  }
}

export async function deleteApiKey(
  provider: string,
  sessionId: string,
): Promise<void> {
  try {
    const cookieStore = await cookies();
    cookieStore.delete(`api_key_${provider}_${sessionId}`);
    console.log(`[API-KEY-SERVICE] Deleted API key for session: ${sessionId}`);
  } catch (error) {
    console.error("[API-KEY-SERVICE] Error deleting API key:", error);
  }
}
