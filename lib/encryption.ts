"use server"

import { createCipheriv, createDecipheriv, randomBytes } from "crypto"

/**
 * Get the encryption key from environment variables
 * This ensures the key is never exposed in client-side code
 */
function getEncryptionKey(): Buffer {
  const key = process.env.ENCRYPTION_KEY
  
  if (!key) {
    throw new Error("ENCRYPTION_KEY environment variable is not set")
  }
  
  // Ensure key is 64 hex characters (32 bytes for AES-256)
  if (key.length !== 64) {
    throw new Error("ENCRYPTION_KEY must be 64 hex characters (32 bytes)")
  }
  
  // Validate it's valid hex
  if (!/^[0-9a-fA-F]{64}$/.test(key)) {
    throw new Error("ENCRYPTION_KEY must contain only hex characters")
  }
  
  return Buffer.from(key, "hex")
}

/**
 * Encrypts a string using AES-256-CBC with environment-based key
 * @param text The text to encrypt
 * @returns The encrypted text with IV prepended (format: iv:encryptedText)
 */
export async function encrypt(text: string): Promise<string> {
  try {
    // Generate a random initialization vector
    const iv = randomBytes(16)
    
    // Get encryption key from environment
    const encryptionKey = getEncryptionKey()

    // Create cipher with the encryption key and IV
    const cipher = createCipheriv("aes-256-cbc", encryptionKey, iv)

    // Encrypt the text
    let encrypted = cipher.update(text, "utf8", "hex")
    encrypted += cipher.final("hex")

    // Return the IV and encrypted text
    return `${iv.toString("hex")}:${encrypted}`
  } catch (error) {
    console.error("[ENCRYPTION] Error encrypting text:", error)
    throw new Error("Failed to encrypt data")
  }
}

/**
 * Decrypts a string that was encrypted with the encrypt function
 * @param text The encrypted text (format: iv:encryptedText)
 * @returns The decrypted text
 */
export async function decrypt(text: string): Promise<string> {
  try {
    // Split the IV and encrypted text
    const [ivHex, encryptedText] = text.split(":")

    if (!ivHex || !encryptedText) {
      throw new Error("Invalid encrypted text format")
    }

    // Convert IV from hex to Buffer
    const iv = Buffer.from(ivHex, "hex")
    
    // Get encryption key from environment
    const encryptionKey = getEncryptionKey()

    // Create decipher with the encryption key and IV
    const decipher = createDecipheriv("aes-256-cbc", encryptionKey, iv)

    // Decrypt the text
    let decrypted = decipher.update(encryptedText, "hex", "utf8")
    decrypted += decipher.final("utf8")

    return decrypted
  } catch (error) {
    console.error("[ENCRYPTION] Error decrypting text:", error)
    throw new Error("Failed to decrypt data")
  }
}
