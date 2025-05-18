import { createCipheriv, createDecipheriv, randomBytes } from "crypto"

// Using a consistent encryption key to ensure data can always be decrypted
// This is critical for features like exporting API keys to work properly
const ENCRYPTION_KEY = "e09d9f6d74764569a755b2275d8a1d46bafd4c2499ca4f693c157cadfde2d9e9"

/**
 * Encrypts a string using AES-256-CBC
 * @param text The text to encrypt
 * @returns The encrypted text with IV prepended (format: iv:encryptedText)
 */
export function encrypt(text: string): string {
  // Generate a random initialization vector
  const iv = randomBytes(16)

  // Create cipher with the encryption key and IV
  const cipher = createCipheriv(
    "aes-256-cbc",
    Buffer.from(ENCRYPTION_KEY.length === 64 ? ENCRYPTION_KEY : ENCRYPTION_KEY.slice(0, 64), "hex"),
    iv,
  )

  // Encrypt the text
  let encrypted = cipher.update(text, "utf8", "hex")
  encrypted += cipher.final("hex")

  // Return the IV and encrypted text
  return `${iv.toString("hex")}:${encrypted}`
}

/**
 * Decrypts a string that was encrypted with the encrypt function
 * @param text The encrypted text (format: iv:encryptedText)
 * @returns The decrypted text
 */
export function decrypt(text: string): string {
  // Split the IV and encrypted text
  const [ivHex, encryptedText] = text.split(":")

  if (!ivHex || !encryptedText) {
    throw new Error("Invalid encrypted text format")
  }

  // Convert IV from hex to Buffer
  const iv = Buffer.from(ivHex, "hex")

  // Create decipher with the encryption key and IV
  const decipher = createDecipheriv(
    "aes-256-cbc",
    Buffer.from(ENCRYPTION_KEY.length === 64 ? ENCRYPTION_KEY : ENCRYPTION_KEY.slice(0, 64), "hex"),
    iv,
  )

  // Decrypt the text
  let decrypted = decipher.update(encryptedText, "hex", "utf8")
  decrypted += decipher.final("utf8")

  return decrypted
}
