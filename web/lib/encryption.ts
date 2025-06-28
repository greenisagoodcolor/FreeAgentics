export function encrypt(text: string): string {
  // Simple mock encryption for testing
  return `encrypted_${text}`;
}

export function decrypt(text: string): string {
  // Simple mock decryption for testing
  return text.replace('encrypted_', '');
}