import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Update the extractTagsFromMarkdown function to also find wiki-style links
export function extractTagsFromMarkdown(markdown: string): string[] {
  // Match both [[tag]] syntax and #tag syntax
  const tagRegex = /\[\[(.*?)\]\]|#(\w+)/g
  const matches = Array.from(markdown.matchAll(tagRegex))

  if (!matches.length) return []

  return matches
    .map((match) => (match[1] || match[2]).trim()) // Get the tag from either capture group
    .filter((tag, index, self) => self.indexOf(tag) === index) // Remove duplicates
}

export function formatTimestamp(date: Date | string | number): string {
  try {
    // Ensure we have a valid Date object
    const validDate = date instanceof Date ? date : new Date(date)

    // Check if the date is valid
    if (isNaN(validDate.getTime())) {
      console.warn("Invalid date value:", date)
      return "Invalid date"
    }

    return validDate.toISOString().split("T")[0]
  } catch (error) {
    console.error("Error formatting timestamp:", error, date)
    return "Invalid date"
  }
}
