import type { KnowledgeEntry } from "@/lib/types"
import { extractTagsFromMarkdown } from "@/lib/utils"
// import { generateResponse } from "@/lib/llm-service" // Avoid direct import from llm-service

// Types for the belief extraction pipeline
export interface ExtractedBelief {
  content: string
  confidence: "High" | "Medium" | "Low"
  source: "conversation"
  tags: string[]
}

export interface RefinedBelief {
  originalIndex: number
  accuracy: number // 1-5 scale
  relevance: number // 1-5 scale
  category: "Fact" | "Opinion" | "Preference" | "Relationship" | "Other"
  title: string
  refined_content: string
  confidence: "High" | "Medium" | "Low"
  tags: string[]
  selected?: boolean
  novelty?: "High" | "Medium" | "Low"
}

/**
 * Creates fallback refined beliefs when LLM refinement fails
 */
export function createFallbackRefinedBeliefs(beliefs: ExtractedBelief[]): RefinedBelief[] {
  return beliefs.map((belief, index) => ({
    originalIndex: index,
    accuracy: 3, // Medium accuracy
    relevance: 3, // Medium relevance
    category: "Fact" as const,
    title: `Knowledge about ${belief.tags[0] || "topic"}`,
    refined_content: belief.content,
    confidence: belief.confidence,
    tags: belief.tags,
    selected: true,
    novelty: "Medium",
  }))
}

/**
 * Parses raw LLM response to extract beliefs
 */
export function parseBeliefs(response: string): ExtractedBelief[] {
  const beliefs: ExtractedBelief[] = []

  // Split by bullet points or numbered lists
  const lines = response
    .split(/\n+/)
    .filter((line) => line.trim().startsWith("-") || line.trim().startsWith("•") || /^\d+\./.test(line.trim()))

  for (const line of lines) {
    const content = line.replace(/^[-•\d.]+\s*/, "").trim()

    // Extract confidence level if present
    let confidence: "High" | "Medium" | "Low" = "Medium"
    const confidenceMatch = content.match(/\s*$$(High|Medium|Low)$$$/i)

    if (confidenceMatch) {
      confidence = confidenceMatch[1] as "High" | "Medium" | "Low"
    }

    // Clean up the content but preserve [[tags]]
    const cleanContent = content.replace(/\s*$$(High|Medium|Low)$$$/i, "").trim()

    // Extract tags using the existing utility
    const tags = extractTagsFromMarkdown(cleanContent)

    if (cleanContent) {
      beliefs.push({
        content: cleanContent,
        confidence,
        source: "conversation",
        tags,
      })
    }
  }

  return beliefs
}

/**
 * Filters out beliefs that are duplicates of existing knowledge
 */
function filterDuplicateBeliefs(beliefs: ExtractedBelief[], existingKnowledge: KnowledgeEntry[]): ExtractedBelief[] {
  // Simple implementation - can be enhanced with more sophisticated similarity detection
  return beliefs.filter((belief) => {
    // Check if this belief is similar to any existing knowledge
    return !existingKnowledge.some((entry) => {
      // Check for content similarity
      const contentSimilarity = calculateTextSimilarity(belief.content, entry.content)

      // Check for tag overlap
      const tagOverlap = belief.tags.some((tag) => entry.tags.includes(tag))

      // Consider it a duplicate if content is very similar or there's significant tag overlap
      return contentSimilarity > 0.7 || (tagOverlap && contentSimilarity > 0.5)
    })
  })
}

/**
 * Calculates text similarity between two strings (simple implementation)
 */
function calculateTextSimilarity(text1: string, text2: string): number {
  // Normalize texts
  const normalize = (text: string) => text.toLowerCase().replace(/[^\w\s]/g, "")
  const normalizedText1 = normalize(text1)
  const normalizedText2 = normalize(text2)

  // Simple word overlap for now
  const words1 = new Set(normalizedText1.split(/\s+/))
  const words2 = new Set(normalizedText2.split(/\s+/))

  // Count common words
  let commonWords = 0
  for (const word of words1) {
    if (words2.has(word)) commonWords++
  }

  // Calculate Jaccard similarity
  const totalUniqueWords = new Set([...words1, ...words2]).size
  return totalUniqueWords > 0 ? commonWords / totalUniqueWords : 0
}

/**
 * Parses refined beliefs from a JSON string, handling potential errors.
 */
export function parseRefinedBeliefs(response: string, sourceBeliefs: ExtractedBelief[]): RefinedBelief[] {
  try {
    // Attempt to parse the JSON response
    const refinedBeliefs = JSON.parse(response) as RefinedBelief[]

    // Validate the parsed beliefs to ensure they have the required properties
    const validBeliefs = refinedBeliefs.filter(
      (belief) =>
        belief &&
        typeof belief.originalIndex === "number" &&
        typeof belief.refined_content === "string" &&
        belief.refined_content.trim() !== "",
    )

    if (validBeliefs.length === 0) {
      console.warn("No valid beliefs found in parsed JSON")
      return createFallbackRefinedBeliefs(sourceBeliefs)
    }

    // Set all beliefs as selected by default and ensure tags are properly extracted
    return validBeliefs.map((belief) => ({
      ...belief,
      selected: belief.selected !== false, // Default to true if not explicitly set to false
      // Ensure tags are properly extracted if missing
      tags: belief.tags || extractTagsFromMarkdown(belief.refined_content),
      // Add novelty field if missing
      novelty: belief.novelty || "Medium",
    }))
  } catch (error) {
    console.error("Error parsing refined beliefs JSON:", error)
    console.log("Raw response:", response)
    return createFallbackRefinedBeliefs(sourceBeliefs)
  }
}
