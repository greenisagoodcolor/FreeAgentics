import type { KnowledgeEntry } from "@/lib/types";

export interface RetrievalResult {
  entries: KnowledgeEntry[];
  relevanceScores: number[];
}

export interface RetrievalOptions {
  maxResults?: number;
  minRelevanceScore?: number;
  includeTags?: string[];
  excludeTags?: string[];
  recencyBoost?: boolean;
}

export class KnowledgeRetriever {
  /**
   * Retrieves relevant knowledge entries based on a query
   */
  retrieveRelevant(
    query: string,
    knowledgeBase: KnowledgeEntry[],
    options: RetrievalOptions = {},
  ): RetrievalResult {
    const {
      maxResults = 5,
      minRelevanceScore = 0.1,
      includeTags = [],
      excludeTags = [],
      recencyBoost = true,
    } = options;

    // Normalize query for matching
    const normalizedQuery = query.toLowerCase();
    const queryTerms = this.extractTerms(normalizedQuery);

    // Calculate relevance for each entry
    const entriesWithScores = knowledgeBase.map((entry) => {
      // Skip entries with excluded tags
      if (
        excludeTags.length > 0 &&
        entry.tags.some((tag) => excludeTags.includes(tag))
      ) {
        return { entry, score: 0 };
      }

      // Boost entries with included tags
      let tagBoost = 0;
      if (includeTags.length > 0) {
        tagBoost =
          entry.tags.filter((tag) => includeTags.includes(tag)).length * 0.2;
      }

      // Calculate base relevance score
      let score = this.calculateRelevance(queryTerms, entry);

      // Apply tag boost
      score += tagBoost;

      // Apply recency boost if enabled
      if (recencyBoost) {
        // Calculate days since entry was created
        const daysSinceCreation =
          (Date.now() - entry.timestamp.getTime()) / (1000 * 60 * 60 * 24);
        // Boost recent entries (max 0.2 boost for entries created today, decreasing over 30 days)
        const recencyBoostValue = Math.max(
          0,
          0.2 * (1 - daysSinceCreation / 30),
        );
        score += recencyBoostValue;
      }

      return { entry, score };
    });

    // Filter by minimum relevance score and sort by relevance
    const filteredEntries = entriesWithScores
      .filter((item) => item.score >= minRelevanceScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, maxResults);

    // Return formatted result
    return {
      entries: filteredEntries.map((item) => item.entry),
      relevanceScores: filteredEntries.map((item) => item.score),
    };
  }

  /**
   * Extracts terms from text for matching
   */
  private extractTerms(text: string): string[] {
    // Remove punctuation and split by whitespace
    return text
      .replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, "")
      .split(/\s+/)
      .filter((term) => term.length > 2); // Filter out very short terms
  }

  /**
   * Calculates relevance score between query terms and a knowledge entry
   */
  private calculateRelevance(
    queryTerms: string[],
    entry: KnowledgeEntry,
  ): number {
    // Extract terms from entry title and content
    const titleTerms = this.extractTerms(entry.title.toLowerCase());
    const contentTerms = this.extractTerms(entry.content.toLowerCase());

    // Count matching terms in title (with higher weight) and content
    const titleMatches = queryTerms.filter((term) =>
      titleTerms.some(
        (titleTerm) => titleTerm.includes(term) || term.includes(titleTerm),
      ),
    ).length;

    const contentMatches = queryTerms.filter((term) =>
      contentTerms.some(
        (contentTerm) =>
          contentTerm.includes(term) || term.includes(contentTerm),
      ),
    ).length;

    // Calculate weighted score
    // Title matches are weighted 3x more than content matches
    const weightedMatches = titleMatches * 3 + contentMatches;

    // Normalize by query length with a slight boost for multiple matches
    const queryLength = queryTerms.length || 1; // Avoid division by zero
    const score =
      (weightedMatches / queryLength) *
      (1 + Math.min(weightedMatches, 5) * 0.05);

    return Math.min(1, score); // Cap at 1.0
  }

  /**
   * Retrieves knowledge entries by tag
   */
  retrieveByTag(
    tag: string,
    knowledgeBase: KnowledgeEntry[],
    maxResults = 10,
  ): KnowledgeEntry[] {
    return knowledgeBase
      .filter((entry) => entry.tags.includes(tag))
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, maxResults);
  }

  /**
   * Retrieves knowledge entries by title similarity
   */
  retrieveByTitle(
    title: string,
    knowledgeBase: KnowledgeEntry[],
    maxResults = 5,
  ): KnowledgeEntry[] {
    const normalizedTitle = title.toLowerCase();

    return knowledgeBase
      .map((entry) => ({
        entry,
        similarity: this.calculateTitleSimilarity(
          normalizedTitle,
          entry.title.toLowerCase(),
        ),
      }))
      .filter((item) => item.similarity > 0.3) // Minimum similarity threshold
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxResults)
      .map((item) => item.entry);
  }

  /**
   * Calculates simple similarity between two titles
   */
  private calculateTitleSimilarity(title1: string, title2: string): number {
    // Simple implementation - can be enhanced with more sophisticated algorithms
    if (title1 === title2) return 1;
    if (title1.includes(title2) || title2.includes(title1)) return 0.8;

    const terms1 = this.extractTerms(title1);
    const terms2 = this.extractTerms(title2);

    const commonTerms = terms1.filter((term) =>
      terms2.some((t2) => t2.includes(term) || term.includes(t2)),
    ).length;

    return commonTerms / Math.max(terms1.length, terms2.length, 1);
  }
}
