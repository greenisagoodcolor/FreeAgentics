/**
 * ExtractedBelief Data Structures and Interfaces
 * Supports Bayesian update tracking, confidence score evolution per ADR-005
 * Integrates with /inference/engine and agent belief systems
 */

import { Agent } from './agents';

/**
 * Confidence score with timestamp for tracking evolution
 */
export interface ConfidenceSnapshot {
  value: number; // 0.0 to 1.0
  timestamp: string; // ISO string
  reason?: string; // Why confidence changed
  evidence?: string[]; // Supporting evidence IDs
}

/**
 * Bayesian update information for belief revision
 */
export interface BayesianUpdate {
  id: string;
  timestamp: string;
  priorProbability: number;
  likelihood: number;
  posteriorProbability: number;
  evidence: {
    type: 'observation' | 'testimony' | 'inference' | 'correction';
    content: string;
    source: string; // Agent ID or external source
    reliability: number; // 0.0 to 1.0
  };
  method: 'standard' | 'jeffrey' | 'adams' | 'custom';
  parameters?: Record<string, any>;
}

/**
 * Belief category and classification
 */
export enum BeliefCategory {
  FACTUAL = 'factual',
  PROCEDURAL = 'procedural',
  CAUSAL = 'causal',
  NORMATIVE = 'normative',
  PREDICTIVE = 'predictive',
  EPISTEMIC = 'epistemic',
  SOCIAL = 'social',
  TEMPORAL = 'temporal'
}

/**
 * Belief origin and extraction context
 */
export interface BeliefOrigin {
  type: 'conversation' | 'observation' | 'inference' | 'memory' | 'external';
  conversationId?: string;
  messageId?: string;
  agentId: string;
  extractionMethod: string;
  extractionConfidence: number;
  context: string;
  timestamp: string;
}

/**
 * Duplicate detection and similarity metadata
 */
export interface DuplicateMetadata {
  similarBeliefs: {
    beliefId: string;
    cosineSimilarity: number;
    semanticSimilarity: number;
    structuralSimilarity: number;
    overallSimilarity: number;
  }[];
  mergeStatus: 'none' | 'candidate' | 'merged' | 'ignored';
  mergeHistory: {
    timestamp: string;
    action: 'identified' | 'merged' | 'split' | 'ignored';
    targetBeliefId?: string;
    reason: string;
    performedBy: string; // Agent or user
  }[];
}

/**
 * Belief validation and verification status
 */
export interface ValidationStatus {
  isValidated: boolean;
  validationMethod?: 'automatic' | 'peer_review' | 'expert_review' | 'external_source';
  validatedBy?: string[];
  validationTimestamp?: string;
  validationConfidence?: number;
  contradictions: {
    beliefId: string;
    contradictionType: 'logical' | 'empirical' | 'temporal' | 'source';
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
  }[];
}

/**
 * Main ExtractedBelief interface
 */
export interface ExtractedBelief {
  // Core identification
  id: string;
  content: string;
  summary: string;
  category: BeliefCategory;
  tags: string[];
  
  // Agent attribution
  originAgent: {
    id: string;
    name: string;
    class: string;
    avatar?: string;
  };
  involvedAgents: string[]; // Other agents who share or discussed this belief
  
  // Confidence and evolution
  currentConfidence: number; // 0.0 to 1.0
  confidenceHistory: ConfidenceSnapshot[];
  bayesianUpdates: BayesianUpdate[];
  
  // Origin and context
  origin: BeliefOrigin;
  relatedConversations: string[];
  relatedBeliefs: string[];
  
  // Duplicate detection
  duplicateMetadata: DuplicateMetadata;
  
  // Validation
  validation: ValidationStatus;
  
  // Temporal information
  createdAt: string;
  lastUpdated: string;
  lastAccessed: string;
  
  // Semantic and structural properties
  semanticEmbedding?: number[]; // Vector embedding for similarity
  logicalStructure?: {
    subject: string;
    predicate: string;
    object: string;
    modality?: string; // 'necessary', 'possible', 'contingent'
    temporality?: string; // 'past', 'present', 'future', 'eternal'
  };
  
  // Evidence and support
  supportingEvidence: {
    id: string;
    type: 'observation' | 'testimony' | 'document' | 'reasoning';
    content: string;
    source: string;
    reliability: number;
    timestamp: string;
  }[];
  
  // Metadata
  importance: number; // 0.0 to 1.0
  complexity: number; // 0.0 to 1.0
  novelty: number; // 0.0 to 1.0
  consensus: number; // How much agents agree: 0.0 to 1.0
  
  // System metadata
  version: number;
  archived: boolean;
  archivedReason?: string;
  archivedAt?: string;
}

/**
 * Belief search and filter criteria
 */
export interface BeliefSearchCriteria {
  query?: string;
  categories?: BeliefCategory[];
  agentIds?: string[];
  conversationIds?: string[];
  confidenceRange?: {
    min: number;
    max: number;
  };
  timeRange?: {
    start: string;
    end: string;
  };
  tags?: string[];
  hasContradictions?: boolean;
  validationStatus?: ('validated' | 'unvalidated' | 'disputed')[];
  importanceRange?: {
    min: number;
    max: number;
  };
  noveltyRange?: {
    min: number;
    max: number;
  };
  consensusRange?: {
    min: number;
    max: number;
  };
  mergeStatus?: DuplicateMetadata['mergeStatus'][];
  archived?: boolean;
}

/**
 * Belief search results with pagination
 */
export interface BeliefSearchResults {
  beliefs: ExtractedBelief[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
  facets: {
    categories: Record<BeliefCategory, number>;
    agents: Record<string, number>;
    tags: Record<string, number>;
    confidenceDistribution: number[];
    timeDistribution: Record<string, number>;
  };
  searchTime: number; // milliseconds
}

/**
 * Belief evolution timeline entry
 */
export interface BeliefTimelineEntry {
  id: string;
  beliefId: string;
  timestamp: string;
  type: 'created' | 'updated' | 'confidence_changed' | 'evidence_added' | 'contradicted' | 'merged' | 'validated' | 'archived';
  description: string;
  details: Record<string, any>;
  agentId?: string;
  conversationId?: string;
  oldValue?: any;
  newValue?: any;
}

/**
 * Belief analytics and statistics
 */
export interface BeliefAnalytics {
  totalBeliefs: number;
  beliefsByCategory: Record<BeliefCategory, number>;
  beliefsByAgent: Record<string, number>;
  averageConfidence: number;
  confidenceDistribution: number[];
  beliefEvolutionRate: number; // beliefs per hour
  duplicateDetectionRate: number;
  validationRate: number;
  consensusMetrics: {
    averageConsensus: number;
    highConsensusBeliefs: number;
    lowConsensusBeliefs: number;
    disputedBeliefs: number;
  };
  temporalTrends: {
    timestamp: string;
    beliefCount: number;
    averageConfidence: number;
    validationRate: number;
  }[];
}

/**
 * Belief interaction events for real-time updates
 */
export interface BeliefEvent {
  type: 'belief_created' | 'belief_updated' | 'belief_merged' | 'belief_validated' | 'belief_contradicted' | 'belief_archived';
  beliefId: string;
  agentId?: string;
  timestamp: string;
  payload: Record<string, any>;
  conversationId?: string;
}

/**
 * Belief merge operation
 */
export interface BeliefMergeOperation {
  id: string;
  sourceBelief: ExtractedBelief;
  targetBelief: ExtractedBelief;
  similarityScore: number;
  mergeStrategy: 'content_union' | 'highest_confidence' | 'most_recent' | 'evidence_weighted' | 'manual';
  resultingBelief: ExtractedBelief;
  performedBy: string;
  timestamp: string;
  approved: boolean;
  approvedBy?: string;
  approvalTimestamp?: string;
}

/**
 * Belief contradiction analysis
 */
export interface BeliefContradiction {
  id: string;
  belief1: ExtractedBelief;
  belief2: ExtractedBelief;
  contradictionType: 'logical' | 'empirical' | 'temporal' | 'source';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  detectionMethod: string;
  detectionConfidence: number;
  resolution?: {
    strategy: 'belief1_precedence' | 'belief2_precedence' | 'synthesis' | 'context_dependent' | 'unresolved';
    reason: string;
    resolvedBy: string;
    timestamp: string;
  };
  createdAt: string;
  lastUpdated: string;
}

/**
 * Export functions for type validation
 */
export function createEmptyBelief(): Partial<ExtractedBelief> {
  return {
    content: '',
    summary: '',
    category: BeliefCategory.FACTUAL,
    tags: [],
    currentConfidence: 0.5,
    confidenceHistory: [],
    bayesianUpdates: [],
    involvedAgents: [],
    relatedConversations: [],
    relatedBeliefs: [],
    supportingEvidence: [],
    importance: 0.5,
    complexity: 0.5,
    novelty: 0.5,
    consensus: 0.5,
    version: 1,
    archived: false
  };
}

export function validateBelief(belief: Partial<ExtractedBelief>): string[] {
  const errors: string[] = [];
  
  if (!belief.content || belief.content.trim().length === 0) {
    errors.push('Belief content is required');
  }
  
  if (!belief.summary || belief.summary.trim().length === 0) {
    errors.push('Belief summary is required');
  }
  
  if (!belief.category || !Object.values(BeliefCategory).includes(belief.category)) {
    errors.push('Valid belief category is required');
  }
  
  if (belief.currentConfidence !== undefined && (belief.currentConfidence < 0 || belief.currentConfidence > 1)) {
    errors.push('Confidence must be between 0 and 1');
  }
  
  if (!belief.originAgent?.id) {
    errors.push('Origin agent ID is required');
  }
  
  return errors;
} 